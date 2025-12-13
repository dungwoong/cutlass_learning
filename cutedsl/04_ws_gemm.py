import argparse
from typing import Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch

import cutlass
from cutlass import Boolean, Int32, const_expr
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait, PipelineState, PipelineUserType
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils


"""
Simple warp-specialized WGMMA kernel

Kernel
- Simple, warp-specialized GEMM. Producer warpgroup uses TMA, Consumer warpgroup uses WGMMA
- Save back from RMEM directly to GMEM

Notes
- The producer if structure is important
    - Entire producer warpgroup must enter the if statement for warpgroup_reg_dealloc, else it waits indefinitely(setmaxnreg.{ind/dec}.SYNC)
    - Then, only the producer warp can enter the mainloop, otherwise the barrier arrivals will be wrong
- Consumer is basic
"""


@cute.jit
def print0(x):
    tidx, _, _ = cute.arch.thread_idx() # threadidx.x, y, z
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

@cute.jit
def print_tidx(x, idx):
    tidx, _, _ = cute.arch.thread_idx() # threadidx.x, y, z
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == idx and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == idx and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)


class GemmKernel1:
    def __init__(self):
        self.acc_dtype = cutlass.Float32

        self.atom_layout_mnk = (2, 1, 1) # layout of the WGMMAs
        self.mma_warpgroups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.mma_warpgroups + 1) * self.num_threads_per_warp_group
        self.tile_shape_mnk = (128, 256, 1) # WGMMA is m64k16, so we parallelize/iterate accordingly

        self.tiled_mma = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.a_dtype, self.b_dtype, self.c_dtype = None, None, None
        self.a_layout, self.b_layout, self.c_layout = None, None, None
        self.a_smem_layout, self.b_smem_layout = None, None
        self.num_stages = 2 # buffer stages for producer-consumer setup
        self.producer_warp_id = self.mma_warpgroups * 4

        self.producer_regs, self.consumer_regs = (40, 232)
    
    def _setup(self):
        self._check_tile_shapes()
        self._create_mma_atom()
        self._get_smem_layouts()
    
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup()

        # TMA load ThrID is just 1, since 1 thread launches it
        # the TV layout is just (1, tile size). You later compose with some other tile to load that tile
        tma_atom_a, tma_tensor_a = self._create_tma_load_and_tensors(a, self.a_smem_layout, (self.tile_shape_mnk[0], self.tile_shape_mnk[2]))
        tma_atom_b, tma_tensor_b = self._create_tma_load_and_tensors(b, self.b_smem_layout, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]))
        tensor_c = c

        grid = self._compute_grid(c, self.tile_shape_mnk)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2] # just one empty, one full I guess
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage

        self.kernel(tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tensor_c, self.tiled_mma, self.a_smem_layout, self.b_smem_layout).launch(grid=grid, block=self.threads_per_cta, stream=stream)
    
    @cute.kernel
    def kernel(self, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor, 
               tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
               mC_mnl: cute.Tensor,
               tiled_mma: cute.TiledMma,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
               ):
        
        # ###########################################################################################
        # logistics
        # ###########################################################################################
        # Indices
        bidx, bidy, _ = cute.arch.block_idx()
        tile_coord_mnk = (bidx, bidy, None) # no batch dim, just use x and y to tile the tensor, simple rasterization
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # ###########################################################################################
        # Prefetch TMA descriptor
        # We'd normally do this earlier on
        # ###########################################################################################
        # Descriptor is stored in GMEM and fetched only by the producer warp
        if warp_idx == self.producer_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ###########################################################################################
        # allocate SMEM
        # ###########################################################################################
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ###########################################################################################
        # Get the pipeline ready
        # ###########################################################################################
        # All of this could maybe be put into the JIT function
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr() # pointer to the barriers
        consumer_arrive_count = tiled_mma.size // cute.arch.WARP_SIZE # num mma warps
        
        # Get TMA copy bytes
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        mainloop_pipeline = self.make_ab_pipeline(consumer_arrive_count, mainloop_pipeline_array_ptr, tma_copy_bytes)
        pipeline_init_arrive()
        pipeline_init_wait()


        # ###########################################################################################
        # Configure the G2S
        # ###########################################################################################

        # 1. Get shared tensors
        # Composed layouts are inner(swizzle) o Offset o outer(layout)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # 2. Partition the global tensors
        # (bM, bK, RestK)
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnk, proj=(1, None, 1)
        )

        # (bN, bK, RestK)
        gB_nkl = gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnk, proj=(None, 1, 1)
        )

        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnk, proj=(1, 1, None)
        )

        # ###########################################################################################
        # Partition shared tensor for TMA load A/B
        # ###########################################################################################
        # (TMA, {stages, k})
        tAsA, tAgA_mkl = self.tma_partition(tma_atom_a, sA, gA_mkl)
        tBsB, tBgB_nkl = self.tma_partition(tma_atom_b, sB, gB_nkl)

        k_tile_count = cute.size(gA_mkl, mode=[2])

        if warp_idx >= self.producer_warp_id: # Entire warpgroup enters producer
            cute.arch.warpgroup_reg_dealloc(self.producer_regs)
            if warp_idx == self.producer_warp_id: # Only producer warps do fetches
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_stages
                )
                self.produce_mainloop(k_tile_count, mainloop_pipeline, ab_producer_state, tAgA_mkl, tAsA, tBgB_nkl, tBsB, tma_atom_a, tma_atom_b)

        if warp_idx < self.producer_warp_id:
            cute.arch.warpgroup_reg_alloc(self.consumer_regs)
            thr_mma = tiled_mma.get_slice(tidx) # Slice by tidx to save back to GMEM
            
            tCgC = thr_mma.partition_C(gC_mnl) # Register indices, in GMEM
            acc_shape = tCgC.shape
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA) # (r, c), k_blocks, pipe_stages
            tCrB = tiled_mma.make_fragment_B(tCsB) # These are descriptor slices in SMEM
            
            mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            self.consume(tiled_mma, accumulators, k_tile_count, mainloop_pipeline, mma_consumer_state, tCrA, tCrB)
            
            # Save back accumulators --> tCgC
            epi_accumulator = cute.make_rmem_tensor_like(accumulators, self.c_dtype)
            epi_accumulator.store(accumulators.load().to(mC_mnl.element_type))
            cute.autovec_copy(epi_accumulator, tCgC)


    @cute.jit
    def produce_mainloop(self, k_tile_count: cutlass.Int32, 
                         ld_pipeline: cutlass.pipeline.PipelineAsync, 
                         producer_state: cutlass.pipeline.PipelineState,
                         tAgA_mkl: cute.Tensor, tAsA: cute.Tensor,
                         tBgB_nkl: cute.Tensor, tBsB: cute.Tensor,
                         tma_atom_a: cute.CopyAtom, tma_atom_b: cute.CopyAtom):
        for _ in cutlass.range(k_tile_count, unroll=1, unroll_full=False):
            tma_bar_ptr = ld_pipeline.producer_get_barrier(producer_state) # need to re-fetch this everytime based on circular buffer
            ld_pipeline.producer_acquire(producer_state)
            tAgA_k = tAgA_mkl[(None, producer_state.count)]
            tAsA_pipe = tAsA[(None, producer_state.index)]

            tBgB_k = tBgB_nkl[(None, producer_state.count)]
            tBsB_pipe = tBsB[(None, producer_state.index)]

            cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar_ptr)
            cute.copy(tma_atom_b, tBgB_k, tBsB_pipe, tma_bar_ptr=tma_bar_ptr)
            ld_pipeline.producer_commit(producer_state)
            producer_state.advance()
        ld_pipeline.producer_tail(producer_state)
        return producer_state

    @cute.jit
    def consume(self, 
                tiled_mma: cute.TiledMma,
                accumulators: cute.Tensor,
                k_tile_count: cutlass.Int32, cons_pipeline: cutlass.pipeline.PipelineAsync, 
                consumer_state: cutlass.pipeline.PipelineState,
                tCrA: cute.Tensor, tCrB: cute.Tensor
                ):
        num_k_blocks = cute.size(tCrA, mode=[2])
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        for _ in cutlass.range(k_tile_count, unroll=1):
            cons_pipeline.consumer_wait(consumer_state)
            cute.nvgpu.warpgroup.fence()
            
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, consumer_state.index)
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            cons_pipeline.consumer_release(consumer_state)
            consumer_state.advance()
    
    def tma_partition(self, tma_atom: cute.CopyAtom, sMatrix: cute.Tensor, gMatrix: cute.Tensor):
        # https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html
        s_tma = cute.group_modes(sMatrix, 0, 2)
        g_tma = cute.group_modes(gMatrix, 0, 2)

        # ((m, n), rest) --> (TMA, rest)
        shared_layout, global_layout = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0, # Since no multicasting, we use 0 and (1) for CTA-related stuff
            cute.make_layout(1),
            s_tma,
            g_tma,
        )
        return shared_layout, global_layout
    
    def make_ab_pipeline(self, consumer_arrive_count: int, mbar_ptr: cute.Pointer, tma_copy_bytes: int):
        """
        Make pipeline
        """
        producer_count = 1
        # TODO can't we replace this with something else
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_count)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_count)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr, 
            num_stages=self.num_stages, 
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=None # this is used for clusters
        )
    
    def _create_tma_load_and_tensors(self, t: cute.Tensor, smem_layout_staged: cute.ComposedLayout, smem_tile: tuple[int, int]) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            t,
            smem_layout,
            smem_tile,
        )
        return tma_atom, tma_tensor

    def _check_tile_shapes(self):
        # This is basic tile shapes that are possible, I think we can do more.
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")
    
    def _create_mma_atom(self):
        # make the MMA first, then get it's K dim and figure out what your tile will be based on that
        # but the k dim should typically be 16
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk, # how we will layout the atoms
            tiler_mn=(64, self.tile_shape_mnk[1]) # shape of each MMA
        )

        mma_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (self.tile_shape_mnk[0], self.tile_shape_mnk[1], mma_k * mma_inst_tile_k)
    
    def _get_smem_layouts(self):
        """
        Return smem layouts for A and B
        """
        (self.a_smem_layout, 
         self.b_smem_layout) = self._create_smem_layouts(
             self.tile_shape_mnk,
             self.a_dtype, self.a_layout, self.b_dtype, 
             self.b_layout, self.c_dtype, self.c_layout, load_stage=self.num_stages)
    
    @staticmethod
    def _create_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        c_dtype: type[cutlass.Numeric], # unused since we save R2G
        c_layout: utils.LayoutEnum,
        load_stage: int,
        ):
        a_smem_layout = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, load_stage
        )
        b_smem_layout = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, load_stage
        )
        return a_smem_layout, b_smem_layout

    @staticmethod
    def _compute_grid(t: cute.Tensor, tile_shape_mnk: tuple[int, int, int]):
        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(t, tiler=c_shape)
        return cute.get(gc.layout, mode=[1]).shape

m = 1024
n = 512
k = 256
a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
c = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
ref = a @ b.t()
convert_from_dlpack = lambda tensor: (
    from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1)
    )
)
a_cute, b_cute, c_cute = [convert_from_dlpack(x) for x in (a, b, c)]
current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
gemm = GemmKernel1()
gemm(a_cute, b_cute, c_cute, current_stream)
# torch.cuda.synchronize()
print(ref)
print(c)

n_incorrect = c.numel() - ((c - ref).abs() < 0.001).sum()
print('n_incorrect :', n_incorrect)