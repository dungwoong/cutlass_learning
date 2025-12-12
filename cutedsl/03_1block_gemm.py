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


# 2 warpgroups stacked on top of each other to do 64x256 MMA let's just go with that for now.
# No pipelining at all, just one warpgroup where warp0 does the load
# sanity check to make sure I'm loading/multiplying properly
# In the future, I should try to load to rmem before doing the multiplication
# capabilities: https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-operation-wgmma-mma-async

# - make sure G2S is right
# - make sure accumulator is right
# - see if I can remove the syncthreads at the end(seems like it should be fine)
# - check the results
# - switch to randn

"""
Setup
- Each CTA is just two warpgroups stacked on top of each other to do 64x256 WGMMA
- 8 warps(2WG), warp 0 does TMA load and all warps do the WGMMA
- Single staged buffer

Learnings
- Make sure global layout is right, based on the CTA tilers
- If the pipeline is erroring, look at the initialization
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

class GemmLoads:
    def __init__(self):
        # User-configurable
        self.atom_layout_mnk = (2, 1, 1)
        self.mma_warpgroups = math.prod(self.atom_layout_mnk)
        self.n_warpgroup_threads = 128
        self.threads_per_cta = self.n_warpgroup_threads * self.mma_warpgroups

        # Nothing to do with the #consumers, they will tile this I think
        self.tile_shape_mnk = (128, 256, 1)
        self.buffer_align_bytes = 1024
        self.acc_type = cutlass.Float32

        # Decided when JIT-compiling
        self.tiled_mma = None
        self.shared_storage = None
        self.a_dtype, self.b_dtype, self.c_dtype = None, None, None
        self.a_layout, self.b_layout, self.c_layout = None, None, None
        self.a_smem_layout, self.b_smem_layout = None, None

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream):
        self.populate_dtypes_and_layouts(a, b, c)
        self.populate_mma_atom()
        self.populate_smem_layouts()
        self.populate_shared_storage()
        grid = self._get_grid(c, self.tile_shape_mnk)
        
        tma_atom_a, tma_tensor_a = self._get_tma_load_and_tensors(a, self.a_smem_layout, (self.tile_shape_mnk[0], self.tile_shape_mnk[2]))
        tma_atom_b, tma_tensor_b = self._get_tma_load_and_tensors(b, self.b_smem_layout, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]))
        tensor_c = c
        print('tma_atom_a :', tma_atom_a)
        print('tma_atom_b :', tma_atom_b)
        print('tma_tensor_a :', tma_tensor_a)
        print('tma_tensor_b :', tma_tensor_b)
        print('tensor c :', tensor_c)

        self.kernel(
            tma_atom_a, tma_atom_b, 
            tma_tensor_a, tma_tensor_b, tensor_c, 
            self.tiled_mma, 
            self.a_smem_layout, self.b_smem_layout).launch(grid=grid, block=self.threads_per_cta, stream=stream)

    @cute.kernel
    def kernel(
        self, 
        tma_atom_a: cute.CopyAtom, tma_atom_b: cute.CopyAtom,
        mA_mkl: cute.Tensor, mB_nkl: cute.Tensor, mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout):
        
        bidx, bidy, _ = cute.arch.block_idx()
        tile_coord_mnk = (bidx, bidy, None)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        pipeline_ws = self.make_ab_pipeline(mbar_ptr, a_smem_layout_staged, b_smem_layout_staged)
        pipeline_init_arrive()
        pipeline_init_wait()

        # Partition global memory
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnk, proj=(1, None, 1)
        )

        gB_mkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnk, proj=(None, 1, 1)
        )

        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnk, proj=(1, 1, None)
        )
        k_tile_count = cute.size(gA_mkl, mode=[2])

        # Get SMEM matrices
        # inner(swizzle) o Offset o outer(Layout)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )

        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # Get TMA-style partitions(TMA, iter)
        tAsA, tAgA_mkl = self.tma_partition(tma_atom_a, sA, gA_mkl)
        tBsB, tBgB_nkl = self.tma_partition(tma_atom_b, sB, gB_mkl)

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )

        # MMA setup
        # I'm not even sure if this warpgroup thread0 even does anything but ok
        warp_group_thread0 = cute.make_layout(
                self.mma_warpgroups, stride=self.n_warpgroup_threads
        )
        thr_mma = tiled_mma.get_slice(tidx)
        tCgC = thr_mma.partition_C(gC_mnl) # register indices in GMEM
        print('tCgC :', tCgC)
        acc_shape = tCgC.shape
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_type)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA) # (r, c), k_blocks, pipe_stages
        tCrB = tiled_mma.make_fragment_B(tCsB)
        print('tCrA :', tCrA)
        print('tCrB :', tCrB)

        num_k_blocks = cute.size(tCrA, mode=[2])
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        print0(f'outer k: {k_tile_count}')
        print0(f'inner k: {num_k_blocks}')
        for _ in cutlass.range(k_tile_count, unroll=1):
            if warp_idx == 0:
                print0('loading')
                pipeline_ws.producer_acquire(producer_state)
                
                tma_mbar_ptr = pipeline_ws.producer_get_barrier(producer_state)
                tAgA_k = tAgA_mkl[(None, producer_state.count)] # (TMA, k)
                tAsA_pipe = tAsA[(None, producer_state.index)]
                tBgB_k = tBgB_nkl[(None, producer_state.count)]
                tBsB_pipe = tBsB[(None, producer_state.index)]
                cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_mbar_ptr)
                cute.copy(tma_atom_b, tBgB_k, tBsB_pipe, tma_bar_ptr=tma_mbar_ptr)
                
                pipeline_ws.producer_commit(producer_state)


            # Everyone does WGMMA now
            print0('doing MMA')
            pipeline_ws.consumer_wait(consumer_state)
            # print0(sA) # this should be all ones
            # print0(sB)
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll=1):
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
            cute.nvgpu.warpgroup.commit_group() # wgmma commit group
            # print0('Committed MMA')
            cute.nvgpu.warpgroup.wait_group(0) # immediately wait
            # print0('Waited')
            pipeline_ws.consumer_release(consumer_state)
            # print0('Released')
            producer_state.advance()
            consumer_state.advance()
            # print0('done one iter')
        
        # cute.arch.barrier() # syncthreads TODO do we need this?
        # epilogue store
        print0('Accumulators for thread 0:')
        print0(accumulators)
        epi_accumulator = cute.make_rmem_tensor_like(accumulators, self.c_dtype)
        epi_accumulator.store(accumulators.load().to(self.c_dtype))
        print0('epi accumulators')
        print0(epi_accumulator)
        # epi_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        cute.autovec_copy(epi_accumulator, tCgC)
        return
    
    # ####################################################
    # Compile-time device-side
    # ####################################################
    def make_ab_pipeline(self, mbar_ptr: cute.Pointer, a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout) -> pipeline.PipelineAsync:
        a_smem_layout_single = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout_single = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout_single) + cute.size_in_bytes(self.b_dtype, b_smem_layout_single)
        
        num_producers = 1
        num_consumers = self.mma_warpgroups * 4
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_producers)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_consumers)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=1,
            tx_count=tma_copy_bytes,
            producer_group=producer_group,
            consumer_group=consumer_group,
        )
    
    def tma_partition(self, tma_atom: cute.CopyAtom, sMatrix: cute.Tensor, gMatrix: cute.Tensor):
        # TMA does first mode. These matrices are already tiled
        s_tma = cute.group_modes(sMatrix, 0, 2) # (r, c), rest
        g_tma = cute.group_modes(gMatrix, 0, 2)

        shared_layout, global_layout = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0, # no cluster
            cute.make_layout(1),
            s_tma,
            g_tma,
        )
        return shared_layout, global_layout

    # ####################################################
    # Compile-time host-side Calculations
    # ####################################################
    def _get_tma_load_and_tensors(self, t: cute.Tensor, smem_layout_staged: cute.ComposedLayout, smem_tile: tuple[int, int]) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            t,
            smem_layout,
            smem_tile,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def _get_grid(t: cute.Tensor, tile_shape_mnk: tuple[int, int, int]):
        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(t, tiler=c_shape)
        return cute.get(gc.layout, mode=[1]).shape
        

    # ####################################################
    # COMPILE-TIME CHECKS/Populating
    # ####################################################
    @cute.jit
    def populate_shared_storage(self):
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, 2] # you still need this
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout)], self.buffer_align_bytes]
        self.shared_storage = SharedStorage

    def populate_dtypes_and_layouts(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        print('Layouts')
        print(f'{self.a_layout=}, {self.b_layout=}, {self.c_layout=}')
    
    @cute.jit
    def populate_mma_atom(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_type,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1])
        )
        mma_k = 16
        mma_inst_tile_k = 1 # 4 instructions along the k-tile
        self.tile_shape_mnk = (self.tile_shape_mnk[0], self.tile_shape_mnk[1], mma_inst_tile_k * mma_k)
        print('Tiled MMA :', self.tiled_mma)

    def populate_smem_layouts(self):
        (self.a_smem_layout,
         self.b_smem_layout) = self._get_smem_layouts(
             self.tile_shape_mnk,
             self.a_dtype, self.a_layout,
             self.b_dtype, self.b_layout,
         )
        print('A SMEM :', self.a_smem_layout)
        print('B SMEM :', self.b_smem_layout)
    
    @staticmethod
    def _get_smem_layouts(
            tile_shape_mnk: tuple[int, int, int],
            a_dtype: Type[cutlass.Numeric],
            a_layout: utils.LayoutEnum,
            b_dtype: Type[cutlass.Numeric],
            b_layout: utils.LayoutEnum):
        # it's an SMEM layout atom, and then tile to shape.
        # we can look at it later.
        a_smem_layout = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, 1
        )
        b_smem_layout = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, 1
        )
        return a_smem_layout, b_smem_layout

# easy initial test
m = 512 # at least 128
n = 512 # at least 256
k = 32
a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
# a[:, 0] = 1
# b[:, 0] = 1
# print(a @ b.t()) # this should be a ones matrix
c = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
convert_from_dlpack = lambda tensor: (
    from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1)
    )
)
a_cute, b_cute, c_cute = [convert_from_dlpack(x) for x in (a, b, c)]
print(a_cute)
print(b_cute)
current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
gemm = GemmLoads()
gemm(a_cute, b_cute, c_cute, current_stream)
torch.cuda.synchronize()

expected = a @ b.t()
print('Output:')
print(c)
print('Expected:')
print(expected)

n_incorrect = c.numel() - ((c - expected).abs() < 0.001).sum()
print('n_incorrect :', n_incorrect)
