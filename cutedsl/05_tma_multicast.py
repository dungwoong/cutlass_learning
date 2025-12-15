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
Kernel
- Two CTAs(2, 1, 1) in one cluster
- Attempt to multicast a TMA load to both CTAs

What to do:
- add a cluster size
- when you make the pipeline, add a CTA layout vmnk to it(this just makes sure all CTAs in the cluster sync on pipeline init)
- for the copy atom, use a multicast op, specify a num_multicast
- you have to make a mcast mask(details are still a bit hazy on this, but we can just copy existing work for GEMM)


I think
- when you tma_partition, you put in the CTA coord as well as the CTA layout
- you then make a multicast mask to specify which CTAs to multicast to
- when you make the TMA atom you need to specify the multicast number
- you can then copy with a multicast mask
"""

class MulticastCopy:
    def __init__(self):
        self.mcast_num = 2

        self.input_dtype = None
        self.g_layout = None
        self.a_smem_layout_staged = None
        self.shared_storage = None
        self.tile_shape = (128, 32)
        self.cta_layout_mn = (2, 1)
    
    @cute.jit
    def __call__(self, a: cute.Tensor, stream: cuda.CUstream):
        self.input_dtype = a.element_type
        self.g_layout = utils.LayoutEnum.from_tensor(a)
        self.make_a_smem_layout(self.input_dtype, self.g_layout, self.tile_shape)
        tma_atom, tma_tensor = self._make_copy(a, self.a_smem_layout_staged, self.tile_shape, mcast_dim=2)

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, 2]
            sA: cute.struct.Align[cute.struct.MemRange[self.input_dtype, cute.cosize(self.a_smem_layout_staged)], 1024]
        self.shared_storage = SharedStorage

        grid = (2, 1, 1)
        block = (32, 1, 1)
        cluster = (self.mcast_num, 1, 1)
        self.kernel(a, (2, 1), tma_atom, tma_tensor, self.a_smem_layout_staged).launch(grid=grid, block=block, cluster=cluster, stream=stream)
        return

    @cute.kernel
    def kernel(self, a: cute.Tensor, cta_layout_mk: tuple[int, int], tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, a_smem_layout_staged: cute.ComposedLayout):
        bidx, bidy, _ = cute.arch.block_idx()
        tile_coord_mnk = (bidx, bidy, None)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar = storage.mbar.data_ptr()
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.input_dtype, a_smem_layout)
        pipe_ = self.make_ab_pipeline(1, mbar, tma_copy_bytes)
        pipeline_init_arrive()
        pipeline_init_wait()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle = a_smem_layout_staged.inner
        )

        gA_tiled = cute.local_tile(tma_tensor, self.tile_shape, (0, None))

        # linear id of the CTA in the cluster
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cta_layout_mk_layout = cute.make_layout(self.cta_layout_mn)

        # this just goes from 1D coord --> 2D layout(?)
        cluster_coord_mn = cta_layout_mk_layout.get_flat_coord(cta_rank_in_cluster)
        tAsA, tAgA = self.tma_partition(cluster_coord_mn, tma_atom, sA, gA_tiled)

        # you just give it a layout, and it slices along a mode, in this case mode0
        # - default if no multicasting is 0
        mcast_mask = cute.make_layout_image_mask(
            cta_layout_mk_layout, cluster_coord_mn, mode=0
        )

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )

        if warp_idx == 0:
            pipe_.producer_acquire(mainloop_producer_state)
            tAgA_k = tAgA[(None, 0)]
            tAsA_pipe = tAsA[(None, 0)]
            cute.copy(
                tma_atom,
                tAgA_k,
                tAsA_pipe,
                tma_bar_ptr=pipe_.producer_get_barrier(mainloop_producer_state),
                mcast_mask=mcast_mask
            )
            pipe_.producer_commit(mainloop_producer_state)
        pipe_.consumer_wait(mainloop_consumer_state)

        # Do one or the other, otherwise the resulting print statement is weird
        # if tidx == 0 and bidx == 0:
        #    cute.print_tensor(sA)
        if tidx == 0 and bidx == 1:
            cute.print_tensor(sA)

    
    def tma_partition(self, cluster_coord, tma_atom: cute.CopyAtom, sMatrix: cute.Tensor, gMatrix: cute.Tensor):
        s_tma = cute.group_modes(sMatrix, 0, 2)
        g_tma = cute.group_modes(gMatrix, 0, 2)

        # ((m, n), rest) --> (TMA, rest)
        shared_layout, global_layout = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            cluster_coord[0],
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
            num_stages=1, 
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 2, 1, 1)) # this is used for clusters
        )

    def _make_copy(self, tensor: cute.Tensor, 
                   smem_layout_staged: cute.ComposedLayout, 
                   smem_tile: tuple[int, int], mcast_dim: int):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim
        )
        return tma_atom, tma_tensor
    
    def make_a_smem_layout(self, a_dtype: type[cutlass.Numeric], 
                           a_layout: utils.LayoutEnum,
                           tile_shape: tuple[int, int]):
        tile_shape_mnk = (tile_shape[0], 1, tile_shape[1])
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            1
        )
    

data = [i for i in range(128 * 32)]
a = torch.tensor(data, dtype=torch.int32, device='cuda').reshape(128, 32)
print(a)
convert_from_dlpack = lambda tensor: (
    from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1)
    )
)
current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
a_cute = convert_from_dlpack(a)
mc = MulticastCopy()
mc(a_cute, current_stream)
