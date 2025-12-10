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
A shell of producer consumer barrier setup
- Only the first thread in each warp will arrive at each barrier. Make sure you get the right arrival count
- Only one warp should be responsible for the TMA load
"""

@cute.jit
def print0(x):
    tidx, _, _ = cute.arch.thread_idx() # threadidx.x, y, z
    bidx, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 0 and bidx == 0:
            cute.print_tensor(x)
    else:
        if tidx == 0 and bidx == 0:
            cute.printf(x)

@cute.jit
def print32(x):
    tidx, _, _ = cute.arch.thread_idx() # threadidx.x, y, z
    bidx, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 32 and bidx == 0:
            cute.print_tensor(x)
    else:
        if tidx == 32 and bidx == 0:
            cute.printf(x)


class BarrierLifetime:
    def __init__(self):
        # 4 warps total, warp 0 is the "producer", warps 1-3 are the "consumers"
        self.nconsumers = 3
        self.nproducers = 1
        self.shared_storage = None
        self.num_stages = 2
        self.n_iters = 3
    
    @cute.jit
    def __call__(self, stream: cuda.CUstream):
        @cute.struct
        class SharedStorage:
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        self.shared_storage = SharedStorage
        self.kernel().launch(grid=1, block=128, stream=stream)
    
    @cute.kernel
    def kernel(self):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mbar_ptr = storage.pipeline_array.data_ptr()

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.nproducers)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.nconsumers)
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self.num_stages,
            tx_count=0,
            producer_group=producer_group,
            consumer_group=consumer_group
        )

        # you should fence after, otherwise threads could try to arrive before thread0 initializes the barrier(?)
        # the quack GEMM example doesn't do this though
        pipeline_init_arrive()
        pipeline_init_wait()

        if warp_idx == 0:
            # producer

            # don't worry about the pipeline states, it's just consistent with the internal representation I guess
            producer_state = pipeline.make_pipeline_state(
                # phase starts at 1
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            for _ in cutlass.range(self.n_iters, unroll=1):
                pipe.producer_acquire(producer_state)  # wait at empty, then arrive at full
                print0('Producer Acquire')
                pipe.producer_commit(producer_state)  # NOOP, this would normally decrement expected_tx, but the cp async does so already
                producer_state.advance()

        if warp_idx > 0:
            # consumer
            consumer_state = pipeline.make_pipeline_state(
                # phase starts at 0
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            for _ in cutlass.range(self.n_iters, unroll=1):
                pipe.consumer_wait(consumer_state)  # wait at full
                print32('Consumer Wait')
                pipe.consumer_release(consumer_state)  # arrive at full(each warp arrives once)
                print32('Consumer Release')
                consumer_state.advance()


demo = BarrierLifetime()
current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
demo(current_stream)
torch.cuda.synchronize()