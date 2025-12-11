import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

"""
This kernel copies a 128x8 tensor using one warp(each thread grabs 8 elements)
It also casts from FP32 to Int32 before copying out(kinda)
The input tensor has row major storage, every element is index*100 e.g. [0, 100, 200, 300, ...]
Threads will copy this value to the output but add their threadidx to it, so we can see which threads copied what.
The result should be that every thread copies a row, and we have 2 blocks of 64 threads so we should see 0-64 then 0-64 again

Tiled Copy:
- The tiler is a CTA tiler, which is the size of the CTA's block
- The TV layout is the layout of the threads and values, mapped onto this CTA block which is col-major
- THEN, this will be mapped onto the input layout
- So (32, 8):(1, 32) maps onto a row-major (32, 8) layout such that every thread gets 8 contiguous elements,
    but (32, 8):(8:1) would not -- thread0 would get (0, 8, ... 56), thread1 gets (64, ...). It's not what it seems

Partitioning stuff for copies
- I dunno I think there's just convenience functions like make_fragment_D or whatever sometimes but I literally just partitioned a thing with the same shape and I'm good

Casting and elementwise ops
- out.store(in.load().to(new_dtype)) for registers
- Autovec_copy is useful for stuff too
- You can do an add like this: dtensor.load().to(self.out_dtype) + tidx
"""

@cute.jit
def print_tid(x, tid=0):
    tidx, _, _ = cute.arch.thread_idx() # threadidx.x, y, z
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == tid and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == tid and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

class Copy:
    def __init__(self):
        self.a_dtype = cutlass.Float32
        self.out_dtype = cutlass.Int32
    
    @cute.jit
    def __call__(self, a: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        assert a.element_type == self.a_dtype and out.element_type == self.out_dtype
        self.kernel(a, out).launch(grid=2, block=64, stream=stream)
    
    @cute.kernel
    def kernel(self, a: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tiled_copy_in = self._make_copy()
        thr_copy_in = tiled_copy_in.get_slice(tidx)
        gA_frag = thr_copy_in.partition_S(a)
        gOut_frag = thr_copy_in.partition_S(out)
        print(gA_frag)

        # normally this would be done with make_fragment or whatever but ok
        dtensor = cute.make_rmem_tensor(cute.make_layout(8), dtype=self.a_dtype)
        cute.copy(tiled_copy_in, gA_frag[None, bidx, 0], dtensor)
        # print(dtensor)
        # print_tid(dtensor, 0)
        # print_tid(dtensor, 1)
        # print_tid(dtensor, 31)

        dtensor_out = cute.make_rmem_tensor_like(dtensor, self.out_dtype)

        # we can do an elementwise add
        dtensor_out.store(dtensor.load().to(self.out_dtype) + tidx)
        cute.autovec_copy(dtensor_out, gOut_frag[None, bidx, 0])


    
    def _make_copy(self):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)

        # See above comment about TV layout
        # Tiled Copy
        #     Tiler MN:        (64:1,8:1)
        #     TV Layout tiled: ((32,2),8):((1,32),64)
        # Copy Atom
        #     ThrID:           1:0
        #     TV Layout Src:   (1,4):(0,1)
        #     TV Layout Dst:   (1,4):(0,1)
        #     Value type:      f32
        tc = cute.make_tiled_copy(atom, cute.make_layout(((32, 2), 8), stride=((1, 32), 64)), (64, 8))
        print(tc)
        return tc

def convert_from_dlpack(tensor, mode=0, stride_order=(0, 1)):
    return from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=mode, stride_order=stride_order
    )

lst = [i * 100 for i in range(128 * 8)]
a = torch.Tensor(lst).reshape((128, 8)).to('cuda')
# print(a)
out = torch.empty((128, 8), dtype=torch.int32, device='cuda')
current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
a_cute = convert_from_dlpack(a)
print(a_cute)
out_cute = convert_from_dlpack(out)
copy_kernel = Copy()
copy_kernel(a_cute, out_cute, current_stream)
print(out)
