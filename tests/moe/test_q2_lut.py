import torch
import triton
import triton.language as tl
import pytest
from freetoken.moe.fused_q2_q8 import _signed_q2_codes as signed_codes


@triton.jit
def unpack(C,Y,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B)
    code=tl.load(C+index).to(tl.uint32)
    lo=signed_codes(code,tl.full((B,),False,tl.int1))
    hi=signed_codes(code,tl.full((B,),True,tl.int1))
    tl.store(Y+index*2,lo);tl.store(Y+index*2+1,hi)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_all_q2_codes_decode_exactly():
    """Compare the assembly decoder with the independently defined code-minus-one domain."""
    values=torch.arange(65536,device='cuda',dtype=torch.int32)
    encoded=values.to(torch.int16)
    packed=torch.empty((65536,2),device='cuda',dtype=torch.int32)
    unpack[(256,)](encoded,packed,256,num_warps=4)
    reference=(((values[:,None]>>(torch.arange(8,device='cuda')*2)[None,:])&3)-1).to(torch.int8)
    torch.testing.assert_close(packed.view(torch.int8),reference,atol=0,rtol=0)
