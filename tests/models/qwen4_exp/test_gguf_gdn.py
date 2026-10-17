"""GGUF GDN 对独立逐 token Torch 参考，覆盖真实卷积视图和状态续接。"""
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GGUFStateTests(unittest.TestCase):
    def test_channels_first_view_varlen_snapshot_and_resume(self):
        from freetoken.kernel.triton.gguf_gdn import gguf_gdn_forward
        torch.manual_seed(17)
        hk,hv,d,total = 2,6,128,12
        raw = torch.randn((2*hk*d+hv*d,total),device='cuda')
        q,k,v = raw.T.split([hk*d,hk*d,hv*d],-1)
        q,k,v = q.reshape(total,hk,d),k.reshape(total,hk,d),v.reshape(total,hv,d)
        self.assertEqual(q.stride(-1),total)
        a,b = torch.randn((total,hv),device='cuda'),torch.randn((total,hv),device='cuda')
        decay = -torch.rand(hv,device='cuda')
        dt = torch.randn(hv,device='cuda')
        state = torch.randn((4,hv,d,d),device='cuda')*.01
        expected_state = state.clone()
        expected = torch.empty_like(v)
        def step(row,slot):
            # Strata L2 epsilon is inside sqrt; do not borrow candidate normalization.
            qh = q[row]*torch.rsqrt(q[row].square().sum(-1,keepdim=True)+1e-6)
            kh = k[row]*torch.rsqrt(k[row].square().sum(-1,keepdim=True)+1e-6)
            head = torch.arange(hv,device='cuda')%hk
            qh,kh = qh[head],kh[head]
            g = torch.exp(decay*F.softplus(a[row]+dt))
            s = expected_state[slot]*g[:,None,None]
            delta = (v[row]-torch.einsum('hkv,hk->hv',s,kh))*b[row].sigmoid()[:,None]
            s = s+kh[:,:,None]*delta[:,None,:]
            expected_state[slot].copy_(s)
            return torch.einsum('hkv,hk->hv',s,qh)*d**-.5
        for begin,end,slot in ((0,9,1),(9,12,3)):
            for row in range(begin,end):
                expected[row] = step(row,slot)
                if row == 4:
                    expected_state[2].copy_(expected_state[slot])
        fla = SimpleNamespace(cu_seqlens=torch.tensor([0,9,12],device='cuda',dtype=torch.int32),
            cache_indices=torch.tensor([1,3],device='cuda',dtype=torch.int32),
            track_boundary_row=torch.tensor([5],device='cuda',dtype=torch.int64),
            track_dst=torch.tensor([2],device='cuda',dtype=torch.int64))
        got = gguf_gdn_forward(q,k,v,a,b,decay,dt,state,fla,is_decode=False)
        torch.testing.assert_close(got,expected,rtol=5e-5,atol=3e-6)
        torch.testing.assert_close(state,expected_state,rtol=5e-5,atol=3e-6)
        one = SimpleNamespace(cu_seqlens=torch.tensor([0,1],device='cuda',dtype=torch.int32),
            cache_indices=torch.tensor([2],device='cuda',dtype=torch.int32),track_boundary_row=None,track_dst=None)
        reference = step(5,2)
        got = gguf_gdn_forward(q[5:6],k[5:6],v[5:6],a[5:6],b[5:6],decay,dt,state,one,is_decode=True)
        torch.testing.assert_close(got[0],reference,rtol=5e-5,atol=3e-6)
        torch.testing.assert_close(state[2],expected_state[2],rtol=5e-5,atol=3e-6)


if __name__ == '__main__':
    unittest.main()
