"""GGUF GDN 对独立逐 token Torch 参考，覆盖真实卷积视图和状态续接。"""
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GGUFStateTests(unittest.TestCase):
    def test_verify_conv_and_recurrence_snapshots_equal_native_decode(self):
        from freetoken.attention.linear import FLAMetadata
        from freetoken.kvcache.linear_state_pool import LinearStatePool
        from freetoken.models.config import LinearGatedDeltaGroupConfig
        from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
        group = LinearGatedDeltaGroupConfig(
            name='linear', layer_ids=(0,), num_key_heads=2, num_value_heads=6,
            key_head_dim=128, value_head_dim=128, conv_kernel_dim=4, output_gate='sigmoid')
        pool = LinearStatePool(group, 8, torch.float32, torch.device('cuda'), tp_size=1)
        op = object.__new__(Qwen4ExpGatedDeltaNet)
        op.layer_id = 0
        op.num_k_heads, op.num_v_heads = 2, 6
        op.head_k_dim, op.head_v_dim = 128, 128
        op.key_dim, op.value_dim = 256, 768
        op.conv1d = SimpleNamespace(weight=torch.randn(1280, 1, 4, device='cuda'))
        op.A_log = -torch.rand(6, device='cuda')
        op.dt_bias = torch.randn(6, device='cuda')
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                pool.conv_states.normal_()
                pool.recurrent_states.normal_(0, .01)
                initial_conv = pool.conv_states.clone()
                initial_rec = pool.recurrent_states.clone()
                conv = torch.randn(count, 1280, device='cuda')
                a, b = torch.randn(count, 6, device='cuda'), torch.randn(count, 6, device='cuda')
                slots = torch.tensor([1], dtype=torch.int32, device='cuda')
                fla = FLAMetadata(torch.tensor([0, count], dtype=torch.int32, device='cuda'), slots,
                    track_dst=torch.arange(2, count + 2, device='cuda'),
                    track_boundary_row=torch.arange(1, count + 1, device='cuda'), is_verify=True)
                output = op._gguf_core(conv, a, b, pool, 0, fla, False)
                verify_conv, verify_rec = pool.conv_states.clone(), pool.recurrent_states.clone()
                pool.conv_states.copy_(initial_conv)
                pool.recurrent_states.copy_(initial_rec)
                one = FLAMetadata(torch.tensor([0, 1], dtype=torch.int32, device='cuda'), slots)
                expected = []
                for row in range(count):
                    expected.append(op._gguf_core(conv[row:row+1], a[row:row+1], b[row:row+1], pool, 0, one, True))
                    torch.testing.assert_close(verify_conv[0, row + 2], pool.conv_states[0, 1], rtol=0, atol=0)
                    torch.testing.assert_close(verify_rec[0, row + 2], pool.recurrent_states[0, 1], rtol=0, atol=0)
                torch.testing.assert_close(output, torch.cat(expected), rtol=0, atol=0)
                if count == 4:
                    # A reusable graph must honor the live request's staged slot,
                    # rather than baking the capture request's slot into snapshots.
                    graph = torch.cuda.CUDAGraph()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        op._gguf_core(conv, a, b, pool, 0, fla, False)
                        with torch.cuda.graph(graph, stream=stream):
                            captured = op._gguf_core(conv, a, b, pool, 0, fla, False)
                    torch.cuda.current_stream().wait_stream(stream)
                    pool.conv_states.copy_(initial_conv)
                    pool.recurrent_states.copy_(initial_rec)
                    slots.fill_(6)
                    graph.replay()
                    verify_conv, verify_rec = pool.conv_states.clone(), pool.recurrent_states.clone()
                    pool.conv_states.copy_(initial_conv)
                    pool.recurrent_states.copy_(initial_rec)
                    expected = []
                    for row in range(count):
                        expected.append(op._gguf_core(conv[row:row+1], a[row:row+1], b[row:row+1], pool, 0, one, True))
                        torch.testing.assert_close(verify_conv[0, row + 2], pool.conv_states[0, 6], rtol=0, atol=0)
                        torch.testing.assert_close(verify_rec[0, row + 2], pool.recurrent_states[0, 6], rtol=0, atol=0)
                    torch.testing.assert_close(captured, torch.cat(expected), rtol=0, atol=0)
                    graph.reset()

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
