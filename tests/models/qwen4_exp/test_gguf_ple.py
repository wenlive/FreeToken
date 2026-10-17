"""Regression for PLE gate rounding amplified by downstream Q8 quantization."""
from types import SimpleNamespace
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GGUFPLEReductionTests(unittest.TestCase):
    def test_real_forward_gate_is_independent_of_request_batch(self):
        from freetoken.models.qwen4_exp.ple import PLELayer
        torch.manual_seed(20261012)
        width=10240
        layer=object.__new__(PLELayer)
        layer.hc_count=4;layer.hidden_size=2560;layer._gguf=True;layer._pending=None
        layer.norm_key=SimpleNamespace(forward=lambda x:x)
        layer.norm_query=SimpleNamespace(forward=lambda x:x)
        layer.norm_conv=SimpleNamespace(forward=lambda x:x)
        layer.key_proj=SimpleNamespace(forward=lambda x:x)
        layer.value_proj=SimpleNamespace(forward=lambda x:x[:,:2560])
        layer._short_conv=lambda x,meta,states:torch.zeros_like(x)
        states=torch.empty((1,1,1),device='cuda')
        for count in (2,7,129):
            embeddings=torch.randn((count,width),device='cuda')
            r=torch.randn_like(embeddings)
            layer.ple_embedding=SimpleNamespace(row_ids=lambda meta:meta,
                table=SimpleNamespace(lookup=lambda ids:embeddings[ids]))
            ids=torch.arange(count,device='cuda')
            full=layer.forward(r,SimpleNamespace(),meta=ids,conv_states=states)
            separate=torch.cat([layer.forward(r[j:j+1],SimpleNamespace(),meta=ids[j:j+1],conv_states=states)
                for j in range(count)])
            torch.testing.assert_close(full,separate,rtol=0,atol=0)

    def test_gate_preserves_fp32_formula_and_noncontiguous_row_stride(self):
        from freetoken.kernel.gguf_quant import gguf_ple_gate
        torch.manual_seed(319)
        key=torch.randn((9,20480),device='cuda')[:,::2].contiguous()
        # Unit feature stride, padded token stride as actual merged projections may use.
        query=torch.randn((9,10256),device='cuda')[:,:10240]
        gate=gguf_ple_gate(key,query,4)
        score=(key.reshape(9,4,2560)*query.reshape(9,4,2560)).sum(-1,keepdim=True)/(2560**.5)
        ref=torch.sigmoid(score.sign()*score.abs().clamp_min(1e-6).sqrt())
        torch.testing.assert_close(gate,ref,rtol=2e-6,atol=2e-7)
        zeros=torch.zeros_like(key)
        torch.testing.assert_close(gguf_ple_gate(zeros,query,4),torch.full_like(gate,.5),rtol=0,atol=0)

class VerifyPLEStateTests(unittest.TestCase):
    def test_window_and_every_snapshot_equal_native_decode(self):
        from freetoken.attention.linear import FLAMetadata
        from freetoken.models.qwen4_exp.ple import PLELayer, PLEMetadata, commit_ngram_context
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            for count in (1, 2, 3, 4):
                with self.subTest(device=device, count=count):
                    torch.manual_seed(205)
                    layer = object.__new__(PLELayer)
                    layer.state_len = 9
                    layer.dilation = 3
                    layer.conv1d = SimpleNamespace(weight=torch.randn(16, 1, 4, device=device))
                    states = torch.randn(7, 16, 9, device=device)
                    reference = states.clone()
                    context_pool = torch.arange(35, device=device).reshape(7, 5).clone()
                    context_ref = context_pool.clone()
                    x = torch.randn(count, 16, device=device)
                    ids = torch.arange(100, 100 + count, device=device)
                    slots = torch.tensor([1], device=device)
                    boundaries = torch.arange(1, count + 1, device=device)
                    dst = torch.arange(2, count + 2, device=device)
                    fla = FLAMetadata(torch.tensor([0, count], device=device), slots,
                                      track_dst=dst, track_boundary_row=boundaries, is_verify=True)
                    meta = PLEMetadata(ids, fla.cu_seqlens, (count,), context_pool[1:2].clone(),
                                       slots, None, False, True)
                    layer._write_track_snapshot(states, x, fla)
                    output = layer._short_conv(x, meta, states)
                    commit_ngram_context(meta, fla, context_pool)
                    expected = []
                    for row in range(count):
                        one = PLEMetadata(ids[row:row+1], torch.tensor([0, 1], device=device),
                                          (1,), context_ref[1:2].clone(), slots, None, True)
                        expected.append(layer._short_conv(x[row:row+1], one, reference))
                        commit_ngram_context(one, None, context_ref)
                        torch.testing.assert_close(states[row + 2], reference[1], rtol=0, atol=0)
                        torch.testing.assert_close(context_pool[row + 2], context_ref[1], rtol=0, atol=0)
                    # CPU SiLU changes vectorization with the row count. CUDA,
                    # the supported inference path, must match native decode exactly.
                    tolerance = 0 if device == 'cuda' else 2e-7
                    torch.testing.assert_close(output, torch.cat(expected), rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(states[1], reference[1], rtol=0, atol=0)
                    torch.testing.assert_close(context_pool[1], context_ref[1], rtol=0, atol=0)


if __name__=='__main__':unittest.main()
