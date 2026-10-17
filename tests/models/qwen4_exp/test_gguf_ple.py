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

if __name__=='__main__':unittest.main()
