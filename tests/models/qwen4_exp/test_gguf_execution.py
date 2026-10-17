"""Batch size must not select a different arithmetic contract in the GGUF path."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GGUFExecutionConsistencyTests(unittest.TestCase):
    def test_qk_norm_is_independent_of_batch_size(self):
        from freetoken.layers.norm import GemmaPlusOneRMSNorm
        from freetoken.models.qwen4_exp.gguf_weight import declare_fp32_small_parameters
        import freetoken.kernel.backend as backend
        original=backend.is_flashinfer_installed
        backend.is_flashinfer_installed=lambda:False
        try:
            norm=GemmaPlusOneRMSNorm(256,1e-6)
        finally:
            backend.is_flashinfer_installed=original
        declare_fp32_small_parameters(norm)
        torch.manual_seed(403)
        norm.weight=torch.randn(256,device="cuda")*.1
        x=torch.randn((2721,24,256),device="cuda")
        full=norm.forward(x)
        tail=norm.forward(x[-33:].contiguous())
        torch.testing.assert_close(full[-33:],tail,rtol=0,atol=0)

    def test_compressed_projection_is_independent_of_batch_size(self):
        from freetoken.kernel.gguf_quant import compressed_linear
        torch.manual_seed(317)
        x=torch.randn((19,256),device="cuda",dtype=torch.float32)
        weight=torch.randn((32,256),device="cuda").bfloat16().view(torch.uint8).flatten()
        full=compressed_linear(weight,x,32,30)
        tail=compressed_linear(weight,x[-1:],32,30)
        torch.testing.assert_close(full[-1:],tail,rtol=0,atol=0)

    def test_experts_are_independent_of_prefill_and_tail_length(self):
        from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
        torch.manual_seed(941)
        hidden,intermediate=256,128
        def bank(rows,width):
            blocks=torch.randint(0,256,(3,rows,width//64,18),device="cuda",dtype=torch.uint8)
            scale=(torch.rand((3,rows,width//64),device="cuda")*.02+.001).half()
            blocks[...,:2]=scale.view(torch.uint8).reshape(3,rows,width//64,2)
            return blocks.flatten(2)
        gu=bank(2*intermediate,hidden);down=bank(hidden,intermediate)
        x=torch.randn((129,hidden),device="cuda")
        ids=torch.stack([torch.arange(129,device="cuda")%3,(torch.arange(129,device="cuda")+1)%3],-1).to(torch.int32)
        weights=torch.tensor([.3,.7],device="cuda").repeat(129,1)
        full=fused_experts_q2_q8(x,gu,down,weights,ids,is_prefill=True)
        tail=fused_experts_q2_q8(x[-3:].contiguous(),gu,down,weights[-3:].contiguous(),ids[-3:].contiguous(),is_prefill=True)
        decode=fused_experts_q2_q8(x[-1:],gu,down,weights[-1:],ids[-1:],is_prefill=False)
        torch.testing.assert_close(full[-3:],tail,rtol=0,atol=0)
        torch.testing.assert_close(full[-1:],decode,rtol=0,atol=0)

    def test_fp32_attention_is_independent_of_batch_size(self):
        from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention
        torch.manual_seed(218)
        q=torch.randn((257,24,256),device="cuda")
        k=torch.randn((4,64,2,256),device="cuda").half();v=torch.randn_like(k)
        table=torch.tensor([[2,0,3,1]],device="cuda",dtype=torch.int32)
        ids=torch.arange(2051,device="cuda",dtype=torch.int32).repeat(257,1)
        ids[:,193:]=-1
        req=torch.zeros(257,device="cuda",dtype=torch.int32)
        full=qsa_sparse_paged_attention(q,k,v,ids,table,req)
        tail=qsa_sparse_paged_attention(q[-1:],k,v,ids[-1:],table,req[-1:])
        torch.testing.assert_close(full[-1:],tail,rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
