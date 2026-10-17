"""Batch size must not select a different arithmetic contract in the GGUF path."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GGUFExecutionConsistencyTests(unittest.TestCase):
    def test_mapped_embedding_preserves_rows_and_changed_graph_inputs(self):
        """Original Q3_K bytes remain exact with an owned host mapping through graph replay."""
        from types import SimpleNamespace
        import gc
        from freetoken.models.qwen4_exp.gguf_weight import GGUFEmbedding
        torch.manual_seed(827)
        vocab, width = 257, 2560
        packed = torch.randint(0, 256, (vocab, width // 256, 110), dtype=torch.uint8)
        scale = (torch.rand((vocab, width // 256)) * .01 + .001).half().contiguous()
        packed[:, :, 108:110] = scale.view(torch.uint8).reshape(vocab, width // 256, 2)
        embedding = GGUFEmbedding(SimpleNamespace(shape=(width, vocab), nbytes=packed.numel(), type_id=11))
        embedding.packed_0 = packed.flatten().cuda()
        ids = torch.tensor([0, 1, vocab // 2, vocab-1], device='cuda', dtype=torch.int32)
        reference = embedding.forward(ids)
        host_bytes = embedding.place_mapped_host()
        self.assertEqual(host_bytes, packed.numel())
        self.assertEqual(embedding.place_mapped_host(), host_bytes)
        self.assertEqual(embedding.packed_0.data_ptr(), embedding._mapped_host.ptr)
        self.assertTrue(torch.equal(embedding._mapped_host.owner, packed.flatten()))
        torch.testing.assert_close(embedding.forward(ids), reference, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = embedding.forward(ids)
        gc.collect()
        ids.copy_(torch.tensor([vocab-1, 3, 11, 0], device='cuda', dtype=torch.int32))
        graph.replay()
        from freetoken.kernel.gguf_quant import dequantize_rows
        expected = dequantize_rows(packed.flatten().cuda(), ids, width, 11)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        graph.reset()

    def test_failed_embedding_mapping_keeps_gpu_weights_and_zero_host_budget(self):
        """A failed pin or CUDA alias import must not replace the table or earn memory credit."""
        from types import SimpleNamespace
        from unittest.mock import patch
        from freetoken.models.qwen4_exp.gguf_weight import GGUFEmbedding
        from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
        for target in ('freetoken.models.qwen4_exp.gguf_weight.alloc_pinned_tensor',
                       'freetoken.kernel.pinned.MappedHostBytes.cuda_view'):
            with self.subTest(failure=target):
                embedding = GGUFEmbedding(SimpleNamespace(shape=(256, 2), nbytes=220, type_id=11))
                embedding.packed_0 = torch.arange(220, device='cuda', dtype=torch.uint8)
                original = embedding.packed_0
                model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
                model.model = SimpleNamespace(embed_tokens=embedding)
                with patch(target, side_effect=RuntimeError('forced mapping failure')):
                    self.assertEqual(model.place_input_embedding(), 0)
                self.assertIs(embedding.packed_0, original)
                self.assertFalse(hasattr(embedding, '_mapped_host'))
                self.assertEqual(model._embedding_placement['actual'], 'cuda')
                self.assertEqual(model._embedding_placement['host_bytes'], 0)

    def test_bf16_tiled_and_batch_projection_real_geometry(self):
        """覆盖新分派边界、HC两段、stride、graph改输入，并与独立FP32矩阵参考比较。"""
        from freetoken.kernel.gguf_native import bf16_gemv
        torch.manual_seed(7802)
        for width, rows, split, counts in [(320, 10240, 10240, (1, 3, 31, 32, 64)),
                                           (10240, 324, 320, (1, 31, 32, 65)),
                                           (2560, 512, 512, (3, 32, 257))]:
            weight = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)
            w0 = weight[:split].contiguous().view(torch.uint8).flatten()
            w1 = weight[split:].contiguous().view(torch.uint8).flatten() if split < rows else None
            for count in counts:
                with self.subTest(width=width, rows=rows, count=count):
                    x = torch.randn(count, width*2, device='cuda')[:, ::2]
                    storage = torch.full((count, rows*2+4), 123., device='cuda')
                    out = storage[:, 2:2+rows*2:2]
                    bf16_gemv(w0, x, rows, packed1=w1, first_rows=split, out=out)
                    reference = x @ weight.float().T
                    error = torch.linalg.vector_norm(out-reference) / torch.linalg.vector_norm(reference)
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertLess(error.item(), 3e-5)
                    self.assertTrue(torch.all(storage[:, 3:2+rows*2:2] == 123.))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        bf16_gemv(w0, x, rows, packed1=w1, first_rows=split, out=out)
                    x.mul_(.375)
                    graph.replay()
                    reference = x @ weight.float().T
                    error = torch.linalg.vector_norm(out-reference) / torch.linalg.vector_norm(reference)
                    self.assertLess(error.item(), 3e-5)

    def test_interleaved_columns_are_bitwise_equal_to_legacy(self):
        """支持格式的2/3/4列、stride和改输入图重放均须匹配原生旧归约。"""
        from freetoken.kernel.gguf_native import _module, quantize_q8_1, mmvq
        torch.manual_seed(8627)
        for typ, size, rows in [(12,144,1024),(13,176,8192),(14,210,640),(23,136,1024)]:
            width=2560
            raw=torch.randint(0,256,(rows*(width//256),size),device='cuda',dtype=torch.uint8)
            scale=torch.tensor([.015625],device='cuda',dtype=torch.float16).view(torch.uint8)
            if typ==14:raw[:,208:210]=scale
            else:raw[:,:2]=scale
            if typ in (12,13):raw[:,2:4]=scale
            weight=raw.flatten()
            for tokens in (2,3,4):
                with self.subTest(type=typ,tokens=tokens):
                    x=torch.randn(tokens,width,device='cuda')
                    q=quantize_q8_1(x)
                    expected=torch.empty(tokens,rows,device='cuda')
                    _module().project(weight,q,expected,typ,width,torch.cuda.current_stream().cuda_stream)
                    out=torch.full((tokens,rows+3),float('nan'),device='cuda')
                    mmvq(weight,q,rows,typ,out=out[:,:rows])
                    torch.testing.assert_close(out[:,:rows],expected,rtol=0,atol=0)
                    self.assertTrue(torch.isnan(out[:,rows:]).all())
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):result=mmvq(weight,quantize_q8_1(x),rows,typ)
                    for _ in range(3):
                        x.normal_();graph.replay()
                        q=quantize_q8_1(x)
                        _module().project(weight,q,expected,typ,width,torch.cuda.current_stream().cuda_stream)
                        torch.testing.assert_close(result,expected,rtol=0,atol=0)
                    graph.reset()

    def test_native_q8_graph_and_layout(self):
        from freetoken.kernel.gguf_native import quantize_q8_1, mmvq
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.manual_seed(839)
            raw = torch.randn((19, 512), device="cuda")
            x = raw[:, ::2]
            q = quantize_q8_1(x)
            weight = torch.zeros((17, 4, 18), device="cuda", dtype=torch.uint8)
            weight[..., :2] = torch.full((17, 4), .125, device="cuda").half().view(torch.uint8).reshape(17, 4, 2)
            weight[..., 2:] = 0b10110100
            p = weight.flatten()
            a = mmvq(p, q, 17, 42)
            separate = torch.cat([mmvq(p, quantize_q8_1(x[i:i+1]), 17, 42) for i in range(len(x))])
            torch.testing.assert_close(a, separate, rtol=3e-5, atol=3e-5)
            out = torch.empty((19, 40), device="cuda")[:, 2:36:2]
            mmvq(p, q, 17, 42, out=out)
            torch.testing.assert_close(out, a, rtol=0, atol=0)
            # A replay must read changed inputs rather than a captured output.
            fixed = x.contiguous()
            quantize_q8_1(fixed)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = mmvq(p, quantize_q8_1(fixed), 17, 42)
            fixed.mul_(1.5)
            graph.replay()
            expected = mmvq(p, quantize_q8_1(fixed), 17, 42)
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        stream.synchronize()

    def test_native_mmvq_rejects_bad_binding_geometry(self):
        from freetoken.kernel.gguf_native import _module, quantize_q8_1, mmvq
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.zeros((2, 256), device="cuda")
            q = quantize_q8_1(x)
            p = torch.zeros(17 * 4 * 18, device="cuda", dtype=torch.uint8)
            y = torch.empty((2, 17), device="cuda")
            with self.assertRaises((RuntimeError, ValueError)):
                _module().project(p[:-1], q, y, 42, 256, stream.cuda_stream)
            with self.assertRaises((RuntimeError, ValueError)):
                _module().project(p, q, y, 29, 256, stream.cuda_stream)
            alias = torch.empty(2 * 256 * 4, device="cuda", dtype=torch.uint8)
            with self.assertRaises((RuntimeError, ValueError)):
                _module().quantize(alias.view(torch.float32).reshape(2, 256),
                                   alias[:2*8*36].reshape(2, 8, 36), stream.cuda_stream)
            self.assertEqual(mmvq(p, q[:0], 17, 42).shape, (0, 17))
        stream.synchronize()

    def test_bf16_fusion_bias_padding_and_strides(self):
        from types import SimpleNamespace
        from freetoken.layers.quantization.linear.gguf import GGUFLinearKernel
        torch.manual_seed(151)
        x = torch.randn((11, 512), device="cuda")[:, ::2]
        w0 = torch.randn((17, 256), device="cuda").bfloat16()
        w1 = torch.randn((3, 256), device="cuda").bfloat16()
        segments = tuple(SimpleNamespace(type_id=30, nbytes=n*256*2, row_bytes=256*2) for n in (17,3))
        bias = torch.randn(24, device="cuda")
        layer = SimpleNamespace(quant_method=SimpleNamespace(cfg=SimpleNamespace(out_features=24, segments=segments)),
            packed_0=w0.view(torch.uint8).flatten(), packed_1=w1.view(torch.uint8).flatten(), bias=bias)
        got = GGUFLinearKernel().apply(layer, x)
        reference = torch.cat([x@w0.float().T, x@w1.float().T], dim=1)+bias[:20]
        torch.testing.assert_close(got[:, :20], reference, rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(got[:, 20:], bias[20:].expand(11,4), rtol=0, atol=0)

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
        # Parallel decode reduces K chunks in a different FP32 order from prefill.
        torch.testing.assert_close(full[-1:],decode,rtol=3e-5,atol=3e-7)
        relative_l2 = (full[-1:] - decode).norm() / full[-1:].norm().clamp_min(1e-10)
        self.assertLess(float(relative_l2), 2e-6)

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
