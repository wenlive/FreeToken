"""完整 GGUF 与 FreeToken state_dict 的加载接口；专家和 PLE 保持独立 bank。"""
import torch
from freetoken.layers.base import BaseOP


class GGUFEmbedding(BaseOP):
    def __init__(self, tensor):
        """embedding 常驻压缩字节，按 token 解量化，输出 FP32 residual。"""
        self.num_embeddings = tensor.shape[1]
        self.embedding_dim = tensor.shape[0]
        self._type = tensor.type_id
        self.packed_0 = torch.empty(tensor.nbytes, dtype=torch.uint8)

    def forward(self, ids):
        from freetoken.kernel.gguf_quant import dequantize_rows
        return dequantize_rows(self.packed_0, ids, self.embedding_dim, self._type)

    def place_mapped_host(self) -> int:
        """Bind after weight loading and before graph capture; retain the host owner for all readers."""
        from freetoken.kernel.pinned import MappedHostBytes, alloc_pinned_tensor
        if hasattr(self, '_mapped_host'):
            return self._mapped_host.owner.numel()
        if not self.packed_0.is_cuda or torch.cuda.is_current_stream_capturing():
            raise ValueError('Embedding mapping requires loaded CUDA weights before graph capture')
        packed = self.packed_0
        host = alloc_pinned_tensor(packed.numel(), dtype=torch.uint8)
        host.copy_(packed.cpu())
        mapping = MappedHostBytes(host)
        view = mapping.cuda_view(packed.device)
        # Publish both only after every fallible allocation/import succeeds; failure keeps the GPU table.
        self._mapped_host = mapping
        self.packed_0 = view
        return host.numel()


def declare_fp32_small_parameters(model):
    """GGUF 主干已是 uint8 buffer；剩余 norm/conv/状态参数声明 FP32，阻止 loader 降精度。"""
    from freetoken.layers.norm import GemmaPlusOneRMSNorm
    if isinstance(model, GemmaPlusOneRMSNorm):
        from freetoken.kernel.gguf_quant import gguf_gemma_rmsnorm
        model.gemma_rmsnorm = gguf_gemma_rmsnorm
    for name, value in tuple(vars(model).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            setattr(model, name, value.float())
        elif isinstance(value, BaseOP):
            declare_fp32_small_parameters(value)
        elif isinstance(value, (tuple, list)):
            for child in value:
                if isinstance(child, BaseOP):
                    declare_fp32_small_parameters(child)


def iter_gguf_weights(quant, device):
    """读取完整主干压缩段、小参数和 GGUF hashing 常量；不读取 HF 主干。"""
    for prefix, segments in quant.groups.items():
        for index, tensor in enumerate(segments):
            rows = tensor.nbytes // tensor.row_bytes
            raw = torch.frombuffer(bytearray(tensor.read_rows(0, rows)), dtype=torch.uint8)
            yield f"{prefix}.packed_{index}", raw.to(device)
    grouped = {t.name for segments in quant.groups.values() for t in segments}
    dtype = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}
    for name, tensor in quant.source.tensors.items():
        if name in grouped:
            continue
        mapping = quant.mapping[name]
        if mapping.semantics in ("expert_bank", "iq4_nl_row_table"):
            continue
        raw = bytearray(tensor.read_rows(0, tensor.nbytes // tensor.row_bytes))
        value = torch.frombuffer(raw, dtype=dtype[tensor.type_id]).float().reshape(tensor.torch_shape)
        if mapping.semantics == "folded_plus_one_norm":
            value = value - 1.0
        elif mapping.semantics == "negative_exp_A":
            if not bool((value < 0).all()):
                raise ValueError(f"{name}: GDN decay parameters must be negative")
            # GGUF 专用递推直接消费负的 exp(A)，避免 log/exp 往返改变权重。
        elif mapping.semantics == "depthwise_conv":
            value = value.unsqueeze(1)
        yield "model." + mapping.target, value.to(device)
    metadata = quant.source.metadata
    for layer in metadata["qwen4exp.ple.layers"]:
        prefix = f"model.layers.{layer}.ple.ple_embedding"
        for suffix, key in (("layer_multipliers", "layer_multipliers"),
                            ("ngram_heads_vocab_sizes", "head_vocab_sizes"),
                            ("ngram_heads_offsets", "head_offsets")):
            values = metadata["qwen4exp.ple." + key]
            yield prefix + "." + suffix, torch.tensor(values, dtype=torch.int64, device=device)
