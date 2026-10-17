"""保留每段 GGUF 原始类型的线性层；不同类型不拼接压缩字节。"""
from dataclasses import dataclass
import torch

from .base import LinearConfig, LinearKernel, LinearMethod


@dataclass(frozen=True)
class GGUFLinearConfig(LinearConfig):
    segments: tuple = ()
    padding_rows: int = 0


class GGUFLinearKernel(LinearKernel):
    name = "triton"

    def apply(self, layer, x):
        """同一输入只量化一次，各投影段写入对应切片，HC 对齐尾部显式填零。"""
        from freetoken.kernel.gguf_quant import compressed_linear, quantize_activation
        cfg = layer.quant_method.cfg
        output = torch.empty((x.shape[0], cfg.out_features), device=x.device, dtype=torch.float32)
        quantized = None
        cursor = 0
        for index, tensor in enumerate(cfg.segments):
            count = tensor.nbytes // tensor.row_bytes
            activation = x
            if tensor.type_id not in (0, 1, 30):
                if quantized is None:
                    quantized = quantize_activation(x)
                activation = quantized
            compressed_linear(getattr(layer, f"packed_{index}"), activation, count, tensor.type_id,
                              quantize_input=False, out=output[:, cursor:cursor + count], q8_source=x)
            cursor += count
        if cursor < cfg.out_features:
            output[:, cursor:].zero_()
        if layer.bias is not None:
            output += layer.bias
        return output


class GGUFLinearMethod(LinearMethod):
    candidates = (GGUFLinearKernel,)

    def create_weights(self, layer):
        """只声明压缩 byte buffer；LM head 的基类 BF16 占位必须释放。"""
        cfg = self.cfg
        if any(t.shape[0] != cfg.in_features for t in cfg.segments):
            raise ValueError("GGUF linear input geometry mismatch")
        rows = sum(t.nbytes // t.row_bytes for t in cfg.segments)
        if rows + cfg.padding_rows != cfg.out_features:
            raise ValueError("GGUF linear output geometry mismatch")
        if hasattr(layer, "weight"):
            del layer.weight
        for index, tensor in enumerate(cfg.segments):
            setattr(layer, f"packed_{index}", torch.empty(tensor.nbytes, dtype=torch.uint8))
