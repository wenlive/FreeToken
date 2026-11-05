"""Packed Clover dense projections using the existing GGML Q8_1 CUDA kernels.

Layout reference: Maphist0/llama.cpp-pub@cb09d4355d8bf968bb907c5de03cf42d8aa073ce,
conversion/clover.py and conversion/qwen.py (MIT). No upstream code is copied.
"""

import torch

from freetoken.layers import BaseOP, LinearReplicated
from freetoken.models.qwen3_5_moe.weight import _DenseReader
from freetoken.models.register import get_model_spec


class CloverDenseGroups(_DenseReader):
    def __init__(self):
        super().__init__(None, get_model_spec("CloverMoeModelLM"))

    def target(self, module):
        parent, _, leaf = module.rpartition(".")
        if leaf in ("in_proj_qkv", "in_proj_z"):
            return parent + ".in_proj_qkvz", ("in_proj_qkv", "in_proj_z").index(leaf), 2
        if leaf in ("in_proj_b", "in_proj_a"):
            return parent + ".in_proj_ba", ("in_proj_b", "in_proj_a").index(leaf), 2
        return super().target(module)


class CloverPackedLinear(BaseOP):
    def __init__(self, rows, cols, quant_type, row_bytes):
        self.weight = torch.empty(rows, row_bytes, dtype=torch.uint8)
        self._rows = rows
        self._cols = cols
        self._type = quant_type

    def forward(self, x):
        from freetoken.kernel.gguf import ggml_mul_mat_a8, ggml_mul_mat_vec_a8

        operation = ggml_mul_mat_vec_a8 if x.shape[0] == 1 else ggml_mul_mat_a8
        return operation(self.weight, x, self._type, self._rows).to(x.dtype)


def install_packed_linears(model, config):
    """Replace only Clover GGUF linears; HF/NVFP4 and other families are unchanged."""
    source = config.expert_source
    for layer in model.model.layers.op_list:
        if layer._is_linear:
            mixer = layer.linear_attn
            if not mixer._split_in_proj:
                del mixer.in_proj
                mixer._split_in_proj = True
                mixer.in_proj_ba = LinearReplicated(config.hidden_size, 2 * mixer.num_v_heads, False)
    for name, (rows, cols, kind, row_bytes) in source["dense"].items():
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = parent.op_list[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], CloverPackedLinear(rows, cols, kind, row_bytes))
    for index, layer in enumerate(model.model.layers.op_list):
        layer.mlp.experts.gguf_types = source["types"][index]
        layer.mlp.experts.gguf_row_bytes = source["row_bytes"][index]
