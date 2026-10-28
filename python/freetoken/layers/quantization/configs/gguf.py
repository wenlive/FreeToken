"""完整 Flash Next GGUF 契约，融合段类型由目录决定，不套用 HF quant 标签。"""
from collections import defaultdict
from pathlib import Path

from .base import QuantConfig, cfg_get
from ..registry import LayerKind
from ..scheme import QuantKind, QuantScheme, WeightDesc


class GGUFQuantConfig(QuantConfig):
    dialect = "gguf_strata"
    STORAGE = {}

    def __init__(self, hf_config, *, name_map=None):
        """按 checkpoint 目录解析可迁移来源；目录和几何校验由 GGUF reader 执行。"""
        super().__init__(name_map)
        from freetoken.models.qwen4_exp.gguf_source import GGUFModelSource, map_tensor
        contract = cfg_get(hf_config, "freetoken_gguf_source")
        if cfg_get(hf_config, "model_type") != "qwen4_exp" or contract.get("format") != "gguf_strata":
            raise ValueError("complete GGUF source requires explicit qwen4_exp contract")
        paths = []
        for shard in contract["shards"]:
            path = Path(shard["path"])
            if not path.is_absolute():
                root = cfg_get(hf_config, "_name_or_path")
                if not root:
                    raise ValueError("Relative GGUF paths require a local checkpoint directory")
                path = Path(root) / path
            stat = path.stat()
            if stat.st_size != shard["size"]:
                raise ValueError("GGUF shard size differs from checkpoint configuration")
            paths.append(path)
        self.source = GGUFModelSource(paths)
        self._validate_geometry(hf_config)
        self.groups = defaultdict(list)
        self.mapping = {}
        for name, tensor in self.source.tensors.items():
            mapping = map_tensor(name)
            self.mapping[name] = mapping
            if mapping.semantics in ("expert_bank", "iq4_nl_row_table"):
                continue
            if (mapping.target.endswith(".weight") and mapping.semantics == "identity"
                    and (len(tensor.shape) == 2 or mapping.target.endswith("shared_expert_gate.weight"))):
                target = mapping.target if mapping.target == "lm_head.weight" else "model." + mapping.target
                self.groups[target.removesuffix(".weight")].append((mapping.segment, tensor))
        order = {"q": 0, "k": 1, "v": 2, "qkv": 0, "z": 1,
                 "b": 0, "a": 1, "gate": 0, "up": 1, "down": 0, "inject": 1, None: 0}
        self.groups = {target: tuple(t for segment, t in sorted(parts, key=lambda p:order[p[0]]))
                       for target, parts in self.groups.items()}

    def _validate_geometry(self, hf_config):
        """HF 文件只提供模型配置与 tokenizer，不能覆盖 GGUF 的真实几何和 hashing 边界。"""
        text = cfg_get(hf_config, "text_config", hf_config)
        fields = {
            "block_count": "num_hidden_layers", "embedding_length": "hidden_size",
            "attention.head_count": "num_attention_heads", "attention.head_count_kv": "num_key_value_heads",
            "attention.key_length": "head_dim", "attention.value_length": "head_dim",
            "expert_count": "num_experts", "expert_used_count": "num_experts_per_tok",
            "expert_feed_forward_length": "moe_intermediate_size",
            "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
            "ssm.conv_kernel": "linear_conv_kernel_dim", "ssm.state_size": "linear_key_head_dim",
            "ssm.group_count": "linear_num_key_heads", "ssm.time_step_rank": "linear_num_value_heads",
            "full_attention_interval": "full_attention_interval",
            "hyper_connection.count": "hc_count", "hyper_connection.low_rank": "hc_lowrank",
            "attention.indexer.head_count": "indexer_n_heads", "attention.indexer.key_length": "indexer_head_dim",
            "attention.indexer.top_k": "indexer_budget", "ple.ngram_size": "ngram_size",
            "ple.heads_per_ngram": "heads_per_ngram", "ple.conv_kernel": "ple_conv_kernel_size",
        }
        metadata = self.source.metadata
        for key, field in fields.items():
            if metadata.get("qwen4exp." + key) != cfg_get(text, field):
                raise ValueError(f"GGUF/config geometry mismatch: {key} vs {field}")
        heads = (cfg_get(text, "ngram_size") - 1) * cfg_get(text, "heads_per_ngram")
        sizes, offsets = metadata["qwen4exp.ple.head_vocab_sizes"], metadata["qwen4exp.ple.head_offsets"]
        if len(sizes) != heads or offsets != [sum(sizes[:i]) for i in range(heads)]:
            raise ValueError("GGUF PLE head sizes/offsets mismatch")
        table = self.source.tensors["per_layer_token_embd.weight"]
        divisor = cfg_get(text, "make_ngram_vocab_size_divisible_by")
        padded_rows = (sum(sizes) + divisor - 1) // divisor * divisor
        if (table.type_id != 20 or table.shape[0] * heads != cfg_get(text, "ple_embed_dim")
                or table.shape[1] != padded_rows):
            raise ValueError("GGUF IQ4_NL PLE table/config mismatch")

    def scheme_for_name(self, name):
        """模型配置仅询问 routed expert 精度；其他调用使用运行时 prefix 目录。"""
        from ..names import is_routed_expert
        if is_routed_expert(name):
            return QuantScheme(QuantKind.Q2_0, WeightDesc("q2_0_q8", (1, 64), "f16"), {"weight"})
        return QuantScheme(QuantKind.GGUF, WeightDesc("gguf", None, None), {"packed"})

    def scheme_for(self, prefix):
        from ..names import is_routed_expert
        if is_routed_expert(prefix):
            return self.scheme_for_name(prefix)
        if prefix not in self.groups:
            raise ValueError(f"GGUF has no linear binding for {prefix}")
        return self.scheme_for_name(prefix)

    def get_quant_method(self, layer, prefix):
        """线性层绑定显式分段 method；专家保留 FreeToken 的 bank/cache 生命周期。"""
        if layer.quant_layer_kind is LayerKind.MOE:
            return super().get_quant_method(layer, prefix)
        from ..linear.gguf import GGUFLinearConfig, GGUFLinearMethod
        segments = self.groups[prefix]
        rows = sum(t.nbytes // t.row_bytes for t in segments)
        padding = (-rows) % 16 if prefix.endswith("input_mix_weight_down_block_inject") else 0
        cfg = GGUFLinearConfig(layer.in_features, layer.out_features, tuple(layer.output_sizes),
                               self.scheme_for(prefix), segments, padding)
        return GGUFLinearMethod(cfg, "triton")
