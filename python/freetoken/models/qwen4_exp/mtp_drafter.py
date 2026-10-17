"""FreeToken-kernel MTP calibration prototype; greedy single request, dense draft attention."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from freetoken.kernel.gguf_quant import compressed_linear, gguf_gemma_rmsnorm
from freetoken.kernel.gguf_native import bf16_gemv
from freetoken.kernel.triton.hc import grouped_gemma_rmsnorm, hc_silu, hc_gate_mix, hc_combine
from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8

class MtpDrafter:

    def __init__(self, runtime: Path, main_model, *, max_context=8192):
        """Own compressed draft weights/KV; borrow target embedding, head and RoPE."""
        manifest = json.loads((runtime / 'runtime.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'packed_unvalidated_runtime' or len(manifest['tensors']) != 31:
            raise ValueError('MTP runtime conversion is incomplete')
        self.device = main_model.model.embed_tokens.packed_0.device
        self.embedding = main_model.model.embed_tokens
        self.head = main_model.lm_head
        self.rope = next((layer.self_attn.rotary for layer in main_model.model.layers.op_list if not layer._is_linear))
        self.renormalize = main_model._config.norm_topk_prob
        self.weights = {}
        self.metadata = {}
        for row in manifest['tensors']:
            path = runtime / row['path']
            with path.open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != row['sha256']:
                    raise ValueError('MTP runtime bytes changed: ' + row['name'])
            raw = bytearray(path.read_bytes())
            dtype = torch.float32 if row['type_id'] == 0 else torch.uint8
            weight = torch.frombuffer(raw, dtype=dtype).to(self.device)
            if row['type_id'] == 0:
                weight = weight.reshape(row['shape'])
            elif row['name'] == 'mlp.experts.gate_up_proj':
                weight = weight.reshape(512, 1280, 720)
            elif row['name'] == 'mlp.experts.down_proj':
                weight = weight.reshape(512, 2560, 180)
            self.weights[row['name']] = weight
            self.metadata[row['name']] = row
            del raw
        self.max_context = max_context
        self.keys = torch.zeros((max_context, 2, 256), device=self.device, dtype=torch.float32)
        self.values = torch.zeros_like(self.keys)
        self.owned_bytes = sum((t.numel() * t.element_size() for t in self.weights.values())) + 2 * self.keys.numel() * self.keys.element_size()

    def linear(self, name, x):
        row = self.metadata[name]
        return compressed_linear(self.weights[name], x, row['shape'][0], row['type_id'])

    def initial_residual(self, residual, tokens):
        """Normalize all four streams before shared per-stream fc_hidden projection."""
        h = gguf_gemma_rmsnorm(residual, self.weights['pre_fc_norm_hidden.weight'])
        hidden = self.linear('fc_hidden.weight', h.reshape(-1, 2560)).reshape(-1, 10240)
        embedding = self.embedding.forward(tokens)
        embedding = gguf_gemma_rmsnorm(embedding, self.weights['pre_fc_norm_embedding.weight'])
        embedding = self.linear('fc_embedding.weight', embedding).repeat(1, 4)
        return hidden + embedding

    def mix(self, prefix, residual, *, inject=True):
        """Use the validated FP32 HC path with merged BF16 down/inject projection."""
        rn = grouped_gemma_rmsnorm(residual, self.weights[prefix + '.hc_norm.weight'], 1e-06, 4)
        if inject:
            down = bf16_gemv(self.weights[prefix + '.input_mix_weight_down.weight'], rn, 324, packed1=self.weights[prefix + '.block_inject_weight.weight'], first_rows=320)
            lowrank, raw_inject = (down[:, :320], down[:, 320:324])
        else:
            lowrank = self.linear(prefix + '.input_mix_weight_down.weight', rn)
            raw_inject = None
        gates = self.linear(prefix + '.input_mix_weight_up.weight', hc_silu(lowrank, 4))
        return (hc_gate_mix(rn, gates, 4), raw_inject)

    def key_value(self, x, positions):
        k = self.linear('self_attn.k_proj.weight', x).reshape(-1, 256)
        k = gguf_gemma_rmsnorm(k, self.weights['self_attn.k_norm.weight']).reshape(-1, 512)
        _, k = self.rope.forward(positions, torch.zeros_like(k), k)
        v = self.linear('self_attn.v_proj.weight', x)
        self.keys.index_copy_(0, positions.long(), k.reshape(-1, 2, 256))
        self.values.index_copy_(0, positions.long(), v.reshape(-1, 2, 256))

    def prime(self, residuals, next_tokens, *, start=0):
        """Build only K/V from verified target residuals and next tokens."""
        if start + len(next_tokens) > self.max_context:
            raise ValueError('MTP draft context capacity exceeded')
        for offset in range(0, len(next_tokens), 8):
            tokens = next_tokens[offset:offset + 8]
            r = self.initial_residual(residuals[offset:offset + 8], tokens)
            x, _ = self.mix('attn_hyper_connection', r)
            positions = torch.arange(start + offset, start + offset + len(tokens), device=self.device, dtype=torch.int32)
            self.key_value(x, positions)

    def step(self, residual, token, cell, *, positions=None, attend=None):
        """Cell p pairs R_p with token p+1 and predicts p+2; recursive KV remains private."""
        if not 0 <= cell < self.max_context:
            raise ValueError('MTP draft cell is outside its own KV capacity')
        if positions is None:
            positions = torch.tensor([cell], device=self.device, dtype=torch.int32)
        r = self.initial_residual(residual, token.reshape(1))
        x, inject = self.mix('attn_hyper_connection', r)
        self.key_value(x, positions)
        qgate = self.linear('self_attn.q_proj.weight', x).reshape(1, 24, 512)
        q = qgate[:, :, :256].contiguous().reshape(-1, 256)
        q = gguf_gemma_rmsnorm(q, self.weights['self_attn.q_norm.weight']).reshape(1, 6144)
        q, _ = self.rope.forward(positions, q, torch.zeros((1, 512), device=self.device, dtype=torch.float32))
        if attend is None:
            attended = F.scaled_dot_product_attention(q.reshape(1, 24, 1, 256), self.keys[:cell + 1].permute(1, 0, 2).unsqueeze(0), self.values[:cell + 1].permute(1, 0, 2).unsqueeze(0), enable_gqa=True)
        else:
            attended = attend(q, positions)
        attended = attended.reshape(1, 6144) * qgate[:, :, 256:].reshape(1, 6144).sigmoid()
        r = hc_combine(r, self.linear('self_attn.o_proj.weight', attended), inject, 4)
        x, inject = self.mix('mlp_hyper_connection', r)
        router = self.linear('mlp.gate.weight', x)
        scores, ids = router.topk(10, dim=-1)
        weights = scores.softmax(-1) if self.renormalize else router.softmax(-1).gather(-1, ids)
        routed = fused_experts_q2_q8(x, self.weights['mlp.experts.gate_up_proj'], self.weights['mlp.experts.down_proj'], weights.contiguous(), ids.to(torch.int32).contiguous())
        shared = F.silu(self.linear('mlp.shared_expert.gate_proj.weight', x)) * self.linear('mlp.shared_expert.up_proj.weight', x)
        shared = self.linear('mlp.shared_expert.down_proj.weight', shared)
        gate = self.linear('mlp.shared_expert_gate.weight', x).sigmoid()
        r = hc_combine(r, routed + gate * shared, inject, 4)
        mixed, _ = self.mix('hyper_connection_mixer', r, inject=False)
        logits = self.head.quant_method.apply(self.head, mixed)
        return (logits.argmax(-1).to(torch.int32), r)
