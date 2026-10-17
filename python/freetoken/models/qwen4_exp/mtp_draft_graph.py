"""Owned greedy draft-chain graphs with GPU positions and FreeToken paged attention."""
from __future__ import annotations
import torch
from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

class DraftGraphs:

    def __init__(self, draft, stream, *, attention_width):
        """Own compressed draft weights/KV; borrow target embedding, head and RoPE."""
        if not 0 < attention_width <= draft.max_context or draft.max_context % 64:
            raise ValueError('Draft graph requires a bounded page-aligned context')
        self.draft = draft
        self.stream = stream
        device = draft.device
        self.cell = torch.zeros(1, device=device, dtype=torch.int32)
        self.r = torch.zeros((1, 10240), device=device, dtype=torch.float32)
        self.token = torch.zeros(1, device=device, dtype=torch.int32)
        self.candidates = torch.empty(3, device=device, dtype=torch.int32)
        self.columns = torch.arange(attention_width, device=device, dtype=torch.int32)[None, :]
        self.table = torch.arange(draft.max_context // 64, device=device, dtype=torch.int32)[None, :]
        self.mapping = torch.zeros(1, device=device, dtype=torch.int32)
        self.keys = draft.keys.view(-1, 64, 2, 256)
        self.values = draft.values.view(-1, 64, 2, 256)
        self.catchup_r = torch.zeros((3, 10240), device=device, dtype=torch.float32)
        self.catchup_tokens = torch.zeros(3, device=device, dtype=torch.int32)
        self.catchup_start = torch.zeros(1, device=device, dtype=torch.int32)
        self.chains = {}
        self.catchup = {}
        self.attention_width = attention_width
        key_before = draft.keys[:3].clone()
        value_before = draft.values[:3].clone()
        try:
            for count in range(1, 4):

                def chain():
                    for row in range(count):
                        token, residual = draft.step(self.r, self.token, 0, positions=self.cell, attend=self.attend)
                        self.r.copy_(residual)
                        self.token.copy_(token)
                        self.candidates[row].copy_(token[0])
                        self.cell.add_(1)
                self.cell.zero_()
                chain()
                self.cell.zero_()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    chain()
                self.chains[count] = graph

                def catchup():
                    r = draft.initial_residual(self.catchup_r[:count], self.catchup_tokens[:count])
                    x, _ = draft.mix('attn_hyper_connection', r)
                    positions = self.catchup_start + torch.arange(count, device=device, dtype=torch.int32)
                    draft.key_value(x, positions)
                catchup()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    catchup()
                self.catchup[count] = graph
        finally:
            draft.keys[:3].copy_(key_before)
            draft.values[:3].copy_(value_before)
            torch.cuda.synchronize()

    def attend(self, q, positions):
        """Mask future/private speculative cells on device before paged FP32 attention."""
        indices = torch.where(self.columns <= positions[0], self.columns, -1).to(torch.int32)
        return qsa_sparse_paged_attention(q.reshape(1, 24, 256), self.keys, self.values, indices, self.table, self.mapping, num_stages=1).reshape(1, 24, 1, 256)

    def chain(self, residual, token, cell, count):
        """Launch one draft chain and read its candidates once on the host."""
        if count not in self.chains or cell < 0 or cell + count > self.attention_width:
            raise ValueError('Draft chain is outside the captured attention capacity')
        self.r.copy_(residual)
        self.token.fill_(int(token))
        self.cell.fill_(cell)
        self.chains[count].replay()
        return self.candidates[:count].cpu().tolist()

    def prime(self, residuals, tokens, start):
        """Build only K/V from verified target residuals and next tokens."""
        count = len(tokens)
        if count not in self.catchup or start + count > self.attention_width:
            raise ValueError('Draft catch-up is outside the captured capacity')
        self.catchup_r[:count].copy_(residuals)
        self.catchup_tokens[:count].copy_(tokens)
        self.catchup_start.fill_(start)
        self.catchup[count].replay()

    def close(self):
        """Destroy owned graphs after their users drain; borrowed target weights stay owned by the model."""
        torch.cuda.synchronize()
        for graph in [*self.chains.values(), *self.catchup.values()]:
            graph.reset()
        self.chains.clear()
        self.catchup.clear()
