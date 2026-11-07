"""A small decoder-only transformer for trace generation.

LLaMA-style block: RMSNorm, softmax multi-head attention, and a ReLU MLP, with
residual connections. Positional information is configurable:

    pos = "nope"  no positional encoding (used for count-based tasks)
    pos = "rel"   a learned relative-position bias added to attention logits

The relative bias is a light, task-independent stand-in for the parameter-free
relative positional encoding of the base construction. For the counting tasks in
this repository, NoPE already suffices, matching the theory: membership depends
only on counter values, which attention recovers as prefix counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_len: int = 512
    pos: str = "nope"  # "nope" or "rel"
    rel_clip: int = 64
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos = cfg.pos
        self.rel_clip = cfg.rel_clip
        if cfg.pos == "rel":
            # one learned scalar bias per head and per clipped relative distance
            self.rel_bias = nn.Parameter(torch.zeros(cfg.n_heads, cfg.rel_clip + 1))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, h, T, dh
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)  # B, h, T, T
        # causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        if self.pos == "rel":
            idx = torch.arange(T, device=x.device)
            dist = (idx.view(T, 1) - idx.view(1, T)).clamp(min=0, max=self.rel_clip)
            bias = self.rel_bias[:, dist]  # h, T, T
            att = att + bias.unsqueeze(0)
        att = F.softmax(att, dim=-1)
        out = att @ v  # B, h, T, dh
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x


class TraceTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, eos_id, max_new_tokens):
        """Greedy decode from a single prompt (list of ints). Returns generated ids."""
        self.eval()
        device = next(self.parameters()).device
        seq = list(prompt_ids)
        gen = []
        for _ in range(max_new_tokens):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            if x.shape[1] > self.cfg.max_len:
                x = x[:, -self.cfg.max_len:]
            logits = self.forward(x)[0, -1]
            nxt = int(torch.argmax(logits).item())
            if nxt == eos_id:
                break
            gen.append(nxt)
            seq.append(nxt)
        return gen

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
