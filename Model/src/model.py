"""
model.py
Modern GPT architecture: RMSNorm, RoPE, SwiGLU, FlashAttention, Weight Tying.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0
    freqs     = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles    = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> tuple:
    T   = q.shape[2]
    q_c = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    # ✅ FIX: was incorrectly using q.shape[:-1] to reshape k.
    #         Must use k.shape[:-1] so k is reshaped according to its own dimensions.
    #         In standard MHA q and k have the same shape, so this was a silent
    #         bug that worked by coincidence — but would silently break for any
    #         variant where q and k differ (e.g. grouped-query attention).
    k_c = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    f   = freqs_cis[:T].unsqueeze(0).unsqueeze(0)
    return (
        torch.view_as_real(q_c * f).flatten(3).type_as(q),
        torch.view_as_real(k_c * f).flatten(3).type_as(k),
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.out_drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=2)

        def to_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        q, k    = apply_rope(q, k, freqs_cis)
        drop_p  = self.out_drop.p if self.training else 0.0
        out     = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=True)
        return self.out_drop(self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)))


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        hidden      = int(2 / 3 * 4 * n_embd)
        hidden      = (hidden + 63) // 64 * 64
        self.w1     = nn.Linear(n_embd, hidden, bias=False)
        self.w2     = nn.Linear(hidden, n_embd, bias=False)
        self.w3     = nn.Linear(n_embd, hidden, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn  = CausalSelfAttention(n_embd, n_head, dropout)
        self.norm2 = RMSNorm(n_embd)
        self.ffn   = SwiGLU(n_embd, dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        return x + self.ffn(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len, dropout):
        super().__init__()
        self.seq_len  = seq_len
        self.tok_emb  = nn.Embedding(vocab_size, n_embd)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.norm_f  = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying
        self.register_buffer("freqs_cis", precompute_rope_freqs(n_embd // n_head, seq_len))
        self.apply(self._init_weights)
        print(f"[GPT] Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        B, T = idx.shape
        x    = self.emb_drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, self.freqs_cis[:T])
        logits = self.lm_head(self.norm_f(x))
        loss   = (
            F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if targets is not None else None
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.seq_len:])
            logits    = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx