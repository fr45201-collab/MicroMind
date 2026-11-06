"""MicroMind model: a small decoder-only transformer."""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """LLaMA-style Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS and normalize
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0

        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.head_dim = config["n_embd"] // config["n_head"]

        # One Linear produces Q, K, V all at once (3x the embedding dim)
        self.qkv = nn.Linear(config["n_embd"], 3 * config["n_embd"], bias=False)
        # Output projection back to n_embd
        self.proj = nn.Linear(config["n_embd"], config["n_embd"], bias=False)

        # We keep dropout as a float for the SDPA call, and as a layer for the output
        self.attn_dropout_p = config["dropout"]
        self.resid_dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        B, T, C = x.shape

        # Project to Q, K, V and split
        qkv = self.qkv(x)                            # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)      # each (B, T, C)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 🚀 THE UPGRADE: PyTorch's fused attention kernel
        y = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True
        )

        # Apply attention to values
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.proj(y))


class SwiGLU(nn.Module):
    """LLaMA-style FeedForward using SwiGLU activation."""
    def __init__(self, config):
        super().__init__()
        n_embd = config["n_embd"]
        # SwiGLU uses 2/3 of the standard 4x multiplier to keep params efficient
        hidden_dim = int(8 * n_embd / 3)
        # Round to nearest multiple of 256 for hardware efficiency
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False) # Down
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False) # Up
        
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        # Swish(x) * Gate(x) -> w2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm FFN."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config["n_embd"])
        self.ffn = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual around attention
        x = x + self.ffn(self.ln2(x))    # residual around FFN
        return x


class MicroMind(nn.Module):
    """The full model: embeddings + transformer blocks + output head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.pos_emb = nn.Embedding(config["context_length"], config["n_embd"])
        self.drop = nn.Dropout(config["dropout"])

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config["n_layer"])])
        self.ln_f = RMSNorm(config["n_embd"])
        self.head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)

        # Weight tying: output head shares weights with token embedding
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config["context_length"], "sequence longer than context_length"

        positions = torch.arange(T, device=idx.device)       # (T,)
        x = self.token_emb(idx) + self.pos_emb(positions)    # (B, T, C)
        x = self.drop(x)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),            # (B*T, vocab)
                targets.view(-1),                            # (B*T,)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0):
        """Autoregressive sampling with optional top-k / top-p / repetition penalty."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["context_length"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature          # (B, vocab)

            # --- repetition penalty: down-weight tokens already in context ---
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    recent = set(idx[b].tolist())
                    for token_id in recent:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            # --- top-k: keep only k most likely tokens ---
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # --- top-p (nucleus): keep smallest set summing to p ---
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                # tokens to remove: cumulative prob > top_p (but keep the first one)
                remove = cumprobs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                # scatter back
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx