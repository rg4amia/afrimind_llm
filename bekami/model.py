import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight)

def rope_tables(head_dim, max_len, theta, factor=1.0):
    # NTK-aware : un facteur > 1 agrandit theta pour étendre le contexte sans réentraîner de zéro
    if factor > 1.0:
        theta = theta * factor ** (head_dim / (head_dim - 2))
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    # x: [B, heads, T, head_dim], cos/sin: [T, head_dim/2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

class Attention(nn.Module):
    """GQA + QK-Norm. Les couches 'locales' voient une fenêtre glissante, une couche sur
    `global_every` voit tout le contexte."""
    def __init__(self, cfg, layer_idx):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads, self.n_kv = cfg.n_heads, cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * self.n_kv * self.head_dim, bias=False)
        self.proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        is_global = cfg.window_size <= 0 or (layer_idx + 1) % cfg.global_every == 0
        self.window = None if is_global else cfg.window_size
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, pos0=0, cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.n_kv, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, cos[pos0:pos0 + T], sin[pos0:pos0 + T])
        k = apply_rope(k, cos[pos0:pos0 + T], sin[pos0:pos0 + T])
        if cache is not None and cache[0] is not None:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        if cache is not None:
            keep = self.window or k.size(2)
            cache[0], cache[1] = k[:, :, -keep:], v[:, :, -keep:]
        k_rep = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v_rep = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)

        Tk = k.size(2)
        drop = self.dropout if self.training else 0.0
        if pos0 == 0 and Tk == T and self.window is None:
            y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True, dropout_p=drop)
        elif T == 1 and self.window is None:
            y = F.scaled_dot_product_attention(q, k_rep, v_rep, dropout_p=drop)
        else:
            qpos = torch.arange(pos0, pos0 + T, device=x.device)[:, None]
            kpos = torch.arange(pos0 + T - Tk, pos0 + T, device=x.device)[None, :]
            mask = kpos <= qpos
            if self.window is not None:
                mask &= (qpos - kpos) < self.window
            y = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask, dropout_p=drop)
        return self.proj(y.transpose(1, 2).reshape(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = (int(8 * dim / 3) + 63) // 64 * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model)

    def forward(self, x, cos, sin, pos0=0, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, pos0, cache)
        x = x + self.mlp(self.norm2(x))
        return x

class GPTConfig:
    DEFAULTS = dict(
        vocab_size=16000, context_length=1024, d_model=384, n_layers=6, n_heads=6,
        n_kv_heads=2, dropout=0.0, rope_theta=500000.0, rope_factor=1.0,
        qk_norm=True, window_size=0, global_every=6, tie_embeddings=True,
    )
    def __init__(self, **kwargs):
        self.__dict__.update(self.DEFAULTS)
        self.__dict__.update(kwargs)

class BekamiGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = rope_tables(head_dim, cfg.context_length, cfg.rope_theta, cfg.rope_factor)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # sorties des résidus réduites pour garder une variance stable avec la profondeur
        for block in self.blocks:
            for w in (block.attn.proj.weight, block.mlp.w3.weight):
                nn.init.normal_(w, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, pos0=0, last_only=False):
        """`caches` : liste de [k, v] (une par couche, initialisée à [None, None]), modifiée sur place."""
        x = self.token_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope_cos, self.rope_sin, pos0, caches[i] if caches else None)
        x = self.norm_f(x[:, -1:] if last_only else x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40):
        self.eval()
        idx = idx[:, -self.cfg.context_length:]
        max_new_tokens = min(max_new_tokens, self.cfg.context_length - idx.size(1))
        caches = [[None, None] for _ in self.blocks]
        logits, _ = self(idx, caches=caches, pos0=0, last_only=True)
        pos = idx.size(1)
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = -float("inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            logits, _ = self(next_token, caches=caches, pos0=pos)
            pos += 1
        return idx
