"""
Masked Diffusion Language Model (from scratch, PyTorch).

A tiny implementation of the *diffusion LM* paradigm behind September 2026's
headline releases (Inception Mercury 2.5, IFM Uno / K2-Horizon): instead of
generating text left-to-right, the model learns to *denoise* — starting from
a fully masked sequence and progressively unmasking tokens in parallel.

Key ideas demonstrated here:
  1. Absorbing-state (masked) diffusion: the forward process masks tokens;
     the model learns the reverse process (predict the masked tokens).
  2. Bidirectional attention: unlike autoregressive GPTs, every position can
     attend to every other position (no causal mask) — context flows both ways.
  3. Parallel decoding: at generation time we unmask many positions per step,
     so a 128-token sequence can be generated in ~16-32 *parallel* steps
     instead of 128 sequential ones.

References (for further reading):
  - LLaDA (Nie et al., 2025): "Large Language Diffusion Models"
  - SEDD (Lou et al., 2024): score-based discrete diffusion
  - IFM Uno (Sep 2026): diffusion adapter giving ~2x speedup, "provably
    lossless" on the vendor's own benchmarks
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = 0  # reserved id for the [MASK] token


# ---------------------------------------------------------------------------
# Tokenizer (character level — keeps the demo dependency-free and fast)
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        # id 0 is reserved for [MASK]; real chars start at 1
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i + 1: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars) + 1  # +1 for [MASK]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids if i != MASK_ID)


# ---------------------------------------------------------------------------
# Model: small transformer with *bidirectional* attention
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        # NOTE: no causal mask — diffusion LMs attend bidirectionally
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class DiffusionTransformer(nn.Module):
    """Predicts the clean token distribution at every position, given a
    partially masked sequence."""

    def __init__(self, vocab_size: int, d_model: int = 192, n_layer: int = 4,
                 n_head: int = 4, block_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(
            torch.arange(T, device=x.device))[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))  # (B, T, vocab)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training: masked-diffusion objective
# ---------------------------------------------------------------------------
@torch.no_grad()
def mask_batch(x0: torch.Tensor, t_max: float = 1.0, eps: float = 1e-3):
    """Forward diffusion: mask each token independently with prob t.

    t is sampled per-example from U[eps, t_max]. t_max < 1.0 focuses training
    on lightly-masked (high-context) examples, which is where a small model
    actually learns conditional structure; annealing t_max 0.3 -> 1.0 over
    training (a curriculum) lets it learn the easy cases first and still see
    the fully-masked regime used at generation time.
    """
    B, T = x0.shape
    t = torch.rand(B, 1, device=x0.device) * (t_max - eps) + eps  # (B,1)
    mask = torch.rand(B, T, device=x0.device) < t                 # (B,T) bool
    x_t = x0.clone()
    x_t[mask] = MASK_ID
    return x_t, t, mask


def diffusion_loss(model: DiffusionTransformer, x0: torch.Tensor,
                   t_max: float = 1.0, use_1_over_t: bool = False) -> torch.Tensor:
    """Masked-diffusion objective: cross-entropy on masked positions only.

    The literature (LLaDA) weights each example by 1/t to keep the estimator
    of the continuous-time ELBO unbiased. In theory that's the right thing;
    in practice at small batch sizes the weight (up to 1/eps = 1000x) makes
    gradients extremely noisy and training collapses to the marginal
    distribution. We default it OFF and instead use a t_max curriculum,
    which trains stably. Pass use_1_over_t=True to reproduce the instability
    yourself — it's an instructive failure mode.
    """
    x_t, t, mask = mask_batch(x0, t_max=t_max)
    logits = model(x_t)                       # (B,T,V)
    # ignore_index on unmasked positions -> loss only on masked ones
    targets = x0.clone()
    targets[~mask] = -100
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                         targets.reshape(-1),
                         reduction="none").reshape(x0.shape)
    n_masked = mask.sum(dim=1).clamp(min=1)
    per_seq = (ce * mask).sum(dim=1) / n_masked
    if use_1_over_t:
        per_seq = per_seq / t.squeeze(1)
    return per_seq.mean()


# ---------------------------------------------------------------------------
# Sampling: iterative parallel denoising
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model: DiffusionTransformer, tok: CharTokenizer, n_tokens: int,
             steps: int = 32, temperature: float = 1.0,
             seed: int | None = None,
             device: str = "cpu") -> str:
    """Generate text by iterative denoising.

    Start from all-[MASK]; at each step predict every masked position and
    permanently unmask the most confident ones (linear schedule). Fewer steps
    = faster but lower quality — the core speed/quality knob of diffusion LMs.
    """
    if n_tokens > model.block_size:
        raise ValueError(
            f"n_tokens={n_tokens} exceeds model block_size={model.block_size}; "
            "generate fewer tokens or train with a larger --block-size.")
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    model.eval()
    x = torch.full((1, n_tokens), MASK_ID, dtype=torch.long, device=device)
    # linear unmasking schedule: unmask ~n_tokens/steps new tokens per step
    for s in range(1, steps + 1):
        logits = model(x) / max(temperature, 1e-6)          # (1,T,V)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)                        # (1,T)
        masked = x == MASK_ID
        n_masked = int(masked.sum())
        if n_masked == 0:
            break
        n_unmask = math.ceil(n_tokens * s / steps) - (n_tokens - n_masked)
        n_unmask = max(1, min(n_unmask, n_masked))
        # pick the n_unmask most confident *masked* positions
        conf_masked = conf.clone()
        conf_masked[~masked] = -1.0
        _, idx = torch.topk(conf_masked.flatten(), n_unmask)
        x.flatten()[idx] = pred.flatten()[idx]
    return tok.decode(x[0].tolist())


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: str):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    return x.to(device)
