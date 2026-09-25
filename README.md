# Weekly Build 01 — Diffusion Language Model, from scratch

A tiny **masked diffusion language model** trained from scratch in PyTorch —
no API keys, no pretrained weights, runs on CPU. It learns to generate text
by *denoising*: starting from a fully masked sequence and unmasking tokens in
parallel, instead of predicting left-to-right like GPT.

## Why this, why now

September 2026 has been the breakout month for diffusion LMs:

- **Inception Mercury 2.5** (announced Sep 8) — a diffusion LLM the vendor
  claims pushes past 1,100 tokens/sec in production by generating chunks of
  text in parallel rather than token-by-token.
- **IFM Uno** (Sep 17) — a diffusion *adapter* that bolts parallel decoding
  onto existing models; IFM reports ~2x throughput gains and says the method
  is already supported in SGLang.
- **K2 Horizon** (IFM) and others shipping the same week.

The bet behind all of them: the next-token bottleneck is architectural, not
just a scaling problem. This repo implements the core idea in ~250 lines so
you can feel how it actually works.

> Numbers above are vendor-reported (not independently verified at the time of
> writing) — treat them as the *motivation*, not the lesson.

## What this repo does

Trains a small character-level transformer (~1.8M params) on Tiny Shakespeare
with a masked-diffusion objective, then generates text by iterative parallel
denoising.

```
pip install -r requirements.txt        # CPU torch
python train.py --steps 3000           # ~30-60 min on CPU; see training notes below
python sample.py --ckpt ckpt.pt --steps 32
```

Compare the speed/quality knob directly:

```
python sample.py --ckpt ckpt.pt --steps 8    # fast, rough
python sample.py --ckpt ckpt.pt --steps 64   # slower, more coherent
```

## Key concepts

### 1. Autoregressive vs. diffusion generation

A GPT predicts token *t+1* given tokens *1..t*, so generating N tokens takes
N strictly sequential steps — each step waits for the last. A diffusion LM
instead learns a *denoising* process: start with `[MASK] [MASK] … [MASK]`,
predict **all** masked positions at once, keep the most confident
predictions, and repeat. A 128-token sequence can be produced in ~16–32
parallel steps instead of 128 sequential ones. That's the entire speed story.

### 2. The forward process: masking as diffusion

Diffusion models (images or text) define a *forward* process that destroys
signal and learn the *reverse* process that restores it. For text, the
simplest choice is **absorbing-state diffusion**: the forward process
replaces each token with `[MASK]` independently with probability `t`, where
`t ~ Uniform(0, 1)` per training example (`mask_batch()` in
`diffusion_lm.py`). No Gaussian noise schedules, no embeddings to corrupt —
just masking.

### 3. The training objective (and why the textbook version fails at small scale)

The model predicts the original token at every masked position; the loss is
cross-entropy **only on masked positions**. The literature (LLaDA) weights
each example by `1/t` to keep the estimator of the continuous-time ELBO
unbiased — at small `t` (little masking) each example reveals little about
the reverse process, so it gets up-weighted.

Here's the catch we hit building this repo: with `t ~ Uniform(0, 1)` and
tiny batches, the `1/t` weight (up to 1000x) makes gradients so noisy that
training **collapses to predicting the marginal character distribution**
— loss converges to exactly the corpus's marginal entropy (3.31 nats here),
and every sample comes out as spaces. We verified this is an optimization
pathology, not an architecture bug: the same model memorizes a fixed batch
to 100% accuracy.

What actually works at this scale is a **masking curriculum**: sample
`t ~ Uniform(0, t_max)` and anneal `t_max` from 0.3 → 1.0 over training
(`train.py` does this by default). Early on, the model sees high-context
examples where conditional structure is learnable; by the end it has seen
the fully-masked regime that generation starts from. This is a genuinely
useful lesson: diffusion LMs must learn denoising at *all* noise levels,
and the easy levels are where the signal is.

### 4. Bidirectional attention

Because the model must fill in a mask using context on *both* sides,
attention is **not causal** — every position attends to every other
position. Compare `Block` here with a GPT block: the only structural
difference is the missing causal mask. Everything else (embeddings,
LayerNorm, MLP) is identical, which is why adapters like IFM Uno can
retrofit this onto existing models.

### 5. Parallel decoding: confidence-based unmasking

`generate()` implements the standard sampler: at each step, score all masked
positions, permanently unmask the top-k most confident (linear schedule),
repeat. Early steps lock in easy tokens (spaces, common letters); later
steps resolve the hard ones with full bidirectional context. The number of
steps is the quality/latency dial — and unlike speculative decoding, there is
no draft model and no wasted work.

## Sample output

Trained for 2500 steps on Tiny Shakespeare (~1.8M params, CPU-only). Same
seed, different denoising budgets — note the speed/quality tradeoff:

**8 parallel steps** (0.15s, ~876 tok/s on CPU):
```
r o  e  our the e     to to he s   he toe toue sthe tathe athe the he the he he e tor othe the e the so   ro tto     tooeoeethe
```

**32 parallel steps** (0.41s, ~314 tok/s on CPU):
```
 he othe he the the the ithe hst e he ithe the e the hthe athe the he ithe e he he at t  e othto tre thre he he he e the thathe
```

4x the steps buys visibly more coherent text for only ~2.7x the wall-clock
time — each step is a single parallel forward pass, which is the whole
economic argument for diffusion LMs. A bigger model and longer training
would sharpen this further; the trajectory is what matters here.

## Experiments to try

1. **Steps vs. quality**: fix a seed, sample with `--steps 4/8/16/32/64`,
   and watch coherence improve while latency grows sub-linearly.
2. **Reproduce the marginal collapse**: train with
   `--t-max-start 1.0 --t-max-end 1.0` (no curriculum) and watch accuracy
   sit at the marginal baseline while loss converges to the corpus's
   marginal entropy. Then re-enable the curriculum and compare.
3. **The `1/t` instability**: train with `--weight-1-over-t` and watch the
   loss spike wildly as tiny-`t` batches get 1000x weight. Think about what
   batch size you'd need for this to average out (this is why the big labs
   can use it and you probably shouldn't at this scale).
4. **Remasking strategy**: the sampler here never revisits an unmasked
   token. Try *low-confidence remasking* (re-mask the least confident
   predictions each step, à la LLaDA) and see if quality improves.
5. **Temperature**: `--temperature 0.5` vs `1.5` — how does it interact
   with confidence-based unmasking?

## Try this next

- **Word-level model**: swap the char tokenizer for a BPE tokenizer
  (`tokenizers` package) and train on your own corpus (docs, code, chat
  logs). Watch how the masking objective behaves on longer-range structure.
- **Infilling demo**: mask the *middle* of a real sentence and let the model
  fill it in — something autoregressive models fundamentally can't do
  without tricks. This is the killer app for editing/coding assistants.
- **Guided generation**: implement classifier-free guidance (train with
  occasional conditioning dropout, e.g. on a "speaker" label in the
  Shakespeare data) to steer style at sampling time.
- **Read the papers**: LLaDA (Nie et al., 2025), SEDD (Lou et al., 2024),
  and the IFM Uno report — then map each paper's claims onto the ~250
  lines in `diffusion_lm.py`.

## Files

| File | What it is |
|---|---|
| `diffusion_lm.py` | Model, masked-diffusion objective, parallel sampler |
| `train.py` | Training loop (CPU-friendly) |
| `sample.py` | Generate text from a checkpoint |
| `data/tinyshakespeare.txt` | Training corpus (~1.1M chars) |
| `requirements.txt` | `torch` (CPU), `numpy`, `tqdm` |

## References

- LLaDA: Large Language Diffusion Models (Nie et al., 2025)
- SEDD: Score Entropy Discrete Diffusion (Lou et al., 2024)
- IFM Uno announcement & report (Sep 2026)
- Inception Mercury 2.5 announcement (Sep 2026)
