"""Sample from a trained diffusion-LM checkpoint.

Usage:
    python sample.py --ckpt ckpt.pt --tokens 200 --steps 32
Compare speed/quality trade-offs, e.g.:
    python sample.py --ckpt ckpt.pt --steps 8
    python sample.py --ckpt ckpt.pt --steps 64
"""
import argparse
import time

import torch

from diffusion_lm import CharTokenizer, DiffusionTransformer, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--tokens", type=int, default=128,
                    help="tokens to generate (must be <= training --block-size)")
    ap.add_argument("--steps", type=int, default=32,
                    help="denoising steps (fewer = faster, worse)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tok = CharTokenizer("".join(ckpt["chars"]))
    cfg = ckpt["config"]
    model = DiffusionTransformer(tok.vocab_size, d_model=cfg["d_model"],
                                 n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                                 block_size=cfg["block_size"])
    model.load_state_dict(ckpt["model"])

    t0 = time.time()
    text = generate(model, tok, args.tokens, steps=args.steps,
                    temperature=args.temperature, seed=args.seed)
    dt = time.time() - t0
    print(f"--- {args.tokens} tokens in {args.steps} parallel steps "
          f"({dt:.2f}s, ~{args.tokens / dt:.0f} tok/s on CPU) ---")
    print(text)


if __name__ == "__main__":
    main()
