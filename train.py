"""Train the tiny masked-diffusion LM on Tiny Shakespeare (CPU-friendly).

Usage:
    python train.py --steps 3000 --out ckpt.pt
"""
import argparse
import time

import torch

from diffusion_lm import (CharTokenizer, DiffusionTransformer, diffusion_loss,
                          get_batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"chars: {len(text):,}  tokens: {len(data):,}  "
          f"vocab: {tok.vocab_size}")

    model = DiffusionTransformer(tok.vocab_size, d_model=args.d_model,
                                 n_layer=args.n_layer, n_head=args.n_head,
                                 block_size=args.block_size)
    print(f"params: {model.n_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x0 = get_batch(data, args.block_size, args.batch_size, "cpu")
        loss = diffusion_loss(model, x0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
                  f"{dt:.0f}s elapsed")

    torch.save({"model": model.state_dict(), "chars": sorted(set(text)),
                "config": vars(args)}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
