"""Train the tiny masked-diffusion LM on Tiny Shakespeare (CPU-friendly).

Usage:
    python train.py --steps 3000 --out ckpt.pt
"""
import argparse
import os
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
    ap.add_argument("--t-max-start", type=float, default=0.3,
                    help="masking curriculum: t_max at step 0")
    ap.add_argument("--t-max-end", type=float, default=1.0,
                    help="masking curriculum: t_max at final step")
    ap.add_argument("--weight-1-over-t", action="store_true",
                    help="use the (unstable at small scale) 1/t weighting")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save a checkpoint every N steps (0 = only at end)")
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

    def save_ckpt():
        # atomic write so an interrupted run never leaves a corrupt file
        tmp = args.out + ".tmp"
        torch.save({"model": model.state_dict(), "chars": sorted(set(text)),
                    "config": vars(args)}, tmp)
        os.replace(tmp, args.out)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        # masking curriculum: start with easy (high-context) examples,
        # anneal toward the full range including fully-masked inputs
        frac = step / args.steps
        t_max = args.t_max_start + (args.t_max_end - args.t_max_start) * frac
        x0 = get_batch(data, args.block_size, args.batch_size, "cpu")
        loss = diffusion_loss(model, x0, t_max=t_max,
                              use_1_over_t=args.weight_1_over_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
                  f"{dt:.0f}s elapsed", flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_ckpt()

    save_ckpt()
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
