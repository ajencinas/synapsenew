#!/usr/bin/env python3
"""Weight-space model soup: average two fine-tunes of the same base checkpoint
(Wortsman et al., "Model Soups"). Both inputs must share the exact SynapseGPT
key set; embeddings stay tied after averaging.

Usage:
  python scripts/model_soup.py --a ckptA.pth --b ckptB.pth --alpha 0.5 --out soup.pth
(alpha = weight on A; 1-alpha on B)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common import load_state_dict_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[soup] A={args.a} (alpha={args.alpha})")
    print(f"[soup] B={args.b} (alpha={1-args.alpha})")
    a, _ = load_state_dict_from_checkpoint(args.a)
    b, _ = load_state_dict_from_checkpoint(args.b)
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))[:6]
        only_b = sorted(set(b) - set(a))[:6]
        raise SystemExit(f"key mismatch: only in A {only_a} / only in B {only_b}")

    out = {}
    for k in a:
        if a[k].shape != b[k].shape:
            raise SystemExit(f"shape mismatch on {k}: {a[k].shape} vs {b[k].shape}")
        out[k] = (a[k].float() * args.alpha + b[k].float() * (1.0 - args.alpha)
                  if a[k].is_floating_point() else a[k].clone())
    # keep the LM head tied to the embedding (save dedupes the shared storage)
    if "token_embedding.weight" in out and "lm_head.weight" in out:
        out["lm_head.weight"] = out["token_embedding.weight"]

    payload = {"schema": "v2", "stage": "sft", "model": out,
               "soup": {"a": os.path.abspath(args.a), "b": os.path.abspath(args.b),
                        "alpha": args.alpha}}
    tmp = args.out + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, args.out)
    print(f"[soup] wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
