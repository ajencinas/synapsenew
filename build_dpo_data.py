#!/usr/bin/env python3
"""Build DPO preference pairs for the RL stage — entirely from Google-Drive
(then rclone-pulled) SFT data. No external preference dataset needed.

Pair construction (all from the tokenized 15-source SFT mix):
  prompt   = input_ids up to the first learned (assistant) token
  chosen   = the curated assistant response (learned region, incl <im_end>)
  rejected = ONE of:
    * "swap"    — for ultra-short answers (nli yes/no etc.): another random
                  chosen response from the same source (usually the wrong
                  answer — a clean negative)
    * "corrupt" — truncated/garbled chosen (truncated to 55%, or a repeated
                  fragment) — guaranteed worse than the full curated answer
    * "gen"     — the PRETRAIN base model's own continuation of the prompt
                  (phase 2, GPU): real base-model failure text

Phase 1 (CPU) writes pairs with swap/corrupt rejects + a candidate list for
gen rejects. Phase 2 (GPU, --gen-ckpt) fills the gen rejects with the base
model, falling back to corrupt if generation fails.

Usage:
  python build_dpo_data.py --out-dir $DPO_DIR --n-train 6000 --n-val 300
  python build_dpo_data.py ... --gen-ckpt $PRETRAIN_CKPT   # phase 2
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import common
from common import DPO_DIR, IGNORE_INDEX, REPORT_DIR, load_jsonl, write_json
from data_utils import SFT_DATA_MIX

# DPO sampling mix — tilts toward sources where "wrong answer" semantics are
# crisply defined (math, format, NLI, tools) vs pure prose.
DPO_MIX = {
    "metamath":          0.12,
    "reasoning_distill": 0.12,
    "format_following":  0.10,
    "nli":               0.08,
    "samsum":            0.08,
    "tool_negative":     0.08,
    "tool_search":       0.06,
    "tool_use":          0.08,
    "tulu3":             0.10,
    "dolly":             0.05,
    "no_robots":         0.02,
    "alpaca_gpt4":       0.03,
    "opencode":          0.04,
    "oasst1":            0.02,
    "creative":          0.02,
}
GEN_SOURCES = ("metamath", "format_following", "samsum", "tool_negative",
               "tool_search", "tool_use", "alpaca_gpt4")

MAX_PROMPT = 768      # left-truncate longer prompts
MAX_CHOSEN = 512      # skip pairs with longer curated answers (keeps gen cheap)


def first_learned(labels):
    for i, l in enumerate(labels):
        if l != IGNORE_INDEX:
            return i
    return None


def corrupt(chosen, rng):
    """Truncation-family corruption — always an incomplete/degraded answer."""
    L = len(chosen)
    if L >= 10:
        k = rng.randint(max(4, int(L * 0.4)), max(5, int(L * 0.65)))
        return chosen[:k], "corrupt_trunc"
    # ultra short: drop the closing marker if present → "unfinished" answer
    return chosen[:-1] if L > 2 else chosen, "corrupt_short"


def build_phase1(out_dir, n_train, n_val, seed, gen_frac):
    rng = random.Random(seed)
    stats = {}
    for split, n in (("train", n_train), ("val", n_val)):
        rows, gen_candidates = [], []
        names = list(DPO_MIX)
        weights = [DPO_MIX[k] for k in names]
        pools = {}
        for name in names:
            src = os.path.join(common.SFT_TOKENIZED_DIR, name,
                               "train.jsonl" if split == "train" else "val.jsonl")
            pools[name] = load_jsonl(src)
            stats[f"{split}/{name}"] = len(pools[name])
        # per-source chosen-tuple pools for swap rejects
        chosen_pools = {name: [ex["input_ids"][first_learned(ex["labels"]):]
                               for ex in pools[name][:600] if first_learned(ex["labels"])]
                        for name in names}
        for _ in range(n):
            name = rng.choices(names, weights=weights, k=1)[0]
            ex = rng.choice(pools[name])
            f = first_learned(ex["labels"])
            if f is None or f >= len(ex["input_ids"]) - 2:
                continue
            prompt = ex["input_ids"][:f][-MAX_PROMPT:]
            chosen = ex["input_ids"][f:]
            if len(chosen) > MAX_CHOSEN:
                continue
            pair = {"source": name, "prompt_ids": prompt, "chosen_ids": chosen,
                    "rejected_ids": None, "rej_kind": None}
            if len(chosen) < 14 and chosen_pools[name]:
                for _try in range(8):
                    alt = rng.choice(chosen_pools[name])
                    if alt != chosen:
                        pair["rejected_ids"] = alt[:MAX_CHOSEN]
                        pair["rej_kind"] = "swap"
                        break
            if pair["rejected_ids"] is None:
                r, kind = corrupt(chosen, rng)
                pair["rejected_ids"] = r
                pair["rej_kind"] = kind
            if (split == "train" and name in GEN_SOURCES
                    and len(chosen) <= 160 and rng.random() < gen_frac):
                gen_candidates.append(len(rows))
            rows.append(pair)
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        if split == "train":
            write_json(os.path.join(out_dir, "gen_candidates.json"),
                       {"rows": gen_candidates})
        print(f"[phase1] {split}: {len(rows)} pairs "
              f"({sum(1 for r in rows if r['rej_kind']=='swap')} swap, "
              f"{sum(1 for r in rows if r['rej_kind'] and r['rej_kind'].startswith('corrupt'))} corrupt), "
              f"{len(gen_candidates)} gen candidates")
    write_json(os.path.join(REPORT_DIR, "dpo_data_stats.json"),
               {"phase1": stats, "dpo_mix": DPO_MIX, "gen_sources": list(GEN_SOURCES)})


def build_phase2(out_dir, gen_ckpt, device="cuda", max_new=128):
    from infer_model import SynapseInfer
    from common import load_tokenizer, resolve_special_ids

    path = os.path.join(out_dir, "train.jsonl")
    rows = load_jsonl(path)
    cand = json.load(open(os.path.join(out_dir, "gen_candidates.json")))["rows"]
    cand = [i for i in cand if i < len(rows)]
    print(f"[phase2] generating base-model rejects for {len(cand)} candidates")

    tok = load_tokenizer()
    _, im_end_id, eot_id = resolve_special_ids(tok)
    model, _ = SynapseInfer.from_checkpoint(gen_ckpt, device=device, no_compile=True)
    model.eval()

    t0 = time.time()
    n_done = 0
    for j, idx in enumerate(cand):
        row = rows[idx]
        ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
        out = []
        try:
            for tid in model.generate(ids, max_new, temperature=0.8, top_k=0,
                                      top_p=0.92, repetition_penalty=1.05,
                                      eot_id=eot_id, stop_tokens={im_end_id, eot_id}):
                out.append(tid)
        except Exception as e:
            print(f"  [gen] pair {idx} failed: {e}")
        if out and len(out) >= 3 and out != row["chosen_ids"]:
            row["rejected_ids"] = out[:MAX_CHOSEN]
            row["rej_kind"] = "gen"
        else:
            rng = random.Random(idx)
            row["rejected_ids"], row["rej_kind"] = corrupt(row["chosen_ids"], rng)
        n_done += 1
        if (j + 1) % 25 == 0 or j + 1 == len(cand):
            dt = time.time() - t0
            print(f"  [gen] {j+1}/{len(cand)} | {dt:.0f}s | "
                  f"{dt/(j+1):.2f}s/pair | eta {dt/(j+1)*(len(cand)-j-1)/60:.1f}min",
                  flush=True)
        if device == "cuda" and (j + 1) % 200 == 0:
            torch.cuda.empty_cache()

    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_gen = sum(1 for r in rows if r["rej_kind"] == "gen")
    print(f"[phase2] done: {n_gen} gen rejects written ({n_gen/len(rows)*100:.1f}%)")
    st = json.load(open(os.path.join(REPORT_DIR, "dpo_data_stats.json")))
    st["phase2"] = {"gen_rejects": n_gen, "total": len(rows),
                    "wall_min": round((time.time() - t0) / 60, 1)}
    write_json(os.path.join(REPORT_DIR, "dpo_data_stats.json"), st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DPO_DIR)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--gen-frac", type=float, default=0.30)
    ap.add_argument("--gen-ckpt", default=None)
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.gen_ckpt is None:
        common.guard_tokenization()
        build_phase1(args.out_dir, args.n_train, args.n_val, args.seed,
                     args.gen_frac)
    else:
        build_phase2(args.out_dir, args.gen_ckpt, max_new=args.max_new)


if __name__ == "__main__":
    main()
