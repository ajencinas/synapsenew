#!/usr/bin/env python3
"""SFT data loading + masked-loss eval utilities (adapted from the original
synapse sft/sft.py so numbers stay comparable with the v1/v2/v3 runs)."""
import math
import os

import torch
import torch.nn.functional as F

from common import (IGNORE_INDEX, PAD_ID, SFT_TOKENIZED_DIR, load_jsonl)

# v3 15-source mix (from synapse sft/sft.py — kept identical on purpose so this
# run's data distribution matches the reference runs).
SFT_DATA_MIX = {
    "tulu3":             0.340,
    "dolly":             0.110,
    "no_robots":         0.080,
    "tool_use":          0.075,
    "tool_negative":     0.060,
    "metamath":          0.050,
    "reasoning_distill": 0.050,
    "samsum":            0.045,
    "alpaca_gpt4":       0.045,
    "format_following":  0.040,
    "opencode":          0.025,
    "tool_search":       0.025,
    "nli":               0.020,
    "oasst1":            0.018,
    "creative":          0.017,
}
MAX_EPOCH_REPEAT = 1.65

CACHE_NAME = "_synapsenew_cache.pt"


def load_all_data(tok_id, use_cache=True, data_dir=SFT_TOKENIZED_DIR):
    """(train_examples, per-example weights, val_by_source) with a signature
    cache keyed on source sizes + mtimes (refreshes when data changes)."""
    from common import tokenizer_fingerprint  # noqa: F401  (doc only)
    cache_path = os.path.join(data_dir, CACHE_NAME)

    sig_parts = []
    for name in sorted(SFT_DATA_MIX):
        p = os.path.join(data_dir, name, "train.jsonl")
        if not os.path.exists(p):
            raise SystemExit(f"[{name}] missing {p} — pull sft_tokenized first")
        sig_parts.append(f"{name}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}")
    signature = "|".join(sig_parts)

    if use_cache and os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path, weights_only=False)
            if cache.get("signature") == signature:
                print(f"[data] cache hit → {cache_path}")
                return cache["train"], cache["weights"], cache["val"]
        except Exception as e:
            print(f"[data] cache unreadable ({e}) — rebuilding")
        print("[data] cache stale — rebuilding")

    train, weights, val_by_source = [], [], {}
    counts = {}
    print("[data] parsing per-source JSONL:")
    for name, mix_w in SFT_DATA_MIX.items():
        d = os.path.join(data_dir, name)
        tr = load_jsonl(os.path.join(d, "train.jsonl"))
        val_by_source[name] = load_jsonl(os.path.join(d, "val.jsonl"))
        if not tr:
            raise SystemExit(f"[{name}] 0 train examples")
        train.extend(tr)
        weights.extend([mix_w / len(tr)] * len(tr))
        counts[name] = len(tr)
        print(f"  {name:18s} train={len(tr):>7,} val={len(val_by_source[name]):>4,} w={mix_w:.3f}")

    P = len(train)
    for name, mix_w in SFT_DATA_MIX.items():
        repeat = mix_w * P / counts[name]
        if repeat > MAX_EPOCH_REPEAT:
            raise SystemExit(f"[{name}] over-drawn {repeat:.2f}x/epoch (> {MAX_EPOCH_REPEAT})")
    if use_cache:
        torch.save({"signature": signature, "train": train,
                    "weights": weights, "val": val_by_source}, cache_path)
        print(f"[data] cached → {cache_path}")
    return train, weights, val_by_source


def collate(batch):
    """Right-pad to batch max; labels padded with IGNORE (pads never learned)."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    n = len(batch)
    input_ids = torch.full((n, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(ex["labels"], dtype=torch.long)
    return input_ids, labels


def collate_prompt_completion(pairs):
    """collate for {"prompt_ids","completion_ids"} pairs (DPO data): returns
    seq ids + a mask True over completion-prediction positions, right-padded."""
    seqs, masks = [], []
    for ex in pairs:
        ids = list(ex["prompt_ids"]) + list(ex["completion_ids"])
        p = len(ex["prompt_ids"])
        m = [False] * len(ids)
        for j in range(max(p, 1), len(ids)):
            m[j] = True          # completion token j is predicted by position j-1
        seqs.append(ids)
        masks.append(m)
    max_len = max(len(s) for s in seqs)
    n = len(seqs)
    input_ids = torch.full((n, max_len), PAD_ID, dtype=torch.long)
    attn = torch.zeros((n, max_len), dtype=torch.long)
    # width max_len-1: position i predicts token i+1, so only the first L-1
    # positions can contribute (logits[:, :-1] is width max_len-1)
    mask = torch.zeros((n, max_len - 1), dtype=torch.bool)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        L = len(s)
        input_ids[i, :L] = torch.tensor(s, dtype=torch.long)
        attn[i, :L] = 1
        if L > 1:
            mask[i, :L - 1] = torch.tensor(m[1:], dtype=torch.bool)  # position i predicts i+1
    return input_ids, attn, mask


def shifted_ce(logits, labels, vocab_size, reduction):
    """Logits[i] predict token i+1; labels are aligned → shift for the loss."""
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, vocab_size),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=IGNORE_INDEX, reduction=reduction)


@torch.no_grad()
def eval_source(model, val_examples, vocab_size, cap=0, batch_size=8, device="cuda"):
    """Token-weighted mean response-token loss; `cap` limits examples for
    faster in-training evals (0 = all).  Gradient checkpointing must be OFF for
    evals: it disables inference_mode-compatible no-grad paths and leaves a
    graph in memory, which OOMs on a full VRAM budget."""
    was_training = model.training
    model.eval()
    # raw module: torch.compile wrappers don't expose our custom attribute
    raw = getattr(model, "_orig_mod", model)
    was_ckpt = getattr(raw, "gradient_checkpointing", False)
    if was_ckpt:
        raw.gradient_checkpointing = False
    try:
        examples = val_examples[:cap] if cap else val_examples
        total_loss, total_tokens = 0.0, 0
        for start in range(0, len(examples), batch_size):
            input_ids, labels = collate(examples[start:start + batch_size])
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = shifted_ce(logits, labels, vocab_size, reduction="sum")
            total_loss += loss.item()
            total_tokens += int((labels[:, 1:] != IGNORE_INDEX).sum().item())
        return total_loss / total_tokens if total_tokens else float("nan")
    finally:
        if was_ckpt:
            raw.gradient_checkpointing = True
        if was_training:
            model.train()


def eval_all(model, val_by_source, vocab_size, cap=0, batch_size=8, device="cuda"):
    per_source = {name: eval_source(model, ex, vocab_size, cap=cap,
                                    batch_size=batch_size, device=device)
                  for name, ex in val_by_source.items()}
    vals = [v for v in per_source.values() if not math.isnan(v)]
    per_source["overall"] = sum(vals) / len(vals) if vals else float("nan")
    return per_source
