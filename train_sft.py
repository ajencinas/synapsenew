#!/usr/bin/env python3
"""synapsenew SFT — LoRA fine-tune of the SynapseGPT 2.1B pretrain checkpoint on
the 15-source ChatML SFT mix, sized for a 16 GB laptop GPU.

Recipe notes (see README for the full improvement story):
  - same data mix + masked shifted-CE + weighted sampling as the reference
    v1/v2/v3 SFT runs → numbers stay comparable
  - LoRA (r=64) on q/k/v/o + SwiGLU instead of full FT → fits 16 GB, and avoids
    the bf16-param update-rounding problem that would silently freeze a small-LR
    full fine-tune
  - gradient checkpointing + bf16 base + fused AdamW on adapters only
  - token-normalized gradient accumulation (short batches don't overweight)
  - best-overall-val checkpointing (v3 lesson: val bottoms before the end)
  - budgeted cosine horizon: the LR schedule completes inside tonight's budget
    (v3 eval history shows ~1200 of 2938 steps captures the whole gain)

Saves merged (weights-only-delta) checkpoints that load directly into the
existing SynapseInfer / sft_bench tooling.

Run:  SYNAPSENEW_DIR=/media/alfonso/shared/synapsenew python train_sft.py
Env knobs: SFT_STEPS, SFT_MAX_LR, SFT_BATCH, SFT_ACCUM, SFT_COMPILE,
           SFT_EVAL_EVERY, SFT_SAVE_EVERY, SFT_EVAL_CAP, SFT_DEADLINE_TS,
           SFT_PROBE=1 (6-step speed/memory probe), SFT_RESUME=1
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import common
from common import (CKPT_DIR, REPORT_DIR, IGNORE_INDEX, guard_tokenization,
                    load_state_dict_from_checkpoint, load_tokenizer, write_json)
from data_utils import collate, eval_all, load_all_data, shifted_ce
from lora import (inject_lora, load_lora_state_dict, lora_parameters,
                  lora_state_dict, merged_state_dict)
from synapse_model import SynapseGPT


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SFT_STEPS     = int(os.environ.get("SFT_STEPS", 1200))       # optimizer steps
WARMUP_STEPS  = int(os.environ.get("SFT_WARMUP", 100))
MAX_LR        = float(os.environ.get("SFT_MAX_LR", 1e-4))
MIN_LR        = float(os.environ.get("SFT_MIN_LR", 1e-5))
WEIGHT_DECAY  = 0.01
BETAS         = (0.9, 0.95)
GRAD_CLIP     = 1.0

BATCH_SIZE       = int(os.environ.get("SFT_BATCH", 4))       # micro-batch
GRAD_ACCUM_STEPS = int(os.environ.get("SFT_ACCUM", 16))      # effective 64 seq

LORA_R      = int(os.environ.get("SFT_LORA_R", 64))
LORA_ALPHA  = LORA_R
LORA_DROP   = 0.05

EVAL_EVERY    = int(os.environ.get("SFT_EVAL_EVERY", 200))
EVAL_CAP      = int(os.environ.get("SFT_EVAL_CAP", 128))     # in-training
SAVE_EVERY    = int(os.environ.get("SFT_SAVE_EVERY", 200))
LOG_EVERY     = 20
USE_COMPILE   = os.environ.get("SFT_COMPILE", "1") == "1"
PROBE         = os.environ.get("SFT_PROBE", "0") == "1"
RESUME        = os.environ.get("SFT_RESUME", "0") == "1"
DEADLINE_TS   = float(os.environ.get("SFT_DEADLINE_TS", 0))  # 0 = no deadline
SEED          = 1337

# Output naming: the default paths are the original run; a tag keeps a second
# experiment (e.g. big-LoRA) from clobbering the first one's checkpoints.
TAG           = os.environ.get("SFT_TAG", "")
_LATEST_BASE  = os.environ.get("SFT_LATEST_NAME", f"sft_latest{'_'+TAG if TAG else ''}.pth")
_BEST_BASE    = os.environ.get("SFT_BEST_NAME", f"sft_best{'_'+TAG if TAG else ''}.pth")
_ADPT_BASE    = os.environ.get("SFT_ADAPTER_NAME", f"sft_adapter{'_'+TAG if TAG else ''}.pth")
LATEST_PATH  = os.path.join(CKPT_DIR, _LATEST_BASE)
BEST_PATH    = os.path.join(CKPT_DIR, _BEST_BASE)
ADAPTER_PATH = os.path.join(CKPT_DIR, _ADPT_BASE)
STATUS_PATH  = os.path.join(REPORT_DIR, f"sft_status{'_'+TAG if TAG else ''}.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lr(it):
    if it < WARMUP_STEPS:
        return MAX_LR * it / WARMUP_STEPS
    ratio = (it - WARMUP_STEPS) / max(1, SFT_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, ratio)))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def save_adapter(path, model, optimizer, curr_step, eval_history, best_val,
                 baseline=None):
    payload = {
        "schema": "synapsenew_sft_adapter_v1",
        "lora": lora_state_dict(model),
        "cfg": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROP,
                "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2"]},
        "optimizer": optimizer.state_dict(),
        "curr_step": curr_step, "total_steps": SFT_STEPS,
        "eval_history": eval_history, "best_val": best_val,
        "baseline_eval": baseline, "tokenization_id": common.EXPECTED_TOK_ID,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_merged(path, model, meta):
    sd = merged_state_dict(model)
    payload = {"schema": "v2", "stage": "sft", "model": sd, **meta}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  saved merged → {path} ({os.path.getsize(path)/1e9:.1f} GB)")


def status_update(**kw):
    try:
        st = {}
        if os.path.exists(STATUS_PATH):
            import json
            st = json.load(open(STATUS_PATH))
        st.update(kw)
        st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(STATUS_PATH, st)
    except Exception as e:
        print(f"[status] write failed: {e}")


def _grad_ckpt_enabled(model):
    return getattr(getattr(model, "_orig_mod", model), "gradient_checkpointing", False)


def _safe_eval(model, val_by_source, vocab_size, cap):
    """Eval with memory headroom: clear the caching allocator, drop to a small
    micro-batch, then restore. Prevents the periodic eval from OOMing on a
    laptop where training already sits near the VRAM limit."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return eval_all(model, val_by_source, vocab_size, cap=cap,
                    batch_size=2, device=device)


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok_id = guard_tokenization()
    print(f"[setup] tokenization_id OK → {tok_id}")

    # ---- model: pretrain weights, bf16, frozen, + LoRA ----
    print(f"[init] loading pretrain {common.PRETRAIN_CKPT} (mmap)")
    sd, _ = load_state_dict_from_checkpoint(common.PRETRAIN_CKPT)
    vocab_size = sd["token_embedding.weight"].shape[0]
    model = SynapseGPT(vocab_size)
    missing = set(model.state_dict()) - set(sd)
    if missing:
        raise SystemExit(f"checkpoint missing keys: {sorted(missing)[:8]}")
    model.load_state_dict(sd, strict=False)
    del sd
    model = model.bfloat16()                     # frozen base in bf16
    model.to(device)
    model.gradient_checkpointing = True
    inject_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROP,
                device=device, dtype=torch.float32)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] SynapseGPT {n_params/1e9:.2f}B params on {device} "
          f"(trainable: {sum(p.numel() for p in lora_parameters(model))/1e6:.1f}M)")

    optimizer = optim.AdamW(
        [{"params": lora_parameters(model), "weight_decay": WEIGHT_DECAY}],
        lr=MAX_LR, betas=BETAS, fused=torch.cuda.is_available())

    curr_step = 0
    eval_history, baseline_eval, best_val = [], None, float("inf")
    if RESUME and os.path.exists(ADAPTER_PATH) and os.path.exists(LATEST_PATH):
        ck = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
        if ck.get("tokenization_id") == tok_id:
            load_lora_state_dict(model, ck["lora"])
            optimizer.load_state_dict(ck["optimizer"])
            curr_step = int(ck.get("curr_step", 0))
            eval_history = list(ck.get("eval_history", []))
            best_val = float(ck.get("best_val", "inf"))
            baseline_eval = ck.get("baseline_eval")
            print(f"[resume] adapter step {curr_step} restored (best {best_val})")

    # ---- torch.compile (dynamic shapes; fallback to eager on any failure) ----
    compiled = False
    if USE_COMPILE:
        try:
            model = torch.compile(model, dynamic=True)
            compiled = True
            print("[compile] torch.compile(dynamic=True) enabled")
        except Exception as e:
            print(f"[compile] failed ({e}) — eager")

    # ---- data ----
    train_examples, weights, val_by_source = load_all_data(tok_id)
    n_val = sum(len(v) for v in val_by_source.values())
    print(f"[data] {len(train_examples):,} train | {n_val:,} val")

    if not (PROBE or curr_step or eval_history):
        t0 = time.time()
        baseline_eval = eval_all(model, val_by_source, vocab_size,
                                 cap=EVAL_CAP, device=device)
        print(f"[baseline] pre-SFT (cap {EVAL_CAP}) overall="
              f"{baseline_eval['overall']:.4f} [{time.time()-t0:.0f}s]")

    # ---- training loop ----
    n_steps = 6 if PROBE else SFT_STEPS
    gen = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(weights,
                                    num_samples=(n_steps - curr_step) * BATCH_SIZE * GRAD_ACCUM_STEPS,
                                    replacement=True, generator=gen)
    loader = DataLoader(train_examples, batch_size=BATCH_SIZE, sampler=sampler,
                        collate_fn=collate, num_workers=2,
                        pin_memory=(device.type == "cuda"), drop_last=False)

    step_times, window_times = [], []
    micro_in_window = 0
    window_loss, window_tokens = 0.0, 0
    log_loss, grad_norm, lr = float("nan"), 0.0, 0.0
    t0 = time.time()
    t_last_window = time.time()
    stopped_early = None

    print(f"\nSTARTING SFT: steps={n_steps} warmup={WARMUP_STEPS} lr={MAX_LR:.1e}->"
          f"{MIN_LR:.1e} micro={BATCH_SIZE}x{GRAD_ACCUM_STEPS} compile={compiled}\n")

    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = shifted_ce(logits, labels, vocab_size, reduction="sum")
        loss.backward()
        window_loss += loss.item()
        window_tokens += int((labels[:, 1:] != IGNORE_INDEX).sum().item())
        micro_in_window += 1

        if micro_in_window == GRAD_ACCUM_STEPS:
            if window_tokens == 0:
                raise SystemExit("accumulation window has 0 learned tokens")
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(window_tokens)
            lr = get_lr(curr_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_parameters(model), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            log_loss = window_loss / window_tokens
            micro_in_window = 0
            window_loss, window_tokens = 0.0, 0
            curr_step += 1

            dt = time.time() - t_last_window
            step_times.append(dt)
            t_last_window = time.time()

            if curr_step % LOG_EVERY == 0 or PROBE:
                recent = step_times[-LOG_EVERY:]
                s_per_step = sum(recent) / len(recent)
                eta_h = (n_steps - curr_step) * s_per_step / 3600
                mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                print(f"  step {curr_step}/{n_steps} loss={log_loss:.4f} "
                      f"lr={lr:.2e} gnorm={grad_norm:.2f} "
                      f"{s_per_step:.2f}s/step eta={eta_h:.2f}h mem={mem:.1f}GB",
                      flush=True)
                status_update(stage="sft", step=curr_step, total=n_steps,
                              loss=log_loss, lr=lr, s_per_step=s_per_step,
                              eta_hours=round(eta_h, 2), mem_gb=round(mem, 1),
                              best_val=best_val, elapsed_min=round((time.time()-t_start)/60, 1))

            if not PROBE and curr_step % SAVE_EVERY == 0:
                save_adapter(ADAPTER_PATH, model, optimizer, curr_step,
                             eval_history, best_val, baseline_eval)
                save_merged(LATEST_PATH, model,
                            {"epoch": 1, "curr_step": curr_step, "total_steps": SFT_STEPS,
                             "tokenization_id": tok_id, "mode": "lora",
                             "baseline_eval": baseline_eval,
                             "eval_history": eval_history})

            if not PROBE and curr_step % EVAL_EVERY == 0:
                t_e = time.time()
                val = _safe_eval(model, val_by_source, vocab_size, EVAL_CAP)
                val["step"] = curr_step
                eval_history.append(val)
                print(f"  [eval] step {curr_step} overall={val['overall']:.4f} "
                      f"[{time.time()-t_e:.0f}s] " +
                      " ".join(f"{k}={v:.3f}" for k, v in val.items()
                               if k in ("tulu3", "no_robots", "creative", "metamath")))
                if val["overall"] < best_val:
                    best_val = val["overall"]
                    save_merged(BEST_PATH, model,
                                {"epoch": 1, "curr_step": curr_step,
                                 "total_steps": SFT_STEPS, "tokenization_id": tok_id,
                                 "mode": "lora", "best_overall_val": best_val})
                    save_adapter(ADAPTER_PATH, model, optimizer, curr_step,
                                 eval_history, best_val, baseline_eval)
                    print(f"  [best] overall val {best_val:.4f} → sft_best.pth")
                write_json(os.path.join(REPORT_DIR, f"sft_eval_history{'_'+TAG if TAG else ''}.json"),
                           {"history": eval_history, "baseline": baseline_eval,
                            "steps": SFT_STEPS})

            if PROBE and curr_step >= 6:
                break
            if DEADLINE_TS and time.time() > DEADLINE_TS:
                stopped_early = f"deadline hit at step {curr_step}"
                print(f"[deadline] {stopped_early} — stopping cleanly")
                break

        if PROBE and curr_step >= 6 and micro_in_window == 0:
            break

    if PROBE:
        recent = step_times[-4:]
        s = sum(recent) / len(recent)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[PROBE] micro={BATCH_SIZE}x{GRAD_ACCUM_STEPS} compile={compiled} "
              f"→ {s:.2f}s/step, {64*276/s:.0f} tok/s est, peak mem {mem:.2f} GB")
        return

    # ---- final: full eval (uncapped) + durable saves ----
    stopped_early = stopped_early or "budget complete"
    print(f"\n[done] {stopped_early}; final full eval (uncapped val)…")
    val_full = _safe_eval(model, val_by_source, vocab_size, 0)
    val_full["step"] = curr_step
    eval_history.append(val_full)
    print("  [final] " + " ".join(f"{k}={v:.4f}" for k, v in val_full.items()))
    if val_full["overall"] < best_val:
        best_val = val_full["overall"]
        save_merged(BEST_PATH, model,
                    {"epoch": 1, "curr_step": curr_step, "total_steps": SFT_STEPS,
                     "tokenization_id": tok_id, "mode": "lora",
                     "best_overall_val": best_val})

    save_adapter(ADAPTER_PATH, model, optimizer, curr_step, eval_history,
                 best_val, baseline_eval)
    save_merged(LATEST_PATH, model,
                {"epoch": 1, "curr_step": curr_step, "total_steps": SFT_STEPS,
                 "tokenization_id": tok_id, "mode": "lora",
                 "baseline_eval": baseline_eval, "eval_history": eval_history,
                 "stopped_early": stopped_early})
    # adapter file must reflect the FINAL model (it may differ from best)
    write_json(os.path.join(REPORT_DIR, f"sft_training_latest{'_'+TAG if TAG else ''}.json"), {
        "stage": "sft", "mode": "lora", "tokenization_id": tok_id,
        "curr_step": curr_step, "total_steps": SFT_STEPS, "stopped_early": stopped_early,
        "hyperparameters": {"micro_batch": BATCH_SIZE, "grad_accum": GRAD_ACCUM_STEPS,
                            "max_lr": MAX_LR, "min_lr": MIN_LR, "warmup": WARMUP_STEPS,
                            "lora_r": LORA_R},
        "baseline_eval": baseline_eval, "final_eval": val_full,
        "best_overall_val": best_val, "eval_history": eval_history,
        "wall_hours": round((time.time() - t_start) / 3600, 2),
    })
    status_update(stage="sft_done", step=curr_step, best_val=best_val,
                  final_overall=val_full["overall"])
    print(f"\nSFT complete in {(time.time()-t_start)/3600:.2f}h | "
          f"best overall val={best_val:.4f} | ship sft_best.pth (and sft_latest.pth)")


if __name__ == "__main__":
    main()
