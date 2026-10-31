#!/usr/bin/env python3
"""synapsenew RL stage — DPO (Direct Preference Optimization) on the SFT model,
fitted to a 16 GB laptop GPU.

Setup:
  policy = SFT checkpoint + fresh LoRA adapters (r=64, trained here)
  ref    = the SAME module with adapters switched off → exactly the frozen
           start-of-DPO policy (the standard DPO reference), zero extra weights
  data   = preference pairs built by build_dpo_data.py (curated chosen vs
           base-model outputs / corrupted answers as rejected)

Loss: standard DPO (sigmoid loss on beta-scaled implicit-reward margins),
response-token log-probs summed, prompt tokens masked out.

Monitors (the "is it improving?" evidence):
  - val DPO accuracy: fraction of held-out pairs where the policy prefers the
    curated answer (>0.5 and rising = RL is working)
  - implicit-reward margin on val pairs
  - SFT-val LM loss on a few sources every few hundred steps (regression guard:
    RL must not wreck SFT abilities)

Run:  python train_dpo.py
Env:  DPO_EPOCHS, DPO_BETA, DPO_LR, DPO_ACCUM, DPO_EVAL_EVERY, DPO_DEADLINE_TS
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import common
from common import CKPT_DIR, DPO_DIR, REPORT_DIR, guard_tokenization, load_jsonl, write_json
from data_utils import collate_prompt_completion, eval_all, load_all_data
from lora import (inject_lora, load_lora_state_dict, lora_parameters,
                  lora_state_dict, merged_state_dict, set_lora_enabled)
from synapse_model import SynapseGPT

DPO_EPOCHS    = int(os.environ.get("DPO_EPOCHS", 3))
DPO_BETA      = float(os.environ.get("DPO_BETA", 0.1))
DPO_LR        = float(os.environ.get("DPO_LR", 5e-5))
DPO_MIN_LR    = DPO_LR * 0.1
DPO_WARMUP    = int(os.environ.get("DPO_WARMUP", 50))
DPO_ACCUM     = int(os.environ.get("DPO_ACCUM", 2))      # micro-batches / step
DPO_MICRO     = int(os.environ.get("DPO_MICRO", 4))      # pairs / micro-batch
DPO_MAXLEN    = int(os.environ.get("DPO_MAXLEN", 1024))
DPO_EVAL_EVERY = int(os.environ.get("DPO_EVAL_EVERY", 150))
DPO_SAVE_EVERY = int(os.environ.get("DPO_SAVE_EVERY", 300))
DPO_LM_EVERY   = int(os.environ.get("DPO_LM_EVERY", 300))
GRAD_CLIP     = 1.0
LORA_R        = int(os.environ.get("DPO_LORA_R", 64))
SEED          = int(os.environ.get("DPO_SEED", 777))
DEADLINE_TS   = float(os.environ.get("DPO_DEADLINE_TS", 0))
SFT_CKPT      = os.environ.get("DPO_INIT_CKPT", os.path.join(CKPT_DIR, "sft_best.pth"))
BEST_VAL_ACC_TO_STOP = 0.95

LATEST_PATH  = os.path.join(CKPT_DIR, "dpo_latest.pth")
BEST_PATH    = os.path.join(CKPT_DIR, "dpo_best.pth")
ADAPTER_PATH = os.path.join(CKPT_DIR, "dpo_adapter.pth")
STATUS_PATH  = os.path.join(REPORT_DIR, "dpo_status.json")

device = torch.device("cuda")

torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")


def trim_pair(pair):
    """Left-truncate prompt so prompt+completion fits DPO_MAXLEN."""
    p, c = list(pair["prompt_ids"]), list(pair["completion_ids"])
    if len(c) >= DPO_MAXLEN - 8:
        c = c[:DPO_MAXLEN - 8]
    keep = DPO_MAXLEN - len(c)
    if len(p) > keep:
        p = p[-keep:]
    return {"prompt_ids": p, "completion_ids": c}


def load_pairs(split):
    rows = load_jsonl(os.path.join(DPO_DIR, f"{split}.jsonl"))
    out = []
    for r in rows:
        if not r.get("rejected_ids") or not r.get("chosen_ids"):
            continue
        out.append(r)
    return out


def pair_batches(rows, micro, shuffle, rng):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), micro):
        yield [rows[j] for j in order[i:i + micro]]


def seq_logp(model, batch, grad):
    """Sum of response-token log-probs for a list of {prompt_ids, completion_ids}."""
    input_ids, _, mask = collate_prompt_completion(batch)
    input_ids, mask = input_ids.to(device), mask.to(device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
        logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        sel = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (sel * mask).sum(dim=1)          # (B,) fp32


def dpo_batch_stats(model, pairs, need_ref=True, policy_grad=True):
    """Return (loss, rewards_ch, rewards_rj) for a micro-batch of pairs."""
    ch = [trim_pair({"prompt_ids": p["prompt_ids"], "completion_ids": p["chosen_ids"]})
          for p in pairs]
    rj = [trim_pair({"prompt_ids": p["prompt_ids"], "completion_ids": p["rejected_ids"]})
          for p in pairs]
    if need_ref:
        set_lora_enabled(model, False)
        ref_ch = seq_logp(model, ch, grad=False).detach()
        ref_rj = seq_logp(model, rj, grad=False).detach()
        set_lora_enabled(model, True)
    else:
        ref_ch = ref_rj = torch.zeros(len(pairs), device=device)
    pg_ch = seq_logp(model, ch, grad=policy_grad)
    pg_rj = seq_logp(model, rj, grad=policy_grad)
    rew_ch = DPO_BETA * (pg_ch - ref_ch)
    rew_rj = DPO_BETA * (pg_rj - ref_rj)
    loss = -F.logsigmoid(rew_ch - rew_rj).mean()
    if not policy_grad:
        return loss, rew_ch, rew_rj
    return loss, rew_ch.detach(), rew_rj.detach()


def eval_pairs(model, pairs, micro=4, cap=0):
    model.eval()
    set_lora_enabled(model, True)
    rows = pairs[:cap] if cap else pairs
    accs, margins, ch_m, rj_m = [], [], [], []
    for i in range(0, len(rows), micro):
        batch = rows[i:i + micro]
        with torch.no_grad():
            _, r_ch, r_rj = dpo_batch_stats(model, batch, need_ref=True,
                                            policy_grad=False)
        accs.append((r_ch > r_rj).float().mean().item())
        margins.append((r_ch - r_rj).mean().item())
        ch_m.append(r_ch.mean().item())
        rj_m.append(r_rj.mean().item())
    model.train()
    n = len(rows)
    return {"val_acc": sum(accs) / len(accs), "val_margin": sum(margins) / len(margins),
            "mean_rew_chosen": sum(ch_m) / len(ch_m), "mean_rew_rejected": sum(rj_m) / len(rj_m),
            "n": n}


def save_adapter(path, model, optimizer, step, history, best_acc, epoch):
    torch.save({"schema": "synapsenew_dpo_adapter_v1",
                "lora": lora_state_dict(model), "optimizer": optimizer.state_dict(),
                "step": step, "epoch": epoch, "history": history,
                "best_val_acc": best_acc, "tokenization_id": common.EXPECTED_TOK_ID},
               path + ".tmp")
    os.replace(path + ".tmp", path)


def save_merged(path, model, meta):
    sd = merged_state_dict(model)
    tmp = path + ".tmp"
    torch.save({"schema": "v2", "stage": "sft", "model": sd, **meta}, tmp)
    os.replace(tmp, path)
    print(f"  saved merged → {path} ({os.path.getsize(path)/1e9:.1f} GB)")


def main():
    t_start = time.time()
    tok_id = guard_tokenization()
    torch.manual_seed(SEED)

    print(f"[init] loading SFT policy from {SFT_CKPT} (mmap)")
    from common import load_state_dict_from_checkpoint
    sd, meta = load_state_dict_from_checkpoint(SFT_CKPT)
    vocab_size = sd["token_embedding.weight"].shape[0]
    model = SynapseGPT(vocab_size)
    model.load_state_dict(sd, strict=False)
    del sd
    model = model.bfloat16().to(device)
    model.gradient_checkpointing = True
    inject_lora(model, r=LORA_R, alpha=LORA_R, dropout=0.0, device=device,
                dtype=torch.float32)
    model.train()
    print(f"[model] policy ready (trainable {sum(p.numel() for p in lora_parameters(model))/1e6:.1f}M)")

    optimizer = optim.AdamW(lora_parameters(model), lr=DPO_LR, betas=(0.9, 0.95),
                            weight_decay=0.01, fused=True)

    train_pairs = load_pairs("train")
    val_pairs = load_pairs("val")
    if not train_pairs or not val_pairs:
        raise SystemExit("DPO data missing — run build_dpo_data.py first")
    steps_per_epoch = math.ceil(len(train_pairs) / (DPO_MICRO * DPO_ACCUM))
    total_steps = DPO_EPOCHS * steps_per_epoch
    print(f"[data] {len(train_pairs)} train / {len(val_pairs)} val pairs | "
          f"{DPO_EPOCHS} ep x {steps_per_epoch} steps = {total_steps}")
    kind_counts = {}
    for r in train_pairs:
        kind_counts[r["rej_kind"]] = kind_counts.get(r["rej_kind"], 0) + 1
    print(f"[data] rejected kinds: {kind_counts}")

    # LM-drift guard data (few small sources, capped)
    _, _, val_by_source = load_all_data(tok_id, use_cache=True)
    drift_sources = ["tulu3", "no_robots", "creative"]
    drift_cap = 48
    drift0 = eval_all(model, {k: val_by_source[k] for k in drift_sources},
                      vocab_size, cap=drift_cap, device=device)
    print(f"[drift] pre-DPO LM loss: " +
          " ".join(f"{k}={v:.3f}" for k, v in drift0.items()))

    def get_lr(step, horizon):
        if step < DPO_WARMUP:
            return DPO_LR * step / DPO_WARMUP
        r = (step - DPO_WARMUP) / max(1, horizon - DPO_WARMUP)
        return DPO_MIN_LR + 0.5 * (1 + math.cos(math.pi * min(1.0, r))) * (DPO_LR - DPO_MIN_LR)

    history, best_acc = [], -1.0
    global_step = 0
    rng = torch.Generator().manual_seed(SEED)
    import random as pyrandom
    pyrng = pyrandom.Random(SEED)
    stop_reason = None

    for epoch in range(1, DPO_EPOCHS + 1):
        acc_loss, nb = 0.0, 0
        t0 = time.time()
        micro_in_window = 0
        for batch in pair_batches(train_pairs, DPO_MICRO, True, pyrng):
            loss, r_ch, r_rj = dpo_batch_stats(model, batch, need_ref=True)
            (loss / DPO_ACCUM).backward()
            acc_loss += loss.item()
            nb += 1
            micro_in_window += 1
            if micro_in_window == DPO_ACCUM:
                lr = get_lr(global_step, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                gn = torch.nn.utils.clip_grad_norm_(lora_parameters(model), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_in_window = 0
                global_step += 1
                if global_step % 20 == 0:
                    dt = time.time() - t0
                    acc = (r_ch > r_rj).float().mean().item()
                    print(f"  ep{epoch} step {global_step}/{total_steps} "
                          f"loss={acc_loss/nb:.4f} train_acc={acc:.2f} "
                          f"margin={(r_ch-r_rj).mean().item():.3f} lr={lr:.1e} "
                          f"gnorm={gn:.2f} ({dt/20:.1f}s/step-wrap)", flush=True)
                    acc_loss, nb = 0.0, 0
                    t0 = time.time()
                    write_json(STATUS_PATH, {
                        "stage": "dpo", "epoch": epoch, "step": global_step,
                        "total_steps": total_steps, "train_acc": acc,
                        "best_val_acc": best_acc,
                        "elapsed_min": round((time.time()-t_start)/60, 1),
                        "updated": time.strftime("%H:%M:%S")})

                if global_step % DPO_EVAL_EVERY == 0:
                    t_e = time.time()
                    ev = eval_pairs(model, val_pairs)
                    ev.update({"step": global_step, "epoch": epoch,
                               "lr": lr, "lm_drift": eval_all(
                                   model, {k: val_by_source[k] for k in drift_sources},
                                   vocab_size, cap=drift_cap, device=device)["overall"]})
                    history.append(ev)
                    print(f"  [eval] step {global_step} val_acc={ev['val_acc']:.3f} "
                          f"margin={ev['val_margin']:.3f} rew_ch={ev['mean_rew_chosen']:.3f} "
                          f"rew_rj={ev['mean_rew_rejected']:.3f} lm_drift={ev['lm_drift']:.3f} "
                          f"[{time.time()-t_e:.0f}s]")
                    write_json(os.path.join(REPORT_DIR, "dpo_eval_history.json"),
                               {"history": history, "pre_dpo_lm": drift0})
                    if ev["val_acc"] > best_acc:
                        best_acc = ev["val_acc"]
                        save_merged(BEST_PATH, model,
                                    {"epoch": epoch, "curr_step": global_step,
                                     "tokenization_id": tok_id, "mode": "lora_dpo",
                                     "best_val_acc": best_acc,
                                     "dpo_beta": DPO_BETA, "init_ckpt": SFT_CKPT})
                        save_adapter(ADAPTER_PATH, model, optimizer, global_step,
                                     history, best_acc, epoch)
                        print(f"  [best] val_acc {best_acc:.3f} → dpo_best.pth")
                    if ev["val_acc"] >= BEST_VAL_ACC_TO_STOP:
                        stop_reason = f"val_acc saturated at {ev['val_acc']:.3f}"
                        break

                if global_step % DPO_SAVE_EVERY == 0:
                    save_adapter(ADAPTER_PATH, model, optimizer, global_step,
                                 history, best_acc, epoch)
                    save_merged(LATEST_PATH, model,
                                {"epoch": epoch, "curr_step": global_step,
                                 "tokenization_id": tok_id, "mode": "lora_dpo",
                                 "best_val_acc": best_acc, "dpo_beta": DPO_BETA,
                                 "init_ckpt": SFT_CKPT, "history": history})

            if DEADLINE_TS and time.time() > DEADLINE_TS:
                stop_reason = f"deadline at step {global_step}"
                break
        if stop_reason:
            break

    # final: save latest + a full val pass
    stop_reason = stop_reason or "epochs complete"
    print(f"[done] {stop_reason}; final val pair eval…")
    ev = eval_pairs(model, val_pairs, cap=0)
    ev.update({"step": global_step, "epoch": DPO_EPOCHS, "final": True})
    history.append(ev)
    print(f"  [final] val_acc={ev['val_acc']:.3f} margin={ev['val_margin']:.3f}")
    if ev["val_acc"] > best_acc:
        best_acc = ev["val_acc"]
        save_merged(BEST_PATH, model,
                    {"epoch": DPO_EPOCHS, "curr_step": global_step,
                     "tokenization_id": tok_id, "mode": "lora_dpo",
                     "best_val_acc": best_acc, "dpo_beta": DPO_BETA,
                     "init_ckpt": SFT_CKPT})
    save_adapter(ADAPTER_PATH, model, optimizer, global_step, history, best_acc, DPO_EPOCHS)
    save_merged(LATEST_PATH, model,
                {"epoch": DPO_EPOCHS, "curr_step": global_step,
                 "tokenization_id": tok_id, "mode": "lora_dpo",
                 "best_val_acc": best_acc, "dpo_beta": DPO_BETA,
                 "init_ckpt": SFT_CKPT, "history": history})
    write_json(os.path.join(REPORT_DIR, "dpo_training_latest.json"), {
        "stage": "dpo", "beta": DPO_BETA, "lr": DPO_LR, "epochs": DPO_EPOCHS,
        "steps": global_step, "best_val_acc": best_acc, "history": history,
        "pre_dpo_lm_drift": drift0, "stop_reason": stop_reason,
        "wall_hours": round((time.time() - t_start) / 3600, 2)})
    write_json(STATUS_PATH, {"stage": "dpo_done", "step": global_step,
                             "best_val_acc": best_acc,
                             "wall_hours": round((time.time() - t_start) / 3600, 2)})
    print(f"\nDPO complete in {(time.time()-t_start)/3600:.2f}h | best val_acc={best_acc:.3f}")


if __name__ == "__main__":
    main()
