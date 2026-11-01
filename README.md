# synapsenew

An improved SFT + RL (DPO) pipeline for the **Synapse** 2.1B-parameter LLM —
retrained locally on a 16 GB laptop GPU, starting from the Synapse pretrain
checkpoint on Google Drive.

> **This repo is the code.** Data, token shards, and checkpoints live on the
> laptop workspace `/media/alfonso/shared/synapsenew` (env `SYNAPSENEW_DIR`)
> and on Google Drive (`gdrive:synapse/…`, pulled with rclone — see
> `scripts/pull_from_drive.sh`).

## What this is

The original Synapse runs did pretraining on cloud GPUs and a full-parameter
SFT on an A100-class machine + Colab. This repo re-does the *alignment* stages
locally, with two changes that make a 2.1B model trainable on a 16 GB RTX 5080
*laptop*:

1. **LoRA fine-tuning** (r=64 on all attention + SwiGLU projections instead of
   full fine-tuning). Beyond the memory win, this avoids a silent trap of
   bf16 laptop training: at SFT learning rates (~1e-5) bf16 parameter updates
   round away (ULP ≈ 4e-5 at weight scale 0.01) — an fp32 full fine-tune fits
   neither memory nor the night. Frozen bf16 base + fp32 adapters sidestep both.
2. **RL stage (DPO)** — the original Synapse pipeline stopped at SFT. Preference
   pairs are built from the same Drive SFT data: *chosen* = the curated
   assistant answer, *rejected* = the pretrain base model's own continuation,
   a corrupted/truncated answer, or (for short deterministic answers) another
   answer sampled from the same source. DPO then sharpens the model toward the
   curated behaviour using the SFT model itself as the frozen reference
   (`adapters off` ⇒ exactly the start-of-DPO policy, zero extra weights).

Everything else deliberately mirrors the reference recipe so numbers stay
comparable: same 15-source v3 data mix, same ChatML tokenization
(`tokenization_id 7a570a7ba9fc7985`), same token-weighted masked cross-entropy,
same token-normalized gradient accumulation, same best-overall-val model
selection.

## Pipeline

```
gdrive:synapse/tokenizer_out, manifests, sft_tokenized      (rclone pull)
        │
        ├─ baseline: per-source val loss + deterministic chat bench (pretrain)
        │
        ├─ DPO pair building: chosen (curated) vs rejected (base-gen/corrupt/swap)
        │
        ├─ SFT:  LoRA r=64, budgeted cosine (~1200 steps), grad-checkpointed bf16
        │        base + fused AdamW on adapters, merged-checkpoint saves
        │
        ├─ SFT evals: full val loss + bench + samples
        │
        ├─ DPO:  policy = SFT + fresh LoRA, ref = adapters-off (frozen start)
        │        monitors: val preference accuracy, reward margin, LM-drift guard
        │
        └─ final evals + reports/MORNING_REPORT.md + rclone push to
           gdrive:synapsenew/{checkpoints,reports}
```

## Layout

| Path | What |
|---|---|
| `synapse_model.py` | SynapseGPT training model (exact pretrain architecture) |
| `infer_model.py` | KV-cache inference model + checkpoint loader (from the synapse repo) |
| `chat_template.py`, `tools_runtime.py` | ChatML template + tool serialization (must match tokenization) |
| `sft_bench.py`, `bench/sft_bench.jsonl` | 49-item deterministic chat bench (offline graders) |
| `lora.py` | LoRA layers, injection, merge-on-save |
| `data_utils.py` | 15-source mix loading, collate, masked loss, per-source eval |
| `train_sft.py` | SFT trainer (laptop-sized recipe) |
| `build_dpo_data.py` | DPO preference pair construction (CPU + optional base-model gen) |
| `train_dpo.py` | DPO trainer |
| `eval_model.py` | `loss` / `samples` / `bench` evals for any checkpoint |
| `make_report.py` | aggregates everything into `reports/MORNING_REPORT.md` |
| `scripts/pull_from_drive.sh` | rclone pull of all Drive assets |
| `scripts/run_pipeline.sh` | full overnight pipeline (nohup-able) |
| `scripts/push_results.sh` | push checkpoints + reports back to Drive |

## Quickstart (new machine)

```bash
# 1. data + checkpoints from Drive (rclone remote "gdrive" configured)
SYNAPSENEW_DIR=/media/alfonso/shared/synapsenew bash scripts/pull_from_drive.sh

# 2. environment (torch cu130 + tokenizers; see requirements.txt)
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 3. full pipeline (baseline → SFT → DPO → evals → report)
SYNAPSENEW_DIR=/media/alfonso/shared/synapsenew bash scripts/run_pipeline.sh
```

Single stages:

```bash
python train_sft.py                  # knobs: SFT_STEPS, SFT_MAX_LR, SFT_BATCH, SFT_ACCUM, ...
python build_dpo_data.py --gen-ckpt $PRETRAIN_CKPT   # phase 2 (base-gen rejects)
python train_dpo.py                  # knobs: DPO_EPOCHS, DPO_BETA, DPO_LR
python eval_model.py loss --ckpt checkpoints/dpo_best.pth --tag dpo_best
python eval_model.py bench --ckpt checkpoints/dpo_best.pth --tag dpo_best
SYNAPSENEW_DIR=... bash scripts/push_results.sh
```

## Hardware / timing notes

- Target: 16 GB GPU (tested RTX 5080 laptop, torch 2.14+cu130, sm_120).
- SFT budget is adaptive: `SFT_DEADLINE_TS` stops cleanly; `SFT_STEPS` sizes the
  cosine horizon. The reference v3 eval history showed ~1200 of its 2938 steps
  captures the whole validation gain, so a single night ≈ a full reference run.
- Full merged checkpoints (`sft_best.pth`, `dpo_best.pth`, ~8.4 GB fp32) load
  directly into the existing `SynapseInfer` / chatbot / bench tooling — no
  adapter plumbing needed downstream. `*_adapter.pth` files keep optimizer
  state + unmerged adapters for exact continuation.

## Results (overnight run, 2026-10-30 → 31, + big-LoRA follow-up)

Run on the laptop's RTX 5080 (16 GB), everything from the Drive pretrain
checkpoint. SFT r=64 = 1200 steps / 6.1h; DPO = 2 epochs / 1.6h;
big-LoRA r=256 = 2750 steps / 9.6h (micro-batch 1x64).

| metric | Pretrain | SFT r=64 | SFT+DPO | **big-LoRA r=256** | Reference v3 (A100) |
|---|---|---|---|---|---|
| Val loss (15-source) | 2.746 | 1.248 | 1.315² | **1.198** | **1.150** |
| Chat bench (49 items) | 32% | 70% | **75%** | 70% | 70% |
| GSM8K (100q, 4-shot) | — | 11.0% | 11.0% | **12.0%** | 10.0% |
| ARC-Challenge (200q) | — | 26.0% | 28.0% | **30.5%** | **30.5%** |
| DPO val preference acc | — | — | 0.669 | — | — |

¹ big-LoRA is the best all-rounder built locally: closest val loss to v3,
best GSM8K, tied ARC-C, tied bench. v3 keeps the val-loss crown (full FT
capacity); DPO keeps the bench crown (preference sharpening).
² DPO optimizes preference margins, not LM loss: bench +5 pts, drift guard
barely moved; val loss rises modestly on format/NLI sources.

Full narratives: `reports/MORNING_REPORT.md`, `reports/samples_*.md`,
`reports/bench_*/`, `reports/standard_eval_*.json`;
checkpoints synced to `gdrive:synapsenew/checkpoints/`.

**Interactive validation dashboard:** `reports/dashboard.html` (17 curated
deterministic-graded tests × all checkpoints incl. big-LoRA and the soup,
side by side with pass/fail badges; generated by `scripts/build_dashboard.py`,
also on Drive).
