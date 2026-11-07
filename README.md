# SynapseGPT + synapsenew

**A complete 2.09B-parameter language-model repository**: raw-corpus download →
custom tokenizer → pretraining → supervised fine-tuning → preference
optimization (DPO) → evaluation → serving, plus the research artifacts and
video explainer used to document it.

This repository combines the original **synapse** pretraining stack with the
improved **synapsenew** post-training stack. The original repo remains intact;
this is the single full-pipeline repo going forward.

## What is here

| Stage | Path | Notes |
|---|---|---|
| Architecture | `synapse_model.py` | Single source of truth used by pretraining and reference SFT |
| Raw corpora | `download_pretrain_data/` | Public C4, Wikipedia, RedPajama, code, FineMath, FineWeb, reasoning notebooks |
| Tokenizer | `tokenize/` | Byte-level BPE, 64k vocab, uint16 shards, merge/eval/upload pipeline |
| Pretraining | `pretrain/` | `train.py`, Colab notebook, Lambda/RunPod bootstrap, checkpoint recovery |
| SFT — improved | `train_sft.py` | **Primary** laptop-friendly LoRA SFT, 15-source mix, merged checkpoints |
| SFT — reference | `sft/sft.py` | Original full-parameter SFT used for v1–v3 A100 runs |
| SFT data | `sft/` | Data builders, teacher generators, verification, ChatML tokenization, tool runtime |
| RL / DPO | `build_dpo_data.py`, `train_dpo.py` | Preference-pair construction + DPO, reference = adapters-off policy |
| Evaluation | `eval_model.py`, `standard_eval.py`, `sft_bench.py`, `make_report.py` | Per-source loss, GSM8K/ARC, deterministic chat bench, reports |
| Dashboard | `scripts/build_dashboard.py` | Static HTML side-by-side validation dashboard |
| Serving | `sparky/` | Flask chatbot, tool loop, inference model, bench, leaderboard, voice mode |
| Utilities | `lora.py`, `data_utils.py`, `chat_template.py`, `tools_runtime.py` | Shared training/inference primitives |
| Orchestration | `scripts/` | rclone pull, overnight SFT+DPO pipeline, checkpoint push, model soup |
| Tests | `tests/` | Data builders, tokenization, tool loop, checkpoint/training logic |
| Documentation | `video/synapsenew-explainer/` | HyperFrames + ElevenLabs architecture/SFT/DPO explainer |

## Model

| | |
|---|---|
| Architecture | decoder-only transformer, **2.09B** params: d=2560, 28 layers, 20 heads (GQA with 4 KV heads), SwiGLU, RoPE, RMSNorm, 2048-token context |
| Tokenizer | custom byte-level BPE, 64k vocab, digit-per-token, reserved tool tokens |
| Pretraining | ~44B tokens: web, code, math, arXiv, Wikipedia, books, verified synthetic data |
| Post-training | 15-source ChatML SFT, assistant-only loss mask; LoRA r=64/256 on 16 GB laptop; DPO on curated/verified preferences |
| Tools | model-emitted JSON calls executed by a shared sandboxed-python / search loop |

## Measured results

Reference v3 full fine-tune and the new laptop pipeline were evaluated on the
same 15-source validation sets and the same deterministic 49-item chat bench.

| checkpoint | val loss (15-source) | chat bench | GSM8K | ARC-C |
|---|---|---|---|---|
| Pretrain base | 2.746 | 32% | — | — |
| **synapsenew SFT r=64** | **1.248** | **70%** | 11.0% | 26.0% |
| synapsenew SFT + DPO | 1.315 | **75%** | 11.0% | 28.0% |
| **synapsenew big-LoRA r=256** | **1.198** | 70% | **12.0%** | **30.5%** |
| Reference v3 full-FT (A100) | **1.150** | 70% | 10.0% | **30.5%** |

DPO validation preference accuracy: **0.669**. SFT improved **15/15** validation
sources over the pretrain base. Full details: `reports/MORNING_REPORT.md`.

## Quickstart

### 1. Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional serving/eval extras:
pip install -r requirements-serving.txt
```

Create a gitignored `.env` only if you use data-generation teachers, the live
search tool, or the chatbot: `DEEPSEEK_API_KEY`, `BRAVE_API_KEY`,
`NGROK_AUTHTOKEN`, etc. Never commit it.

### 2. Data directory

All trainers share one data root. The new pipeline uses `SYNAPSENEW_DIR`; the
imported legacy scripts also accept `SYNAPSE_DIR`.

```bash
export SYNAPSENEW_DIR=/media/alfonso/shared/synapsenew
export SYNAPSE_DIR="$SYNAPSENEW_DIR"
```

Pull the SFT/checkpoint assets:

```bash
bash scripts/pull_from_drive.sh
```

Pull the pretraining shards (~329 GB) only when pretraining locally/on a VM:

```bash
bash scripts/pull_pretrain_assets.sh
```

Bridge the original and new workspace layouts (legacy stages expect assets
directly under `$SYNAPSE_DIR`; new stages use `$SYNAPSENEW_DIR/data`):

```bash
bash scripts/link_workspace_layout.sh
```

### 3. Pretrain

```bash
# bare GPU VM / laptop with enough VRAM
python pretrain/train.py

# or on Colab:
# pretrain/pre_train_mar23.ipynb
# or on Lambda/RunPod:
bash pretrain/run_on_vm.sh
```

Pretraining reads `$SYNAPSE_DIR/token_shards_merged`, `tokenizer_out`, and
`manifests`, and writes versioned `v2` checkpoints to `$SYNAPSE_DIR/checkpoints`.
The original pretraining notes and hardware guidance are in
`pretrain/README_LAMBDA.md`.

### 4. SFT — improved (recommended)

```bash
python train_sft.py
```

The default runs the 15-source mix with LoRA r=64, gradient checkpointing,
token-normalized accumulation, best-val tracking, and merged checkpoints. Key
environment variables: `SFT_STEPS`, `SFT_MAX_LR`, `SFT_BATCH`, `SFT_ACCUM`,
`SFT_LORA_R`, `SFT_DEADLINE_TS`, `SFT_TAG`, `SFT_COMPILE`.

For the original full-parameter recipe (A100-class):

```bash
CHECKPOINT_NAME=pretrain_v2_full.pth python sft/sft.py
# or: CHECKPOINT_NAME=pretrain_v2_full.pth bash sft/run_sft_on_vm.sh
```

The legacy trainer expects the old checkpoint filename; the new pipeline keeps
the same weights as `checkpoints/pretrain_v2_full.pth`.

### 5. RL — DPO

```bash
python build_dpo_data.py --out-dir "$SYNAPSENEW_DIR/data/dpo"
python train_dpo.py
```

`build_dpo_data.py` builds pairs from the SFT data: curated chosen answers vs
corrupted/base-model/swapped negatives. `train_dpo.py` uses the SFT checkpoint
as policy and the adapters-off model as the frozen reference.

### 6. Evaluate

```bash
python eval_model.py loss  --ckpt "$SYNAPSENEW_DIR/checkpoints/sft_best.pth" --tag sft
python eval_model.py bench --ckpt "$SYNAPSENEW_DIR/checkpoints/sft_best.pth" --tag sft
python standard_eval.py    --ckpt "$SYNAPSENEW_DIR/checkpoints/sft_best.pth" --tag sft
python make_report.py
python scripts/build_dashboard.py
```

The reference-era harnesses are under `sparky/`:
`sft_bench.py`, `eval_tool_use.py`, `sparky_leaderboard.py`, and the bulk
checkpoint dashboard. See `sparky/README.md`.

### 7. Serve

```bash
python sparky/sparky_chatbot.py --help
```

The Flask chatbot exposes streaming chat, optional tool use (sandboxed python +
Brave search), checkpoint switching, and hands-free voice mode. The Colab entry
point is `sparky/sparky_chatbot_colab.ipynb`.

## Google Drive layout

The full workspace lives under `gdrive:synapse/` (about 1.9 TB):

| Path | Contents |
|---|---|
| `datasets_pretrain/` | 23,064 files, ~577 GB raw pretraining corpora |
| `datasets_sft/` | 38 files, ~256 MB raw SFT sources |
| `tokenizer_out/` | tokenizer.json + eval samples, ~1.9 GB |
| `token_shards/` | 1,621 files, ~329 GB per-source token shards |
| `token_shards_merged/` | 2,771 files, ~329 GB merged uint16 shards |
| `sft_tokenized/` and `sft_tokenized_v{1,2}_*/` | 15-source v3 data + historical tokenizations, ~2.1 GB |
| `checkpoints/` | 23 pretraining/versioned checkpoints, ~513 GB |
| `sft_checkpoints/` | 26 SFT/DPO checkpoints and archives, ~134 GB |
| `manifests/` | tokenization IDs, data registries, training manifests |

`gdrive:synapsenew/` holds the new pipeline's artifacts (~32.5 GB):
`checkpoints/` (SFT, DPO, big-LoRA, adapters), `reports/` (evals, dashboard,
samples, model cards), and `video/` (the explainer MP4).

The separately named `synapse (Copy 2)` Drive folder is a historical duplicate,
not part of the live pipeline.

## Design principles

- **One model definition, structurally shared.** Pretraining and reference SFT
  import the same root `synapse_model.py`.
- **One data contract.** Tokenization IDs are guarded at every trainer; SFT
  chat wire format, tool serialization, and serving use the same code.
- **Verified data only.** Tool traces and math reasoning are kept only after
  independent verification; failed calls are not trained on.
- **Leak-free splits.** Train/validation is a global hash of the normalized
  question, so a question stays on one side across all sources and reruns.
- **Fail loud.** Tokenizer mismatch, missing keys, over-repeated sources, and
  stale data stop the run instead of training silently wrong.
- **Artifacts survive runs.** Checkpoints, manifests, eval results, and model
  cards are versioned on Google Drive.

## Not in Git

Secrets (`.env`), local virtualenvs, generated corpora/token shards, and model
checkpoints stay out of the repository. The excluded private data-generation
pipelines (`download_pretrain_books/`, `download_pretrain_others/`,
`common_pretrain_text_processing/`) are noted in `sft/README.md`; the public
pipeline in this repo is runnable without them except for those optional
teachers.

## Credits

Original Synapse pretraining, tokenizer, SFT v1–v3, tooling, and chatbot by
Alfonso Encinas. The `synapsenew` stage adds the laptop LoRA recipe, DPO stage,
standardized evaluations, dashboards, and explainer video.
