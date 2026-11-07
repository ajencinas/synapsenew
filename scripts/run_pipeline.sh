#!/usr/bin/env bash
# synapsenew overnight pipeline: baseline evals → DPO data → SFT → evals →
# DPO (RL) → evals → report. Designed to be launched with nohup; every stage
# logs its own file under logs/ and the orchestrator continues on soft failures
# so the morning report still gets built.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${SYNAPSENEW_VENV:-$REPO/.venv}
export SYNAPSENEW_DIR=${SYNAPSENEW_DIR:-$HOME/synapsenew}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR="$SYNAPSENEW_DIR/logs"

cd "$REPO"
if [ -x "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
mkdir -p "$LOGDIR" "$SYNAPSENEW_DIR"/{checkpoints,reports}

# tuned knobs (overridden by config.env if present)
SFT_STEPS=${SFT_STEPS:-1200}
SFT_BATCH=${SFT_BATCH:-4}
SFT_ACCUM=${SFT_ACCUM:-16}
SFT_COMPILE=${SFT_COMPILE:-1}
DPO_EPOCHS=${DPO_EPOCHS:-3}
SFT_DEADLINE_TS=${SFT_DEADLINE_TS:-$(date -d 'tomorrow 04:30' +%s)}
DPO_DEADLINE_TS=${DPO_DEADLINE_TS:-$(date -d 'tomorrow 06:45' +%s)}
export SFT_STEPS SFT_BATCH SFT_ACCUM SFT_COMPILE DPO_EPOCHS SFT_DEADLINE_TS DPO_DEADLINE_TS

CK=$SYNAPSENEW_DIR/checkpoints
PRETRAIN=$CK/pretrain_v2_full.pth
REF_V3=${REF_V3_SFT_BEST:-$CK/ref_v3_sft_best.pth}

stage() {  # stage <name> <cmd...>
  local name=$1; shift
  echo "[pipeline] START $name at $(date '+%F %T')"
  "$@" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo "[pipeline] END   $name rc=$rc at $(date '+%F %T')"
  return $rc
}

echo "[pipeline] SFT: steps=$SFT_STEPS batch=$SFT_BATCH accum=$SFT_ACCUM compile=$SFT_COMPILE"
echo "[pipeline] deadlines: SFT=$(date -d @$SFT_DEADLINE_TS '+%F %T') DPO=$(date -d @$DPO_DEADLINE_TS '+%F %T')"

# ---------- Stage 0: baseline (pretrain) evals ----------
stage 10_eval_pretrain python eval_model.py loss --ckpt "$PRETRAIN" --tag pretrain
stage 11_samples_pretrain python eval_model.py samples --ckpt "$PRETRAIN" --tag pretrain --max-tokens 160 || true
stage 12_bench_pretrain python eval_model.py bench --ckpt "$PRETRAIN" --tag pretrain || true

# ---------- Stage 1: DPO preference data ----------
stage 15_dpo_data_phase1 python build_dpo_data.py --out-dir "$SYNAPSENEW_DIR/data/dpo" \
  --n-train 6000 --n-val 300
stage 16_dpo_data_phase2 python build_dpo_data.py --gen-ckpt "$PRETRAIN" --max-new 96 || true

# ---------- Stage 2: SFT (LoRA, budgeted) ----------
stage 20_train_sft python train_sft.py

# ---------- Stage 3: post-SFT evals ----------
SFT_CKPT=$CK/sft_best.pth
[ -f "$SFT_CKPT" ] || SFT_CKPT=$CK/sft_latest.pth
stage 30_eval_sft python eval_model.py loss --ckpt "$SFT_CKPT" --tag sft_best
stage 31_samples_sft python eval_model.py samples --ckpt "$SFT_CKPT" --tag sft_best --max-tokens 200 || true
stage 32_bench_sft python eval_model.py bench --ckpt "$SFT_CKPT" --tag sft_best || true

# ---------- Stage 4: DPO (RL) ----------
export DPO_INIT_CKPT=$SFT_CKPT
stage 40_train_dpo python train_dpo.py

# ---------- Stage 5: post-DPO evals + reference ----------
DPO_CKPT=$CK/dpo_best.pth
[ -f "$DPO_CKPT" ] || DPO_CKPT=$CK/dpo_latest.pth
[ -f "$DPO_CKPT" ] && stage 50_eval_dpo python eval_model.py loss --ckpt "$DPO_CKPT" --tag dpo_best
[ -f "$DPO_CKPT" ] && stage 51_samples_dpo python eval_model.py samples --ckpt "$DPO_CKPT" --tag dpo_best --max-tokens 200 || true
[ -f "$DPO_CKPT" ] && stage 52_bench_dpo python eval_model.py bench --ckpt "$DPO_CKPT" --tag dpo_best || true
if [ -f "$REF_V3" ]; then
  stage 55_eval_v3ref python eval_model.py loss --ckpt "$REF_V3" --tag v3ref
  stage 56_bench_v3ref python eval_model.py bench --ckpt "$REF_V3" --tag v3ref || true
fi

# ---------- Stage 6: report + push ----------
stage 60_report python make_report.py
stage 61_push bash scripts/push_results.sh || true

echo "[pipeline] ALL DONE at $(date '+%F %T')"
