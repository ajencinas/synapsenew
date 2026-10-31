#!/usr/bin/env bash
# Re-run just the DPO stage + post-DPO evals + report + push (after the
# collate-mask fix). Launched 2026-10-31 04:5x.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=/home/alfonso/Python/synapse/synapse
export SYNAPSENEW_DIR=${SYNAPSENEW_DIR:-/media/alfonso/shared/synapsenew}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR="$SYNAPSENEW_DIR/logs"

cd "$REPO"
source "$VENV/bin/activate"
mkdir -p "$LOGDIR"

export DPO_EPOCHS=${DPO_EPOCHS:-2}
export DPO_DEADLINE_TS=${DPO_DEADLINE_TS:-$(date -d 'today 10:00' +%s)}
export DPO_INIT_CKPT=$SYNAPSENEW_DIR/checkpoints/sft_best.pth

CK=$SYNAPSENEW_DIR/checkpoints

stage() {
  local name=$1; shift
  echo "[dpo-run] START $name at $(date '+%F %T')"
  "$@" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo "[dpo-run] END   $name rc=$rc at $(date '+%F %T')"
  return $rc
}

echo "[dpo-run] DPO epochs=$DPO_EPOCHS init=$DPO_INIT_CKPT"
stage 40b_train_dpo python train_dpo.py
DPO_CKPT=$CK/dpo_best.pth
[ -f "$DPO_CKPT" ] || DPO_CKPT=$CK/dpo_latest.pth
[ -f "$DPO_CKPT" ] && stage 50b_eval_dpo python eval_model.py loss --ckpt "$DPO_CKPT" --tag dpo_best
[ -f "$DPO_CKPT" ] && { stage 51b_samples_dpo python eval_model.py samples --ckpt "$DPO_CKPT" --tag dpo_best --max-tokens 200 || true; }
[ -f "$DPO_CKPT" ] && { stage 52b_bench_dpo python eval_model.py bench --ckpt "$DPO_CKPT" --tag dpo_best || true; }
stage 60b_report python make_report.py
stage 61b_push bash scripts/push_results.sh || true
echo "[dpo-run] ALL DONE at $(date '+%F %T')"
