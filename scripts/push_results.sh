#!/usr/bin/env bash
# Push checkpoints + reports back to Google Drive (gdrive:synapsenew/).
set -u
WS=${SYNAPSENEW_DIR:-$HOME/synapsenew}

echo "[push] checkpoints → gdrive:synapsenew/checkpoints/"
# best + adapter files (small); skip the ~8 GB *_latest.pth mid-run snapshots
rclone copy "$WS/checkpoints" gdrive:synapsenew/checkpoints/ \
  --include "sft_best.pth" --include "sft_adapter.pth" \
  --include "sft_best_biglora.pth" --include "sft_adapter_biglora.pth" \
  --include "dpo_best.pth" --include "dpo_adapter.pth" \
  --checksum --drive-chunk-size 128M --transfers 2

echo "[push] reports → gdrive:synapsenew/reports/"
rclone copy "$WS/reports" gdrive:synapsenew/reports/ --checksum --transfers 8

echo "[push] done at $(date '+%F %T')"
