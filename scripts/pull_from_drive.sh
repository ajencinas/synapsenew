#!/usr/bin/env bash
# Pull everything synapsenew needs from Google Drive (rclone remote "gdrive",
# folder synapse/). Safe to re-run: rclone skips already-transferred files.
set -eu
WS=${SYNAPSENEW_DIR:-/media/alfonso/shared/synapsenew}
mkdir -p "$WS"/{data,checkpoints,logs,reports}

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "tokenizer + manifests…"
rclone copy gdrive:synapse/tokenizer_out/tokenizer.json "$WS/data/tokenizer_out/"
rclone copy gdrive:synapse/manifests "$WS/data/manifests/" --transfers 4

log "SFT tokenized data (v3, 15 sources, ~1 GB)…"
rclone copy gdrive:synapse/sft_tokenized "$WS/data/sft_tokenized/" \
  --transfers 16 --checkers 16 --checksum

log "pretrain checkpoint (25 GB, model+optimizer; loaded via mmap — the object
     is written as a single file so the ~8 GB model part can be read alone)…"
rclone copyto gdrive:synapse/checkpoints/synapse_2b_d2560_l28.pth \
  "$WS/checkpoints/pretrain_v2_full.pth.partial" \
  --drive-chunk-size 128M --multi-thread-streams 4 --multi-thread-cutoff 64M \
  && mv "$WS/checkpoints/pretrain_v2_full.pth.partial" \
        "$WS/checkpoints/pretrain_v2_full.pth"

log "reference v3 SFT best (8.4 GB, for head-to-head comparison)…"
rclone copyto gdrive:synapse/sft_checkpoints/v3_15source/sft_best.pth \
  "$WS/checkpoints/ref_v3_sft_best.pth" \
  --drive-chunk-size 128M --multi-thread-streams 4

log "reference v3 logs (MODEL_CARD + training json + bench)…"
rclone copy gdrive:synapse/sft_checkpoints/v3_15source "$WS/reports/ref_v3/" \
  --include "MODEL_CARD.md" --include "sft_training_latest.json" \
  --include "bench/**" --transfers 4 || true

log "done. du:"
du -sh "$WS"/data "$WS"/checkpoints
