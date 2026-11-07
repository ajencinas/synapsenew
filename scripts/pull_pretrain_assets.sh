#!/usr/bin/env bash
# Pull the pretraining corpus shards and tokenizer metadata from Drive.
# This is intentionally separate from scripts/pull_from_drive.sh because the
# merged shards are ~329 GB.
set -eu
WS=${SYNAPSENEW_DIR:-/media/alfonso/shared/synapsenew}
DATA="$WS/data"
mkdir -p "$DATA"/{token_shards_merged,tokenizer_out,manifests}

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "tokenizer metadata…"
rclone copy gdrive:synapse/tokenizer_out "$DATA/tokenizer_out/" \
  --include "tokenizer.json" --include "tokenizer_config.json"
rclone copy gdrive:synapse/manifests "$DATA/manifests/" --transfers 4

log "merged token shards (~329 GB) → $DATA/token_shards_merged"
rclone copy gdrive:synapse/token_shards_merged "$DATA/token_shards_merged/" \
  --transfers 8 --checkers 16 --drive-chunk-size 128M --checksum -P

log "linking legacy and new workspace layouts…"
bash "$(dirname "$0")/link_workspace_layout.sh"

log "done. du:"
du -sh "$DATA/token_shards_merged" "$DATA/tokenizer_out" "$DATA/manifests"
