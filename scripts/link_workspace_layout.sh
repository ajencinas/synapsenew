#!/usr/bin/env bash
# Bridge the legacy layout (assets directly under $SYNAPSE_DIR) and the
# synapsenew layout ($SYNAPSENEW_DIR/data/<asset>) with symlinks.
# Safe to re-run; only creates links when the target exists.
set -eu
WS=${SYNAPSENEW_DIR:-/media/alfonso/shared/synapsenew}
DATA="$WS/data"
mkdir -p "$DATA"

link() {
  local name=$1
  local target="$DATA/$name"
  local path="$WS/$name"
  if [ -e "$target" ] && [ ! -e "$path" ]; then
    ln -s "$target" "$path"
    echo "linked $path -> $target"
  fi
}

for name in tokenizer_out manifests sft_tokenized sft_tokenized_v1_7source \
            sft_tokenized_v2_12source token_shards token_shards_merged \
            datasets_pretrain datasets_sft dpo; do
  link "$name"
done
