#!/usr/bin/env bash
# Source this file to point every pipeline stage at the same workspace.
# Usage: source scripts/env.sh
export SYNAPSENEW_DIR="${SYNAPSENEW_DIR:-/media/alfonso/shared/synapsenew}"
export SYNAPSE_DIR="${SYNAPSE_DIR:-$SYNAPSENEW_DIR}"
export SYNAPSE_EVAL_DIR="${SYNAPSE_EVAL_DIR:-$SYNAPSENEW_DIR}"
echo "SYNAPSENEW_DIR=$SYNAPSENEW_DIR"
echo "SYNAPSE_DIR=$SYNAPSE_DIR"
