#!/usr/bin/env python3
"""Shared constants, paths, and helpers for synapsenew.

Layout assumptions (all overridable by env):
  SYNAPSENEW_DIR  default ~/synapsenew
    data/         tokenizer_out/, manifests/, sft_tokenized/, dpo/
    checkpoints/  pretrain + trained checkpoints
    logs/         run logs
    reports/      eval JSONs + markdown reports

Data comes from Google Drive (rclone remote "gdrive", folder synapse/) —
see scripts/pull_from_drive.sh. The public GitHub repo contains code only;
corpora and checkpoints are external.
"""
import hashlib
import json
import os
import sys

import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
WS = os.environ.get("SYNAPSENEW_DIR") or os.path.expanduser("~/synapsenew")
DATA_DIR = os.path.join(WS, "data")
CKPT_DIR = os.path.join(WS, "checkpoints")
LOG_DIR = os.path.join(WS, "logs")
REPORT_DIR = os.path.join(WS, "reports")

TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer_out", "tokenizer.json")
SFT_TOKENIZED_DIR = os.path.join(DATA_DIR, "sft_tokenized")
MANIFEST_DIR = os.path.join(DATA_DIR, "manifests")
DPO_DIR = os.path.join(DATA_DIR, "dpo")

PRETRAIN_CKPT = os.environ.get("PRETRAIN_CKPT",
                               os.path.join(CKPT_DIR, "pretrain_v2_full.pth"))

# Reference checkpoint (previous full-FT run, v3 15-source). Supply
# REF_V3_SFT_BEST or place ref_v3_sft_best.pth in the checkpoints directory.
REF_V3_SFT_BEST = os.environ.get(
    "REF_V3_SFT_BEST", os.path.join(CKPT_DIR, "ref_v3_sft_best.pth"))

for d in (DATA_DIR, CKPT_DIR, LOG_DIR, REPORT_DIR, DPO_DIR):
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------
# Tokenization constants (must match the pretrain + SFT tokenization)
# --------------------------------------------------------------------------
EXPECTED_TOK_ID = "7a570a7ba9fc7985"
IGNORE_INDEX = -100
PAD_ID = 0          # <||endoftext|>; right-padding + IGNORE labels make it inert
EOT_ID = 0

BLOCK_SIZE = 2048

# Reserved role tokens (ids asserted at runtime against the tokenizer)
ROLE_SPECIAL = {"system": "<|reserved_0|>", "user": "<|reserved_1|>",
                "assistant": "<|reserved_2|>"}
EXPECTED_ROLE_ID = {"system": 9, "user": 10, "assistant": 11}
IM_END = "<|im_end|>"


def load_tokenizer(path=TOKENIZER_PATH):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def tokenizer_fingerprint(path=TOKENIZER_PATH):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def resolve_special_ids(tok):
    """(role_id: dict, im_end_id, eot_id) with a loud guard on reserved layout."""
    role_id = {}
    for role, name in ROLE_SPECIAL.items():
        i = tok.token_to_id(name)
        if i is None or i != EXPECTED_ROLE_ID[role]:
            raise SystemExit(f"tokenizer reserved layout changed: {name} -> {i} "
                             f"(expected {EXPECTED_ROLE_ID[role]})")
        role_id[role] = i
    im_end_id = tok.token_to_id(IM_END)
    eot_id = tok.token_to_id("<|endoftext|>")
    if im_end_id is None or eot_id is None:
        raise SystemExit("tokenizer missing <im_end> or <endoftext> special tokens")
    return role_id, im_end_id, eot_id


def guard_tokenization(tok_path=TOKENIZER_PATH):
    """Fail loud unless tokenizer.json fingerprint == pretrain == SFT registry."""
    tok_id = tokenizer_fingerprint(tok_path)
    pretrain = json.load(open(os.path.join(MANIFEST_DIR, "training_latest.json")))
    sft = json.load(open(os.path.join(MANIFEST_DIR, "sft_data_registry.json")))
    ids = {"tokenizer.json": tok_id,
           "pretrain manifest": pretrain["tokenization_id"],
           "sft registry": sft["tokenization_id"]}
    bad = {k: v for k, v in ids.items() if v != EXPECTED_TOK_ID}
    if bad:
        raise SystemExit(f"tokenization_id mismatch — refusing to continue: {ids}")
    return tok_id


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def set_repro(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_state_dict_from_checkpoint(path, mmap=True):
    """Load {'schema':'v2', 'model': ...} or a raw state_dict; strip compile prefix."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except (TypeError, RuntimeError):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model"}
    else:
        sd, meta = ckpt, {}
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd, meta
