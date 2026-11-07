# Results and evaluation protocol

These numbers come from the `synapsenew` laptop pipeline and the reference v3
full fine-tune. They were measured locally on the same evaluation data and the
same deterministic graders.

## Protocol

- **Validation loss** — token-weighted response-token cross-entropy over the
  same 15 SFT validation sources used by v1–v3. Lower is better.
- **Chat bench** — 49-item deterministic suite in `bench/sft_bench.jsonl`;
  40 items are automatically gradable without an LLM judge. Pass rate is over
  graded items.
- **GSM8K** — 100 test questions, 4-shot chain-of-thought prompting, greedy
  decoding, final numeric answer extraction.
- **ARC-Challenge** — 200 test questions, answer choice scored by mean token
  log-probability; accuracy over 4-way multiple choice (random = 25%).
- **DPO preference accuracy** — fraction of held-out preference pairs where the
  policy assigns a higher implicit reward to the curated answer than to the
  negative.

## Results

| checkpoint | val loss | chat bench | GSM8K | ARC-C |
|---|---:|---:|---:|---:|
| Pretrain base | 2.746 | 32% | — | — |
| synapsenew SFT r=64 | **1.248** | **70%** | 11.0% | 26.0% |
| synapsenew SFT + DPO | 1.315 | **75%** | 11.0% | 28.0% |
| synapsenew big-LoRA r=256 | **1.198** | 70% | **12.0%** | **30.5%** |
| Reference v3 full-FT (A100) | **1.150** | 70% | 10.0% | **30.5%** |

- SFT improved **15/15** validation sources over the pretrain base.
- DPO reached **0.669** best held-out preference accuracy. Its LM loss rises
  slightly because DPO optimizes preference margins, not token likelihood.
- big-LoRA is the best all-rounder built on the 16 GB laptop: closest val loss
  to the full fine-tune and best GSM8K within the local runs.
- The reference full fine-tune remains ahead on raw validation loss, but ties
  the laptop runs on ARC-C and the chat bench.

## Reproduce

```bash
export SYNAPSENEW_DIR="$HOME/synapsenew"
python eval_model.py loss  --ckpt "$SYNAPSENEW_DIR/checkpoints/sft_best.pth" --tag sft
python eval_model.py bench --ckpt "$SYNAPSENEW_DIR/checkpoints/sft_best.pth" --tag sft
python standard_eval.py    --ckpt "$SYNAPSENEW_DIR/checkpoints/sft_best.pth" --tag sft
python make_report.py
python scripts/build_dashboard.py
```

The full generated report is written to
`$SYNAPSENEW_DIR/reports/MORNING_REPORT.md`; the HTML dashboard is
`$SYNAPSENEW_DIR/reports/dashboard.html`. Generated reports and checkpoints are
external artifacts and are not committed.
