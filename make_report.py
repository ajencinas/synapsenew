#!/usr/bin/env python3
"""Aggregate all eval artifacts into reports/MORNING_REPORT.md."""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPORT_DIR

TAGS = ["pretrain", "sft_best", "dpo_best", "v3ref", "biglora", "soup_a50"]
LABELS = {"pretrain": "Pretrain base", "sft_best": "synapsenew SFT (r=64)",
          "dpo_best": "synapsenew SFT+DPO", "v3ref": "Reference v3 full-FT SFT",
          "biglora": "big-LoRA SFT (r=256)", "soup_a50": "soup v3+SFT"}


def jload(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def bench_summary(tag):
    """(overall_rate, per_cat, n_graded) parsed from bench_<tag>/step*.json."""
    files = sorted(glob.glob(os.path.join(REPORT_DIR, f"bench_{tag}", "*.json")))
    files = [f for f in files if not f.endswith(("bench_summary.json",))]
    if not files:
        return None
    d = jload(files[-1])
    if not d or "results" not in d:
        return None
    recs = d["results"].get("SFT", [])
    by_cat, overall = {}, [0, 0, 0]
    for r in recs:
        cat = r.get("category", "uncat")
        c = by_cat.setdefault(cat, [0, 0, 0])
        if r.get("verdict") is None:
            c[2] += 1; overall[2] += 1
        else:
            c[1] += 1; overall[1] += 1
            if r["verdict"]:
                c[0] += 1; overall[0] += 1
    rate = (overall[0] / overall[1]) if overall[1] else None
    return {"overall_rate": rate, "overall": overall, "by_cat": by_cat}


def main():
    L = []
    A = L.append
    now = time.strftime("%Y-%m-%d %H:%M")
    A(f"# synapsenew — overnight run report\n\n_generated {now}_\n")
    A("""> Pipeline: pull assets from Google Drive (rclone) → baseline evals →
> build DPO preference pairs from the SFT data → LoRA SFT on the pretrain
> checkpoint (16 GB laptop GPU) → DPO (RL) on top → full before/after evals.
> Val loss = token-weighted response-token CE on the 15 SFT val sets
> (comparable across all checkpoints, incl. the reference v1–v3 runs).
> Bench = 49-item deterministic suite (`sft_bench.jsonl`, no LLM judge).\n""")

    # ---- val loss table ----
    loss = {t: jload(os.path.join(REPORT_DIR, f"eval_loss_{t}.json")) for t in TAGS}
    if any(loss.values()):
        A("\n## 1. Per-source validation loss (lower = better)\n")
        srcs = sorted(loss["pretrain"]["per_source"]) if loss.get("pretrain") else \
            sorted(next(v["per_source"] for v in loss.values() if v)["per_source"])
        head = "| source | " + " | ".join(LABELS[t] for t in TAGS) + " |"
        A(head)
        A("|---|" + "---|" * len(TAGS))
        for s in srcs:
            cells = []
            for t in TAGS:
                v = loss.get(t)
                cells.append(f"{v['per_source'].get(s, float('nan')):.3f}" if v else "—")
            A(f"| {s} | " + " | ".join(cells) + " |")
        cells = []
        for t in TAGS:
            v = loss.get(t)
            cells.append(f"**{v['overall']:.3f}**" if v else "—")
        A("| **overall** | " + " | ".join(cells) + " |")
        if loss.get("pretrain") and loss.get("sft_best"):
            n = sum(1 for s in srcs
                    if loss["sft_best"]["per_source"].get(s, 9e9) <
                       loss["pretrain"]["per_source"].get(s, -9e9))
            A(f"\nSFT improves **{n}/{len(srcs)}** sources vs the pretrain base "
              f"(overall {loss['pretrain']['overall']:.3f} → "
              f"{loss['sft_best']['overall']:.3f}).")
        if loss.get("dpo_best") and loss.get("sft_best"):
            A(f"DPO keeps overall val at {loss['dpo_best']['overall']:.3f} "
              f"(vs SFT {loss['sft_best']['overall']:.3f}) — small drift is expected; "
              f"the RL objective is preference accuracy, not LM loss.")

    # ---- bench table ----
    bench = {t: bench_summary(t) for t in TAGS}
    if any(bench.values()):
        A("\n## 2. Deterministic chat bench (49 items, higher = better)\n")
        A("| checkpoint | overall pass | graded | categories |")
        A("|---|---|---|---|")
        for t in TAGS:
            b = bench.get(t)
            if not b:
                continue
            rate = f"{100*b['overall_rate']:.0f}%" if b["overall_rate"] is not None else "n/a"
            cats = " ".join(f"{c}:{100*p/t:.0f}% " if t else f"{c}:n/a "
                            for c, (p, t, u) in sorted(b["by_cat"].items()))
            A(f"| {LABELS[t]} | **{rate}** | {b['overall'][1]} | {cats.strip()} |")
        if bench.get("pretrain") and bench.get("sft_best") and \
           bench["pretrain"]["overall_rate"] is not None:
            A(f"\nSFT lift: {100*bench['pretrain']['overall_rate']:.0f}% → "
              f"{100*bench['sft_best']['overall_rate']:.0f}%.")
            if bench.get("dpo_best") and bench["dpo_best"]["overall_rate"] is not None:
                A(f"After DPO: {100*bench['dpo_best']['overall_rate']:.0f}%.")

    # ---- SFT training ----
    sft = jload(os.path.join(REPORT_DIR, "sft_training_latest.json"))
    sft_big = jload(os.path.join(REPORT_DIR, "sft_training_latest_biglora.json"))
    if sft_big:
        A("\n## 3b. big-LoRA SFT run (r=256, max local)\n")
        h = sft_big.get("hyperparameters", {})
        A(f"- {sft_big['curr_step']}/{sft_big['total_steps']} steps, "
          f"micro {h.get('micro_batch')}x{h.get('grad_accum')}, LoRA r={h.get('lora_r')} "
          f"(302.8M trainable), lr {h.get('max_lr')}")
        A(f"- wall: {sft_big.get('wall_hours')}h; best overall val "
          f"{sft_big.get('best_overall_val', 0):.4f}")
        hist = sft_big.get("eval_history", [])
        if hist:
            A("\n| step | overall | tulu3 | metamath | format_following | creative |\n|---|---|---|---|---|---|")
            for e in hist:
                A(f"| {e.get('step')} | {e.get('overall', float('nan')):.4f} | "
                  f"{e.get('tulu3',0):.3f} | {e.get('metamath',0):.3f} | "
                  f"{e.get('format_following',0):.3f} | {e.get('creative',0):.3f} |")
    if sft:
        A("\n## 3. SFT run\n")
        h = sft.get("hyperparameters", {})
        A(f"- budget: {sft['curr_step']}/{sft['total_steps']} steps "
          f"({sft.get('stopped_early','?')}), micro {h.get('micro_batch')}x{h.get('grad_accum')}, "
          f"LoRA r={h.get('lora_r')}, lr {h.get('max_lr')}")
        A(f"- wall: {sft.get('wall_hours')}h; best overall val {sft.get('best_overall_val', 0):.4f}")
        hist = sft.get("eval_history", [])
        if hist:
            A("\n| step | overall | tulu3 | no_robots | creative | metamath | nli |\n|---|---|---|---|---|---|---|")
            for e in hist:
                A(f"| {e.get('step')} | {e.get('overall', float('nan')):.4f} | "
                  f"{e.get('tulu3',0):.3f} | {e.get('no_robots',0):.3f} | "
                  f"{e.get('creative',0):.3f} | {e.get('metamath',0):.3f} | {e.get('nli',0):.3f} |")

    # ---- standardized evals (GSM8K / ARC) ----
    std = {t: jload(os.path.join(REPORT_DIR, f"standard_eval_{t}.json"))
           for t in TAGS}
    if any(std.values()):
        A("\n## 4b. Standardized benchmarks (no judge)\n")
        A("| checkpoint | GSM8K (4-shot CoT) | ARC-Challenge (loglikelihood) |")
        A("|---|---|---|")
        for t in TAGS:
            s = std.get(t)
            if not s:
                continue
            g = s.get("gsm8k", {})
            a = s.get("arc_challenge", {})
            A(f"| {LABELS[t]} | {g.get('acc',0)*100:.1f}% ({g.get('correct',0)}/{g.get('n',0)}) | "
              f"{a.get('acc',0)*100:.1f}% ({a.get('correct',0)}/{a.get('n',0)}) |")
        A("\nARC-C random baseline = 25% (4 choices).")

    # ---- DPO training ----
    dpo = jload(os.path.join(REPORT_DIR, "dpo_training_latest.json"))
    stats = jload(os.path.join(REPORT_DIR, "dpo_data_stats.json"))
    if dpo:
        A("\n## 4. DPO (RL) run\n")
        A(f"- beta={dpo.get('beta')}, lr={dpo.get('lr')}, epochs={dpo.get('epochs')}, "
          f"steps={dpo.get('steps')}, stop: {dpo.get('stop_reason')}, wall {dpo.get('wall_hours')}h")
        if stats:
            p2 = stats.get("phase2", {})
            A(f"- pairs: {p2.get('gen_rejects','?')} base-model rejects of "
              f"{p2.get('total','?')} pairs (rest: corrupted/swap negatives)")
        A(f"- **val preference accuracy: {dpo.get('best_val_acc', 0):.3f}** "
          f"(>0.5 = policy prefers the curated answer over the negative)")
        hist = dpo.get("history", [])
        if hist:
            A("\n| step | val_acc | margin | rew_chosen | rew_rejected | LM drift (tulu3/nr/creative) |\n|---|---|---|---|---|---|")
            for e in hist:
                A(f"| {e.get('step')} | {e.get('val_acc', float('nan')):.3f} | "
                  f"{e.get('val_margin',0):.3f} | {e.get('mean_rew_chosen',0):.3f} | "
                  f"{e.get('mean_rew_rejected',0):.3f} | "
                  f"{e.get('lm_drift', float('nan')):.3f} |")
        if dpo.get("pre_dpo_lm_drift"):
            A(f"\nPre-DPO LM drift guard: {dpo['pre_dpo_lm_drift']}")

    A("\n## 5. Artifacts\n")
    A("- checkpoints: `checkpoints/sft_best.pth`, `sft_latest.pth`, `dpo_best.pth`, "
      "`dpo_latest.pth` (+ adapter files with optimizer state)")
    A("- per-checkpoint samples: `reports/samples_*.md`; bench transcripts: `reports/bench_*/`")
    A("- logs: `logs/` (per-stage) and `logs/pipeline.log` (orchestrator)")
    A("- push back to Drive: `scripts/push_results.sh` (rclone → gdrive:synapsenew/)")

    path = os.path.join(REPORT_DIR, "MORNING_REPORT.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
