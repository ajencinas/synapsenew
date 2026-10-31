#!/usr/bin/env python3
"""Build a self-contained HTML validation dashboard.

Runs a curated set of deterministic-graded bench items through every
checkpoint (pretrain base / synapsenew SFT / synapsenew SFT+DPO / reference
v3 full-FT), grades with the SAME graders as sft_bench (`--no-judge`), and
renders everything side by side into one static HTML file.

Output:
  $SYNAPSENEW_DIR/reports/dashboard.html          (self-contained, no deps)
  $SYNAPSENEW_DIR/reports/dashboard_results.json  (raw generations + verdicts)

Usage:  python scripts/build_dashboard.py [--max-tokens 200]
"""
import argparse
import html
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

import sft_bench as sb
from common import REPORT_DIR, TOKENIZER_PATH, CKPT_DIR

# curated deterministic-gradable items (mix: math, format, retrieval, code,
# language, logic, multi-turn, translation)
CASES = ["arith_01", "reason_01", "algebra_01", "geo_01", "sci_01", "hist_01",
         "if_02", "if_04", "text_01", "text_02", "code_02", "closedqa_01",
         "lang_02", "logic_03", "yesno_02", "trans_02", "mt_03"]

MODELS = [
    ("Pretrain base", os.path.join(CKPT_DIR, "pretrain_v2_full.pth"),
     "84347 steps of pretraining · raw base model (no chat tuning)"),
    ("synapsenew SFT", os.path.join(CKPT_DIR, "sft_best.pth"),
     "LoRA r=64 · 1200 steps · 6.1 h · val 1.248 · bench 70%"),
    ("synapsenew SFT+DPO", os.path.join(CKPT_DIR, "dpo_best.pth"),
     "DPO 2 ep · val-acc 0.669 · bench 75%"),
    ("Reference v3 full-FT", os.environ.get(
        "REF_V3_SFT_BEST", os.path.join(CKPT_DIR, "ref_v3_sft_best.pth")),
     "original A100 full fine-tune · 2938 steps · bench 70%"),
    ("Soup: v3 + SFT (50/50)", os.path.join(CKPT_DIR, "soup_v3_ours_a50.pth"),
     "weight average · val 1.155 · bench 65%"),
    ("synapsenew big-LoRA SFT", os.path.join(CKPT_DIR, "sft_best_biglora.pth"),
     "LoRA r=256 · 2500 steps · max local"),
]

GEN_CFG = {"max_tokens": 200, "temperature": 1.0, "top_k": 1, "top_p": 1.0,
           "repetition_penalty": 1.0}   # greedy — same decoding as the bench


def load_cases():
    items = {}
    path = os.path.join(REPO, "bench", "sft_bench.jsonl")
    for line in open(path):
        d = json.loads(line)
        items[d["id"]] = d
    missing = [c for c in CASES if c not in items]
    if missing:
        raise SystemExit(f"bench missing ids: {missing}")
    return [items[c] for c in CASES]


def run_model(label, ckpt, tok, eot_id, device, items):
    print(f"\n[dashboard] {label}: {ckpt}")
    model, info = sb.SynapseInfer.from_checkpoint(ckpt, device=device)
    gens = {}
    for item in items:
        t0 = time.time()
        resp = sb.generate_reply(model, tok, eot_id, device, item,
                                 sb.DEFAULT_SYSTEM_PROMPT, GEN_CFG)
        det = sb.grade_deterministic(item, resp)
        gens[item["id"]] = {"response": resp, "verdict": det,
                            "secs": round(time.time() - t0, 1)}
        mark = {True: "PASS", False: "FAIL", None: "-"}[det]
        print(f"  [{mark}] {item['id']:<14} {resp[:70]!r}")
    del model
    torch.cuda.empty_cache()
    return {"ckpt": ckpt, "step": (info or {}).get("step"), "gens": gens}


CSS = """
:root{--bg:#0b1220;--panel:#111a2e;--panel2:#0f172a;--ink:#e8eefc;--muted:#8fa3c7;
--ok:#22c55e;--ok-bg:#0c2817;--bad:#ef4444;--bad-bg:#2a0f12;--na:#94a3b8;--line:#22304d;--accent:#5eead4;}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:28px 20px 80px}
.wrap{max-width:1180px;margin:0 auto}h1{font-size:26px;margin:0 0 4px}h1 .ac{color:var(--accent)}
.sub{color:var(--muted);margin-bottom:22px}
table.sum{border-collapse:collapse;width:100%;background:var(--panel);border-radius:12px;overflow:hidden;margin:14px 0 30px}
table.sum th,table.sum td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line)}
table.sum th{color:var(--muted);font-weight:600;background:var(--panel2)}
.big{font-weight:700;font-size:16px}.score{font-weight:700}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:18px 0}
.card h3{margin:0 0 2px;font-size:17px}.card .cat{color:var(--muted);font-size:12.5px;text-transform:uppercase;letter-spacing:.06em}
.prompt{background:var(--panel2);border-radius:10px;padding:12px 14px;margin:10px 0;white-space:pre-wrap}
.expected{color:var(--muted);font-size:13.5px;margin:6px 0 14px}
.expected b{color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
.model{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.model .mhead{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px}
.model .mname{font-weight:600;font-size:13.5px}
.badge{font-size:11.5px;font-weight:700;padding:2px 9px;border-radius:20px;letter-spacing:.03em}
.b-ok{color:var(--ok);background:var(--ok-bg);border:1px solid #14532d}
.b-bad{color:var(--bad);background:var(--bad-bg);border:1px solid #7f1d1d}
.b-na{color:var(--na);background:#1e293b;border:1px solid var(--line)}
.resp{white-space:pre-wrap;font-size:13.5px;max-height:170px;overflow:auto;color:#dbe6ff}
.pass .resp{border-left:3px solid var(--ok);padding-left:9px}
.fail .resp{border-left:3px solid var(--bad);padding-left:9px}
.na .resp{border-left:3px solid var(--line);padding-left:9px}
.tcount{color:var(--muted);font-size:12.5px;margin-top:10px}
footer{color:var(--muted);font-size:13px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}
code{background:var(--panel2);padding:1px 6px;border-radius:6px}
"""


def esc(x):
    return html.escape(str(x))


def build_html(results, items, wall):
    totals = {label: [0, 0] for label in results}
    for label, data in results.items():
        for it in items:
            v = data["gens"][it["id"]]["verdict"]
            if v is not None:
                totals[label][1] += 1
                totals[label][0] += 1 if v else 0

    H = ["<!doctype html><html><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>synapsenew — validation dashboard</title>",
         f"<style>{CSS}</style></head><body><div class='wrap'>",
         "<h1>synapsenew <span class='ac'>validation dashboard</span></h1>",
         f"<div class='sub'>{len(items)} deterministic-graded tests × "
         f"{len(results)} checkpoints · greedy decoding · generated "
         f"{time.strftime('%Y-%m-%d %H:%M')} · {wall/60:.1f} min GPU time<br>"
         "Grading = the exact deterministic graders from <code>sft_bench.py</code> "
         "(numeric / contains / regex / keywords). Same items for every model.</div>"]

    # summary table
    H.append("<table class='sum'><tr><th>checkpoint</th><th>pass</th>"
             "<th>rate</th><th>description</th></tr>")
    for label, desc in [(m[0], m[2]) for m in MODELS if m[0] in results]:
        p, t = totals[label]
        rate = f"{100*p/t:.0f}%" if t else "n/a"
        cls = "b-ok" if t and p/t >= 0.7 else ("b-bad" if t and p/t < 0.4 else "b-na")
        H.append(f"<tr><td class='big'>{esc(label)}</td>"
                 f"<td class='score'>{p}/{t}</td>"
                 f"<td><span class='badge {cls}'>{rate}</span></td>"
                 f"<td style='color:var(--muted)'>{esc(desc)}</td></tr>")
    H.append("</table>")

    # per-test cards
    for it in items:
        iid = it["id"]
        npass = sum(1 for lb in results if results[lb]["gens"][iid]["verdict"])
        maxt = sum(1 for lb in results if results[lb]["gens"][iid]["verdict"] is not None)
        q = sb.render_for_judge(it)
        H.append("<div class='card'>")
        H.append(f"<h3>{esc(iid)}</h3><div class='cat'>{esc(it.get('category',''))}</div>")
        H.append(f"<div class='prompt'>{esc(q)}</div>")
        if it.get("expected") not in (None, "None", ""):
            H.append(f"<div class='expected'>expected: <b>{esc(it['expected'])}</b> "
                     f"(grader: {esc(it.get('grader'))})</div>")
        H.append(f"<div class='tcount'>{npass}/{maxt} models passed</div>")
        H.append("<div class='grid'>")
        for label, _, _ in MODELS:
            if label not in results:
                continue
            g = results[label]["gens"][iid]
            v = g["verdict"]
            badge = ("b-ok", "PASS") if v else (("b-bad", "FAIL") if v is False
                                                else ("b-na", "—"))
            cls = "pass" if v else ("na" if v is None else "fail")
            H.append(f"<div class='model {cls}'><div class='mhead'>"
                     f"<span class='mname'>{esc(label)}</span>"
                     f"<span class='badge {badge[0]}'>{badge[1]}</span></div>"
                     f"<div class='resp'>{esc(g['response']) or '<i>(empty)</i>'}</div></div>")
        H.append("</div></div>")

    H.append("<footer>Generated by <code>scripts/build_dashboard.py</code> · "
             "checkpoints: <code>sft_best.pth</code> / <code>dpo_best.pth</code> "
             "(synapsenew) · full transcripts + per-source val loss in the "
             "companion reports · pushed to <code>gdrive:synapsenew/reports/</code>"
             "</footer></div></body></html>")
    return "\n".join(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()
    GEN_CFG["max_tokens"] = args.max_tokens

    items = load_cases()
    tok, eot_id = sb.load_tokenizer(TOKENIZER_PATH)
    device = torch.device("cuda")

    results = {}
    t0 = time.time()
    for label, ckpt, _desc in MODELS:
        if not os.path.exists(ckpt):
            print(f"[dashboard] SKIP {label}: {ckpt} not found")
            continue
        results[label] = run_model(label, ckpt, tok, eot_id, device, items)

    wall = time.time() - t0
    html_txt = build_html(results, items, wall)
    html_path = os.path.join(REPORT_DIR, "dashboard.html")
    with open(html_path, "w") as f:
        f.write(html_txt)
    json_path = os.path.join(REPORT_DIR, "dashboard_results.json")
    with open(json_path, "w") as f:
        json.dump({"items": items, "models": MODELS, "results":
                   {k: {"ckpt": v["ckpt"], "step": v["step"], "gens": v["gens"]}
                    for k, v in results.items()}, "wall_s": round(wall, 1)}, f, indent=2)
    print(f"\n[dashboard] wrote {html_path} ({os.path.getsize(html_path)/1e3:.0f} KB)")
    print(f"[dashboard] wrote {json_path}")
    for label, data in results.items():
        n = sum(1 for it in items if data["gens"][it["id"]]["verdict"])
        t = sum(1 for it in items if data["gens"][it["id"]]["verdict"] is not None)
        print(f"  {label:<24} {n}/{t} passed")


if __name__ == "__main__":
    main()
