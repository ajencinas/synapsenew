#!/usr/bin/env python3
"""Standardized evals for the Synapse checkpoints: GSM8K (few-shot CoT) and
ARC-Challenge (loglikelihood MC).  No API keys, no LLM judge.

GSM8K  --  4-shot CoT prompting (greedy), answer extracted from the last number.
ARC-C  --  each choice scored by mean token logprob of the continuation; answer
           = argmax.  This is the classic lm-eval-harness protocol (no generation).

Usage:
  python standard_eval.py --ckpt <path> --tag v3ref --n-gsm 200 --n-arc 300
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from common import (REPORT_DIR, DATA_DIR, TOKENIZER_PATH, load_tokenizer,
                    load_state_dict_from_checkpoint, write_json)
from synapse_model import SynapseGPT

GSM_SHOTS = [
    {"q": "Natalia sold clips to 48 of her friends in April, and then she sold "
           "half as many clips in May. How many clips did Natalia sell "
           "altogether in April and May?",
     "a": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold "
          "48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"},
    {"q": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
           "minutes of babysitting. How much did she earn?",
     "a": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 "
          "minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"},
    {"q": "Betty is saving money for a new wallet which costs $100. Betty has "
           "only half of the money she needs. Her parents decided to give her "
           "$15 for that purpose, and her grandparents twice as much as her "
           "parents. How much more money does Betty need to buy the wallet?",
     "a": "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's "
          "grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty "
          "needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5"},
    {"q": "James decides to run 3 sprints 3 times a week. He runs 60 meters "
           "each sprint. How many total meters does he run a week?",
     "a": "He sprints 3*3=<<3*3=9>>9 times\nSo he runs 9*60=<<9*60=540>>540 "
          "meters\n#### 540"},
]


def extract_gsm_answer(text):
    m = re.search(r"####\s*([\-\d,\.\s]+)", text)
    s = m.group(1) if m else None
    if s is None:
        nums = re.findall(r"-?\$?\s?\d[\d,\.]*", text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def load_model(ckpt):
    sd, meta = load_state_dict_from_checkpoint(ckpt)
    vocab = sd["token_embedding.weight"].shape[0]
    model = SynapseGPT(vocab)
    model.load_state_dict(sd, strict=False)
    del sd
    return model.bfloat16().cuda().eval(), vocab, meta


@torch.no_grad()
def greedy_generate(model, ids, max_new, stop_ids):
    for _ in range(max_new):
        if ids.shape[1] >= 2048:
            break
        logits = model(ids)[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if int(nxt) in stop_ids:
            break
        ids = torch.cat([ids, nxt], dim=1)
    return ids


@torch.no_grad()
def choice_logprob(model, tok, prompt, cont):
    ids_p = tok.encode(prompt, add_special_tokens=False).ids
    ids_c = tok.encode(cont, add_special_tokens=False).ids
    if len(ids_c) == 0:
        return float("-inf")
    ids = ids_p + ids_c
    if len(ids) > 2048:
        ids = ids[-2048:]
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    target = torch.tensor(ids[1:], device="cuda")
    sel = logp[0].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    n_cont = min(len(ids_c), len(ids) - 1)
    vals = sel[-n_cont:]
    return vals.mean().item()


def eval_gsm(model, tok, n, stop_ids, fewshot=4):
    path = os.path.join(DATA_DIR, "gsm8k_test.jsonl")
    items = [json.loads(l) for l in open(path)][:n]
    prefix = "".join(f"Q: {s['q']}\nA: {s['a']}\n\n" for s in GSM_SHOTS[:fewshot])
    correct = 0
    samples = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        prompt = prefix + f"Q: {it['question']}\nA:"
        ids = tok.encode(prompt, add_special_tokens=False).ids
        if len(ids) > 1800:
            ids = ids[-1800:]
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        out = greedy_generate(model, x.clone(), 220, stop_ids)
        gen_ids = out[0, len(ids):].tolist()
        text = tok.decode(gen_ids)
        text = text.split("Q:")[0].strip()
        gold = float(it["answer"].split("####")[-1].replace(",", "").strip())
        pred = extract_gsm_answer(text)
        ok = pred is not None and abs(pred - gold) < 1e-4
        correct += int(ok)
        if i <= 6 or (ok and len(samples) < 12):
            samples.append({"q": it["question"][:160], "gold": gold,
                            "pred": pred, "ok": ok, "resp": text[:400]})
        if i % 20 == 0:
            print(f"  gsm {i}/{len(items)} acc={correct/i:.3f} "
                  f"[{(time.time()-t0)/i:.1f}s/q]", flush=True)
    return {"n": len(items), "correct": correct, "acc": correct / len(items),
            "samples": samples}


def eval_arc(model, tok, n):
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(DATA_DIR, "arc_test.parquet")).to_pylist()[:n]
    correct = 0
    samples = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        prompt = f"Question: {r['question']}\nAnswer:"
        scores = []
        for text, lab in zip(r["choices"]["text"], r["choices"]["label"]):
            scores.append((choice_logprob(model, tok, prompt, " " + text), lab))
        scores.sort(key=lambda x: -x[0])
        pred = scores[0][1]
        ok = pred == r["answerKey"]
        correct += int(ok)
        if len(samples) < 8:
            samples.append({"q": r["question"][:150], "gold": r["answerKey"],
                            "pred": pred, "ok": ok})
        if i % 50 == 0:
            print(f"  arc {i}/{len(rows)} acc={correct/i:.3f} "
                  f"[{(time.time()-t0)/i:.1f}s/q]", flush=True)
    return {"n": len(rows), "correct": correct, "acc": correct / len(rows),
            "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-gsm", type=int, default=200)
    ap.add_argument("--n-arc", type=int, default=300)
    ap.add_argument("--fewshot", type=int, default=4)
    ap.add_argument("--skip-gsm", action="store_true")
    ap.add_argument("--skip-arc", action="store_true")
    args = ap.parse_args()

    tok = load_tokenizer(TOKENIZER_PATH)
    eot = tok.token_to_id("<|endoftext|>") or 0
    stop_ids = {eot}
    model, vocab, meta = load_model(args.ckpt)
    print(f"[{args.tag}] loaded {args.ckpt} ({vocab} vocab)")

    out = {"tag": args.tag, "ckpt": args.ckpt, "fewshot": args.fewshot}
    if not args.skip_gsm:
        out["gsm8k"] = eval_gsm(model, tok, args.n_gsm, stop_ids, args.fewshot)
        print(f"[{args.tag}] GSM8K acc = {out['gsm8k']['acc']:.3f} "
              f"({out['gsm8k']['correct']}/{out['gsm8k']['n']})")
    if not args.skip_arc:
        out["arc_challenge"] = eval_arc(model, tok, args.n_arc)
        print(f"[{args.tag}] ARC-C acc = {out['arc_challenge']['acc']:.3f} "
              f"({out['arc_challenge']['correct']}/{out['arc_challenge']['n']})")
    write_json(os.path.join(REPORT_DIR, f"standard_eval_{args.tag}.json"), out)
    print(f"[{args.tag}] wrote reports/standard_eval_{args.tag}.json")


if __name__ == "__main__":
    main()
