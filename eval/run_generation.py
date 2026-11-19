"""End-to-end check with an LLM reader (Claude Haiku 5.5): does retrieval recall predict answer
quality, and does more context hurt the reader?

    python eval/run_generation.py --pilot   # 20 standard calls on plan questions; measures tokens,
                                            # cost and latency -> results/generation_pilot.json
    python eval/run_generation.py           # <=6 configs x 200 test questions via the Batch API
                                            # -> results/generation_summary.json, generation_report.md

Configs: the top-k sweep (k = 1, 3, 5) at the chunk size / retriever / re-ranker of the
highest-recall test config, plus up to 3 of the agent's most frequent picks (one per scenario,
deduplicated). Spend is capped by rag_optimizer.reader (hard $1 across all runs).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_optimizer.data import load_squad  # noqa: E402
from rag_optimizer.pipeline import Config, Lab, normalize_answer  # noqa: E402
from rag_optimizer.reader import (BATCH_DISCOUNT, CAP_USD, MODEL, Reader, build_prompt,  # noqa: E402
                                  exact_match, f1)

RESULTS = ROOT / "results"
N_QUESTIONS, SEED, MAX_CONFIGS, PILOT_N = 200, 7, 6, 20


def parse_key(key: str) -> Config:
    c, r, k, rr = key.split("-")
    return Config(int(c[1:]), r, int(k[1:]), rr)


def choose_configs() -> list[tuple[str, str]]:
    grid = list(csv.DictReader((RESULTS / "grid_test.csv").open()))
    best = max(grid, key=lambda r: float(r["accuracy"]))
    chosen = [(f"c{best['chunk_size']}-{best['retriever']}-k{k}-{best['rerank']}", f"top-k sweep (k={k})")
              for k in (1, 3, 5)]
    summary = json.loads((RESULTS / "eval_summary.json").read_text())
    for name, sc in summary["scenarios"].items():
        picks = sc["agent"].get("picks") or {}
        if not picks:
            continue
        top = Counter(picks).most_common(1)[0][0]
        if top not in [c for c, _ in chosen] and len(chosen) < MAX_CONFIGS:
            chosen.append((top, f"agent pick, {name}"))
    return chosen


def contexts(lab: Lab, cfg: Config, questions) -> list[list[str]]:
    idx = lab.index(cfg.chunk_size)
    return [[idx.texts[i] for i in idx.retrieve(q.text, cfg)[0]] for q in questions]


def pilot(lab: Lab, reader: Reader) -> None:
    cfg = Config(128, "bm25", 3, "none")
    qs = lab.dataset.splits["plan"][:PILOT_N]
    recs = []
    for q, ctx in zip(qs, contexts(lab, cfg, qs)):
        rec = reader.answer(build_prompt(q.text, ctx), tag=f"pilot:{q.qid}")
        rec["em"], rec["f1"] = exact_match(rec["answer"], q.answers), f1(rec["answer"], q.answers)
        recs.append(rec)
    fresh = [r for r in recs if "latency_ms" in r]
    out = {
        "model": MODEL, "config": cfg.key(), "calls": len(recs),
        "mean_input_tokens": statistics.mean(r["input_tokens"] for r in recs),
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in recs),
        "mean_cost_usd_standard": statistics.mean(r["cost_usd"] for r in recs),
        "total_cost_usd": sum(r["cost_usd"] for r in recs),
        "latency_ms_p50": statistics.median(r["latency_ms"] for r in fresh) if fresh else None,
        "latency_ms_max": max(r["latency_ms"] for r in fresh) if fresh else None,
        "em": statistics.mean(r["em"] for r in recs), "f1": statistics.mean(r["f1"] for r in recs),
        "examples": [{"question": q.text, "gold": q.answers[0], "answer": r["answer"]}
                     for q, r in list(zip(qs, recs))[:5]],
        "ledger": reader.ledger.load(),
    }
    (RESULTS / "generation_pilot.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))


def main_run(lab: Lab, reader: Reader) -> None:
    pil = json.loads((RESULTS / "generation_pilot.json").read_text())
    rng = random.Random(SEED)
    qs = rng.sample(lab.dataset.splits["test"], N_QUESTIONS)
    configs = choose_configs()
    prompts, meta = {}, {}
    for key, why in configs:
        cfg = parse_key(key)
        for q, ctx in zip(qs, contexts(lab, cfg, qs)):
            cid = f"{key}|{q.qid}"
            prompts[cid] = build_prompt(q.text, ctx)
            norm = normalize_answer(" ".join(ctx))
            meta[cid] = (key, q, any(normalize_answer(a) in norm for a in q.answers if normalize_answer(a)))
    uncached = sum(1 for p in set(prompts.values()) if reader.key(p) not in reader.cache)
    # projection from the pilot's measured tokens, scaled by this run's longer contexts
    words = statistics.mean(len(p.split()) for p in prompts.values())
    pilot_words = pil["mean_input_tokens"] / 1.3
    per_call = pil["mean_cost_usd_standard"] * max(1.0, words / pilot_words) * BATCH_DISCOUNT
    spent = reader.ledger.load()["total_usd"]
    print(f"{len(configs)} configs x {N_QUESTIONS} questions = {len(prompts)} prompts, "
          f"{uncached} not cached; projected ${uncached * per_call:.4f} "
          f"(spent so far ${spent:.4f} of ${CAP_USD:.2f})")
    for k, why in configs:
        print(f"  {k}: {why}")
    results = reader.answer_batch(prompts)

    rows = []
    for key, why in configs:
        ids = [cid for cid in prompts if meta[cid][0] == key and cid in results]
        ems = [exact_match(results[c]["answer"], meta[c][1].answers) for c in ids]
        f1s = [f1(results[c]["answer"], meta[c][1].answers) for c in ids]
        rec = [meta[c][2] for c in ids]
        nf = [results[c]["answer"].lower().startswith("not found") for c in ids]
        # answer quality given the answer was retrieved: isolates the reader from retrieval
        em_hit = [e for e, h in zip(ems, rec) if h]
        rows.append({
            "config": key, "why": why, "n": len(ids),
            "recall": round(statistics.mean(rec), 3), "em": round(statistics.mean(ems), 3),
            "f1": round(statistics.mean(f1s), 3),
            "em_when_retrieved": round(statistics.mean(em_hit), 3) if em_hit else None,
            "not_found_rate": round(statistics.mean(nf), 3),
            "mean_input_tokens": round(statistics.mean(results[c]["input_tokens"] for c in ids), 1),
            "cost_usd": round(sum(results[c]["cost_usd"] for c in ids), 5),
        })
    out = {"model": MODEL, "questions": N_QUESTIONS, "seed": SEED, "split": "test",
           "metric_note": "recall = gold answer string in retrieved context; EM/F1 = official SQuAD "
                          "metrics on Haiku's answer; costs from API-reported tokens at list price, "
                          "Batch API 50%",
           "configs": rows, "ledger": reader.ledger.load()}
    (RESULTS / "generation_summary.json").write_text(json.dumps(out, indent=2))
    L = ["# End-to-end results: retrieval + Claude Haiku 5.5 reader", "",
         f"{N_QUESTIONS} questions sampled (seed {SEED}) from the held-out test split. Recall is the "
         "share of questions whose gold answer appears in the retrieved context; EM and F1 are the "
         "official SQuAD metrics on the model's answer. *EM when retrieved* counts only questions "
         "whose answer was in the context, so it measures the reader alone.", "",
         "| config | why | recall | EM | F1 | EM when retrieved | 'not found' | input tokens | cost |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['config']}` | {r['why']} | {r['recall']:.3f} | {r['em']:.3f} | {r['f1']:.3f} | "
                 f"{r['em_when_retrieved']} | {r['not_found_rate']:.3f} | {r['mean_input_tokens']:.0f} | "
                 f"${r['cost_usd']:.4f} |")
    led = reader.ledger.load()
    L += ["", f"Total LLM spend for the project so far: ${led['total_usd']:.4f} over {led['calls']} "
              f"calls ({led['input_tokens']:,} input / {led['output_tokens']:,} output tokens, "
              f"as reported by the API). Hard cap ${CAP_USD:.2f}."]
    (RESULTS / "generation_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    lab = Lab(load_squad())
    reader = Reader()
    pilot(lab, reader) if args.pilot else main_run(lab, reader)
