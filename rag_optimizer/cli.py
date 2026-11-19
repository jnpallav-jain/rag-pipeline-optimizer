"""Command line entry point.

    python -m rag_optimizer.cli --max-context-tokens 400 --max-p95-ms 5
    python -m rag_optimizer.cli --max-context-tokens 400 --max-p95-ms 5 --interactive
    python -m rag_optimizer.cli --max-context-tokens 400 --max-p95-ms 5 --generate 3
        (also answers 3 validation questions with Claude Haiku 5.5 using the recommended config;
         needs ANTHROPIC_API_KEY, counts toward the $1 project cap)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_squad
from .guardrails import verify_report
from .human import HumanGate
from .memory import ExperimentLog, KnowledgeBase
from .orchestrator import Orchestrator
from .pipeline import Lab
from .report import render

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find the best RAG retrieval configuration under constraints.")
    ap.add_argument("--max-context-tokens", type=float, required=False)
    ap.add_argument("--max-p95-ms", type=float, required=False)
    ap.add_argument("--dataset", default="squad-v1.1-dev")
    ap.add_argument("--interactive", action="store_true", help="ask a human at every checkpoint")
    ap.add_argument("--log", default=str(ROOT / "results" / "experiment_log.jsonl"),
                    help="experiment log (long-term memory); reused across runs")
    ap.add_argument("--out", default=str(ROOT / "results" / "report.md"))
    ap.add_argument("--generate", type=int, default=0, metavar="N",
                    help="answer N validation questions with the LLM reader (off by default)")
    args = ap.parse_args(argv)

    lab = Lab(load_squad())
    kb = KnowledgeBase()
    log = ExperimentLog(Path(args.log))
    human = HumanGate(mode="interactive" if args.interactive else "auto")
    res = Orchestrator(lab, log, kb, human).run(args.max_context_tokens, args.max_p95_ms, args.dataset)
    md = render(res, kb)
    check = verify_report(md, log)
    md += (f"\nSource verification: {check['cited_numbers']} cited numbers checked against the "
           f"experiment log, {len(check['mismatches'])} mismatches.\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(md)
    print(f"Report written to {args.out}")
    if args.generate and res.recommendation:
        demo_answers(lab, res.recommendation["config"], args.generate)
    return 0 if check["grounded"] else 1


def demo_answers(lab: Lab, config_key: str, n: int) -> None:
    from .pipeline import Config
    from .reader import Reader, build_prompt, exact_match
    c, r, k, rr = config_key.split("-")
    cfg = Config(int(c[1:]), r, int(k[1:]), rr)
    idx, reader = lab.index(cfg.chunk_size), Reader()
    print(f"\nLLM answers with {config_key} (Claude Haiku 5.5):")
    for q in lab.dataset.splits["validation"][:n]:
        ids, _ = idx.retrieve(q.text, cfg)
        rec = reader.answer(build_prompt(q.text, [idx.texts[i] for i in ids]), tag=f"cli:{q.qid}")
        ok = "correct" if exact_match(rec["answer"], q.answers) else "wrong"
        print(f"  Q: {q.text}\n  A: {rec['answer']}  (gold: {q.answers[0]}; {ok})")
    print(f"LLM spend so far: ${reader.ledger.load()['total_usd']:.4f} of $1.00")


if __name__ == "__main__":
    raise SystemExit(main())
