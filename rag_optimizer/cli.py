"""Command line entry point.

    python -m rag_optimizer.cli --max-context-tokens 400 --max-p95-ms 5
    python -m rag_optimizer.cli --max-context-tokens 400 --max-p95-ms 5 --interactive
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
    return 0 if check["grounded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
