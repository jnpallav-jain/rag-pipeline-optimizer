"""Exhaustive baseline: run all 81 configurations on the held-out test split (2,000
questions). The agent never sees this split; the grid is the ground truth for
judging the agent's choices.

    python eval/grid.py            -> results/grid_test.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_optimizer.data import load_squad  # noqa: E402
from rag_optimizer.pipeline import Lab, all_configs  # noqa: E402

OUT = ROOT / "results" / "grid_test.csv"


def main(split: str = "test") -> None:
    lab = Lab(load_squad())
    for c in (64, 128, 256):
        lab.index(c)  # build indexes first so they don't distort latency
    rows = []
    t0 = time.perf_counter()
    for cfg in all_configs():
        m = lab.run(cfg, lab.dataset.splits[split])
        rows.append({"config": cfg.key(), **cfg.as_dict(), "accuracy": round(m["accuracy"], 4),
                     "p95_ms": round(m["p95_ms"], 3), "tokens": round(m["tokens"], 1), "n": m["n"]})
        print(f"{cfg.key():28s} acc={m['accuracy']:.3f} p95={m['p95_ms']:.2f}ms tokens={m['tokens']:.0f}")
    OUT.parent.mkdir(exist_ok=True)
    out = ROOT / "results" / f"grid_{split}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} configurations in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
