"""Headline metrics from results/eval_summary.json -> results/headline_metrics.json
and an extra section in results/eval_report.md.

Exact rank is a harsh measure here because several configurations tie within noise on the
test set, so we also report the share of runs whose pick is within 0.01 / 0.02 utility of
the best configuration found by exhaustive search.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_optimizer.agents.evaluator import Evaluator  # noqa: E402
from rag_optimizer.guardrails import Constraints  # noqa: E402

RESULTS = ROOT / "results"
VARIANTS = ("agent", "greedy", "no_kb", "no_score_floor")


def main():
    s = json.loads((RESULTS / "eval_summary.json").read_text())
    grid = list(csv.DictReader((RESULTS / "grid_test.csv").open()))
    for r in grid:
        for k in ("accuracy", "p95_ms", "tokens"):
            r[k] = float(r[k])
    out = {"per_scenario": {}, "overall": {}}
    for name, row in s["scenarios"].items():
        c = Constraints(row["constraints"]["max_context_tokens"], row["constraints"]["max_p95_ms"])
        ev = Evaluator(c)
        feas = [r for r in grid if not ev.violations(r)]
        best_u = max(ev.utility(r) for r in feas)
        sc = {"feasible_configs": len(feas),
              "near_ties_within_0.01": sum(best_u - ev.utility(r) <= 0.01 for r in feas)}
        for v in VARIANTS:
            runs = row[v]["per_run"]
            ok = [p for p in runs if p.get("pick")]
            sc[v] = {
                "recommended_rate": len(ok) / len(runs),
                "within_0.01": sum(p["regret"] <= 0.01 for p in ok) / len(runs),
                "within_0.02": sum(p["regret"] <= 0.02 for p in ok) / len(runs),
                "mean_regret": row[v]["mean_regret"],
                "mean_experiments": row[v]["mean_experiments"],
                "mean_seconds": row[v]["mean_seconds"],
                "mean_test_recall": row[v]["mean_test_accuracy"],
                "low_confidence_rate": row[v]["low_confidence_rate"],
                "all_grounded": row[v]["all_grounded"],
                "cited_numbers": row[v]["cited_numbers"],
            }
        out["per_scenario"][name] = sc
    for v in VARIANTS:
        per = [out["per_scenario"][n][v] for n in out["per_scenario"]]
        out["overall"][v] = {k: round(statistics.mean(p[k] for p in per), 4)
                             for k in ("within_0.01", "within_0.02", "mean_regret", "mean_experiments",
                                       "mean_seconds", "low_confidence_rate", "recommended_rate")}
        out["overall"][v]["all_grounded"] = all(p["all_grounded"] for p in per)
        out["overall"][v]["cited_numbers"] = sum(p["cited_numbers"] for p in per)
    a = out["overall"]["agent"]
    out["overall"]["agent"]["experiment_reduction_vs_exhaustive"] = round(1 - a["mean_experiments"] / 81, 3)
    fresh = sum(s["scenarios"][m["scenario"]]["agent"]["mean_experiments"] for m in s["memory"])
    shared = sum(m["new_experiments"] for m in s["memory"])
    out["memory"] = {"fresh_mean_experiments_sum": round(fresh, 1), "shared_log_new_experiments": shared,
                     "reduction": round(1 - shared / fresh, 3), "detail": s["memory"]}
    out["calibration"] = s["calibration"]
    out["safety_tests"] = {k: v for k, v in s["safety"].items()}
    v1 = RESULTS / "eval_summary_v1_before_replan.json"
    if v1.exists():
        old = json.loads(v1.read_text())["scenarios"]["S5_low_latency"]["agent"]
        out["before_replan_fix_S5"] = {"pick": old.get("pick"), "stopped": old.get("stopped")}
    (RESULTS / "headline_metrics.json").write_text(json.dumps(out, indent=2))

    L = ["", "## Tolerance view (share of runs within 0.01 / 0.02 utility of the exhaustive best)", "",
         "| scenario | feasible | near-ties (<= 0.01) | " + " | ".join(VARIANTS) + " |",
         "|---|---|---|" + "---|" * len(VARIANTS)]
    for n, sc in out["per_scenario"].items():
        L.append(f"| {n} | {sc['feasible_configs']} | {sc['near_ties_within_0.01']} | " + " | ".join(
            f"{sc[v]['within_0.01']:.1f} / {sc[v]['within_0.02']:.1f}" for v in VARIANTS) + " |")
    L.append("| **overall** | | | " + " | ".join(
        f"**{out['overall'][v]['within_0.01']:.2f} / {out['overall'][v]['within_0.02']:.2f}**" for v in VARIANTS) + " |")
    L += ["", f"Experiments per request (agent): {a['mean_experiments']} of 81 "
          f"({out['overall']['agent']['experiment_reduction_vs_exhaustive']:.0%} fewer than exhaustive search).",
          f"Memory: three consecutive requests on one shared log needed {shared} new experiments vs "
          f"{fresh:.1f} when each starts empty ({out['memory']['reduction']:.0%} fewer).", ""]
    report = RESULTS / "eval_report.md"
    text = report.read_text().split("\n## Tolerance view")[0].rstrip("\n") + "\n"
    report.write_text(text + "\n".join(L) + "\n")
    print(json.dumps(out["overall"], indent=2))
    print(json.dumps(out["memory"], indent=2)[:400])


if __name__ == "__main__":
    main()
