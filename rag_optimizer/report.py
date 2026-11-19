"""Planner's final report. Every measured number is cited as `value [run_id.metric]`
so `guardrails.verify_report` can check it against the experiment log."""
from __future__ import annotations

from .orchestrator import Result


def _cite(run_id: str, metrics: dict, metric: str, fmt: str) -> str:
    return f"{format(metrics[metric], fmt)} [{run_id}.{metric}]"


def render(res: Result, kb=None) -> str:
    lines = ["# RAG Pipeline Optimizer report", ""]
    if res.constraints is None:
        lines += [f"**No recommendation.** {res.stopped_reason}", ""]
        return "\n".join(lines + _escalations(res))
    c = res.constraints
    lines += ["## Request", f"- Dataset: `{c.dataset}` (public, allowlisted)",
              f"- Max context tokens: {c.max_context_tokens:.0f}", f"- Max p95 retrieval latency: {c.max_p95_ms} ms", ""]
    if res.recommendation:
        r = res.recommendation
        m, rid = r["metrics"], r["run_id"]
        lines += ["## Recommendation (validation split, 1,000 questions)",
                  f"**`{r['config']}`**: answer recall {_cite(rid, m, 'accuracy', '.3f')}, "
                  f"context tokens {_cite(rid, m, 'tokens', '.1f')}, "
                  f"p95 latency {_cite(rid, m, 'p95_ms', '.2f')} ms.", ""]
        if res.runner_up:
            u = res.runner_up
            mu, ru = u["metrics"], u["run_id"]
            lines += [f"Runner-up `{u['config']}`: answer recall {_cite(ru, mu, 'accuracy', '.3f')}, "
                      f"context tokens {_cite(ru, mu, 'tokens', '.1f')}, "
                      f"p95 latency {_cite(ru, mu, 'p95_ms', '.2f')} ms.", ""]
        conf = res.confidence if res.confidence is not None else 1.0
        flag = " (low: a human reviewed this choice)" if conf < 0.8 else ""
        lines += [f"Confidence that the recommendation beats the runner-up: {conf:.2f}{flag}.",
                  "This recommendation is advisory: an engineer must approve it before production use.", ""]
    else:
        lines += [f"**No recommendation.** {res.stopped_reason}", ""]

    lines += ["## Evidence used (retrieved from the knowledge base)"]
    used = sorted({nid for dim in res.evidence.values() for v in dim.values() for nid in v["notes"]})
    notes = {n.id: n for n in (kb.notes if kb else [])}
    for nid in used:
        if nid in notes:
            lines.append(f"- **{nid}** {notes[nid].title}. {notes[nid].source}")
    if not used:
        lines.append("- none (knowledge base disabled or nothing passed the relevance threshold)")
    for nid, why in res.quarantined:
        lines.append(f"- quarantined {nid}: {why}")
    lines.append("")

    lines += ["## Search tree (plan split, 300 questions)", "",
              "| depth | configuration tested | score | status | note |", "|---|---|---|---|---|"]
    for n in res.tree:
        key = n.config.key() if n.config else str(n.partial)
        sc = f"{n.score:.3f}" if n.metrics else "-"
        src = f" ({n.run_id}{', cached' if n.cached else ''})" if n.run_id else ""
        lines.append(f"| {n.depth} | `{key}`{src} | {sc} | {n.status} | {n.reason} |")
    b = res.budget
    lines += ["", f"Budget: {b.used_evaluations}/{b.max_evaluations} sample experiments, "
              f"{b.used_validations}/{b.max_validations} validations, {b.cache_hits} reused from memory. "
              f"Wall time {res.seconds:.1f} s.", ""]
    return "\n".join(lines + _escalations(res))


def _escalations(res: Result) -> list[str]:
    out = ["## Human checkpoints and notices"]
    if not res.escalations:
        out.append("- none")
    for e in res.escalations:
        kind = "decision" if e.blocking else "notice"
        out.append(f"- [{kind}] **{e.kind}**: {e.reason} -> {e.decision}")
    return out + [""]
