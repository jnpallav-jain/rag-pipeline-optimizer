"""LangGraph controller tests. A deterministic fake lab (no data, no models) lets the graph and
the v2 plain-Python loop be compared decision by decision."""
import random

import pytest

from rag_optimizer.data import Dataset, Question
from rag_optimizer.guardrails import Budget
from rag_optimizer.human import HumanGate
from rag_optimizer.memory import ExperimentLog, KnowledgeBase
from rag_optimizer.orchestrator import Orchestrator
from tests.legacy_orchestrator import LegacyOrchestrator

LAT = {"bm25": 0.6, "tfidf": 0.5, "bge": 9.0}


class FakeLab:
    """Metrics depend only on the configuration and split, so every run is reproducible."""

    def __init__(self, flat: bool = False):
        self.flat = flat  # flat=True: every config has the same recall, so finalists tie
        qs = {s: [Question(f"{s}{i}", "q", ["a"], "t") for i in range(n)]
              for s, n in (("plan", 300), ("validation", 1000), ("test", 50))}
        self.dataset = Dataset("squad-v1.1-dev", {}, qs)

    def run(self, cfg, questions):
        key = cfg.key()
        rng = random.Random(f"{key}|{len(questions)}")
        p = 0.55 + 0.04 * cfg.top_k + 0.0004 * cfg.chunk_size + {"bm25": 0.05, "tfidf": 0.03, "bge": 0.02}[cfg.retriever] \
            + {"none": 0.0, "overlap": 0.01, "rrf": 0.02}[cfg.rerank]
        if self.flat:
            p = 0.7
        hits = [int(rng.random() < min(p, 0.97)) for _ in questions]
        lat = LAT[cfg.retriever] + (9.0 if cfg.rerank == "rrf" and cfg.retriever != "bge" else 0.0) \
            + (0.3 if cfg.rerank == "overlap" else 0.0)
        tokens = cfg.chunk_size * cfg.top_k * 1.3
        if self.flat:
            lat, tokens = 0.5, 100.0
        return {"accuracy": sum(hits) / len(hits), "p95_ms": lat,
                "tokens": tokens, "n": len(questions), "hits": hits}


def snapshot(res):
    return {
        "rec": res.recommendation and (res.recommendation["config"], res.recommendation["run_id"]),
        "runner": res.runner_up and res.runner_up["config"],
        "conf": res.confidence,
        "stopped": res.stopped_reason,
        "tree": [((n.config.key() if n.config else str(n.partial)), n.depth, n.status, n.reason,
                  round(n.score, 6)) for n in res.tree],
        "esc": [(e.kind, e.decision) for e in res.escalations],
        "budget": res.budget and (res.budget.used_evaluations, res.budget.used_validations),
    }


CASES = [
    dict(req=(200, 5.0)), dict(req=(400, 5.0)), dict(req=(1000, 5.0)), dict(req=(2000, 20.0)),
    dict(req=(1000, 0.55)), dict(req=(50, 5.0)), dict(req=(None, 5.0)),
    dict(req=(400, 5.0, "private-crm-tickets")),
    dict(req=(1000, 5.0), budget=6),
    dict(req=(1000, 5.0), budget=4, scripted={"high_impact_step": "deny"}),
    dict(req=(2000, 20.0), kb=False),
    dict(req=(2000, 20.0), flat=True),
    dict(req=(2000, 20.0), flat=True, scripted={"low_confidence": "runner-up"}),
]


@pytest.mark.parametrize("case", CASES, ids=[str(c) for c in CASES])
def test_graph_matches_v2_loop(case):
    out = []
    for cls in (LegacyOrchestrator, Orchestrator):
        lab = FakeLab(flat=case.get("flat", False))
        kb = KnowledgeBase() if case.get("kb", True) else None
        scripted = dict(case.get("scripted", {}))
        if scripted.get("low_confidence") == "runner-up":
            # learn the runner-up's key from a default run, then have the human choose it
            probe = Orchestrator(FakeLab(flat=True), ExperimentLog(), kb, HumanGate()).run(*case["req"])
            scripted["low_confidence"] = probe.runner_up["config"]
        orch = cls(lab, ExperimentLog(), kb, HumanGate(scripted=scripted),
                   budget=Budget(max_evaluations=case.get("budget", 30)), use_kb=case.get("kb", True))
        out.append(snapshot(orch.run(*case["req"])))
    assert out[0] == out[1]
    if case.get("flat"):
        assert any(k == "low_confidence" for k, _ in out[1]["esc"])
    if case.get("scripted", {}).get("low_confidence") == "runner-up":
        assert out[1]["rec"][0] == scripted["low_confidence"]   # the human's choice was applied


def test_interactive_checkpoint_pauses_and_resumes():
    """Low-confidence / high-impact checkpoints go through LangGraph interrupt() and resume."""
    asked = []

    def prompt(payload):
        asked.append(payload["kind"])
        return "deny" if payload["kind"] == "high_impact_step" else payload["default"]
    orch = Orchestrator(FakeLab(), ExperimentLog(), KnowledgeBase(), HumanGate(mode="interactive"),
                        budget=Budget(max_evaluations=4))
    res = orch.run(1000, 5.0, prompt=prompt)
    assert asked == ["high_impact_step"]
    assert res.stopped_reason == "human denied the depth-1 step"
    assert [(e.kind, e.decision) for e in res.escalations] == [("high_impact_step", "deny")]
