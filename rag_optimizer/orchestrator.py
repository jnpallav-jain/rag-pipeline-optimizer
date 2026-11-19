"""Orchestrator (controller): runs the beam search over the configuration tree,
routes work between agents, enforces the budget, and escalates to a human. The control flow
is a LangGraph state graph (graph.py); this class holds the shared helpers and resources.

Coordination: sequential outer workflow (clarify -> evidence -> plan/test loop ->
validate -> report) with an iterative Planner -> Experimenter -> Evaluator loop inside.
Only the orchestrator changes a node's status (single writer of shared state)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .agents.evaluator import CONFIDENCE_FLOOR, Evaluator
from .agents.experimenter import Experimenter
from .agents.rule_agents import RuleAgents
from .agents.planner import Planner
from .agents.researcher import Researcher
from .guardrails import Budget, Constraints, GuardrailError, validate_request
from .human import HumanGate
from .memory import ExperimentLog, KnowledgeBase
from .pipeline import DIMENSIONS, OPTIONS, TOKENS_PER_WORD, Config, Lab


@dataclass
class Node:
    partial: dict
    depth: int
    thought: str = ""
    config: Config | None = None
    run_id: str | None = None
    metrics: dict | None = None
    score: float = 0.0
    evidence: float = 0.5
    status: str = "open"          # open | kept | pruned | failed | skipped
    reason: str = ""
    cached: bool = False


@dataclass
class Result:
    constraints: Constraints | None
    recommendation: dict | None = None
    runner_up: dict | None = None
    confidence: float | None = None
    tree: list[Node] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    budget: Budget | None = None
    escalations: list = field(default_factory=list)
    quarantined: list = field(default_factory=list)
    stopped_reason: str = ""
    seconds: float = 0.0
    experimenter_stats: dict = field(default_factory=dict)
    agents: str = "rule"
    agent_notes: list = field(default_factory=list)   # LLM agents' commentary (thoughts, critiques)
    summary: str = ""                                 # Planner agent's written summary (LLM mode)


class Orchestrator:
    def __init__(self, lab: Lab, log: ExperimentLog | None = None, kb: KnowledgeBase | None = None,
                 human: HumanGate | None = None, beam_width: int = 3, budget: Budget | None = None,
                 use_kb: bool = True, experimenter: Experimenter | None = None, agents=None):
        self.lab = lab
        self.log = log or ExperimentLog()
        self.kb = kb if use_kb else None
        self.human = human or HumanGate()
        self.beam_width = beam_width
        self.budget = budget or Budget()
        self.researcher = Researcher(self.kb, self.log)
        self.experimenter = experimenter or Experimenter(lab, self.log)
        self.agents = agents or RuleAgents()
        from .graph import build_graph
        self.graph = build_graph(self)

    # ------------------------------------------------------------------ helpers
    def _measure(self, node: Node, split: str, count_against: str) -> None:
        cached = self.researcher.recall(node.config.key(), self.lab.dataset.name, split)
        if cached:
            run, node.cached = cached, True
            self.budget.cache_hits += 1
        else:
            run = self.experimenter.run(node.config, split)
            if count_against == "evaluations":
                self.budget.used_evaluations += 1
            else:
                self.budget.used_validations += 1
        node.run_id = run["run_id"]
        if run["status"] != "ok":
            node.status, node.reason = "failed", "; ".join(run.get("errors", []))
            return
        node.metrics = run["metrics"]

    def _suggest_relaxation(self, nodes: list[Node], c: Constraints) -> str:
        measured = [n.metrics for n in nodes if n.metrics]
        min_tokens = min([m["tokens"] for m in measured] or
                         [min(OPTIONS["chunk_size"]) * min(OPTIONS["top_k"]) * TOKENS_PER_WORD])
        min_ms = min([m["p95_ms"] for m in measured] or [c.max_p95_ms])
        parts = []
        if min_tokens > c.max_context_tokens:
            parts.append(f"raise max_context_tokens to at least {min_tokens:.0f}")
        if min_ms > c.max_p95_ms:
            parts.append(f"raise max_p95_ms to at least {min_ms:.2f}")
        return " and ".join(parts) or "relax either constraint"

    def _replan(self, planner: Planner, evaluator: Evaluator, children: list[Node],
                constraints: Constraints, res: Result) -> list[Node]:
        base = [n for n in children if n.config is not None]
        if not base:
            return []
        undecided = [d for d in ("retriever", "rerank") if d not in base[0].partial]
        if not undecided:
            return []
        tried = {(n.config.retriever, n.config.rerank) for n in base}
        rerank_order = ["none", "overlap", "rrf"] if "rerank" in undecided else [base[0].config.rerank]
        retr_order = (sorted(OPTIONS["retriever"], key=lambda o: -planner.evidence["retriever"][o]["score"])
                      if "retriever" in undecided else [base[0].config.retriever])
        fills = [(r, rr) for rr in rerank_order for r in retr_order if (r, rr) not in tried]
        self.human.notify("replan", f"all {len(base)} branches at depth {base[0].depth} broke a hard "
                          f"constraint; retesting with alternative fills {fills}")
        added: list[Node] = []
        for retriever, rerank in fills:
            if self.budget.remaining < len(base):
                break
            round_nodes = []
            for n in base:
                fill = {**n.config.as_dict(), **{k: v for k, v in
                        (("retriever", retriever), ("rerank", rerank)) if k in undecided}}
                m = Node(partial=n.partial, depth=n.depth, thought=n.thought + f" Re-planned fill: {fill}.",
                         config=Config(**fill), evidence=n.evidence)
                self._measure(m, "plan", "evaluations")
                if m.metrics:
                    m.score = evaluator.score(m.metrics, m.evidence)
                round_nodes.append(m)
            res.tree.extend(round_nodes)
            added += [m for m in round_nodes if m.metrics]
            if any(m.metrics and not evaluator.violations(m.metrics) for m in round_nodes):
                # remember what worked so later fills don't repeat the failed default
                planner.overrides.update({d: v for d, v in (("retriever", retriever), ("rerank", rerank))
                                          if d in undecided})
                break
        return added

    # ------------------------------------------------------------------ main loop
    def run(self, max_context_tokens, max_p95_ms, dataset: str = "squad-v1.1-dev", prompt=None) -> Result:
        """Run the LangGraph controller (rag_optimizer/graph.py). `prompt(payload) -> choice`
        answers human checkpoints in interactive mode."""
        from .graph import run_graph
        ctx = run_graph(self, {"max_context_tokens": max_context_tokens, "max_p95_ms": max_p95_ms,
                               "dataset": dataset}, prompt=prompt)
        res = ctx.res
        res.seconds = time.perf_counter() - ctx.t0
        if res.constraints is not None:
            res.experimenter_stats = {"failures": self.experimenter.failures,
                                      "recovered": self.experimenter.recovered}
        res.agents = self.agents.name
        return res

    @staticmethod
    def _summary(n: Node) -> dict:
        return {"config": n.config.key(), "params": n.config.as_dict(), "run_id": n.run_id,
                "metrics": n.metrics, "utility": n.score}
