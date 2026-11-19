"""Orchestrator (controller): runs the beam search over the configuration tree,
routes work between agents, enforces the budget, and escalates to a human.

Coordination: sequential outer workflow (clarify -> evidence -> plan/test loop ->
validate -> report) with an iterative Planner -> Experimenter -> Evaluator loop inside.
Only the orchestrator changes a node's status (single writer of shared state)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .agents.evaluator import CONFIDENCE_FLOOR, Evaluator
from .agents.experimenter import Experimenter
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


class Orchestrator:
    def __init__(self, lab: Lab, log: ExperimentLog | None = None, kb: KnowledgeBase | None = None,
                 human: HumanGate | None = None, beam_width: int = 3, budget: Budget | None = None,
                 use_kb: bool = True, experimenter: Experimenter | None = None):
        self.lab = lab
        self.log = log or ExperimentLog()
        self.kb = kb if use_kb else None
        self.human = human or HumanGate()
        self.beam_width = beam_width
        self.budget = budget or Budget()
        self.researcher = Researcher(self.kb, self.log)
        self.experimenter = experimenter or Experimenter(lab, self.log)

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
    def run(self, max_context_tokens, max_p95_ms, dataset: str = "squad-v1.1-dev") -> Result:
        t0 = time.perf_counter()
        # Phase 1: clarify goal (input guardrails)
        try:
            constraints, problems = validate_request(max_context_tokens, max_p95_ms, dataset)
        except GuardrailError as e:
            self.human.notify("policy_block", str(e))
            return Result(None, stopped_reason=f"blocked by guardrail: {e}",
                          escalations=self.human.log, seconds=time.perf_counter() - t0)
        if problems:
            self.human.ask("ambiguous_goal", "; ".join(problems),
                           ["stop and ask the user to restate constraints"],
                           default="stop and ask the user to restate constraints")
            return Result(None, stopped_reason="ambiguous goal: " + "; ".join(problems),
                          escalations=self.human.log, seconds=time.perf_counter() - t0)
        res = Result(constraints, budget=self.budget, escalations=self.human.log,
                     quarantined=self.kb.quarantined if self.kb else [])

        # Phase 2: gather evidence (retrieval)
        evidence = {d: self.researcher.evidence(d) for d in DIMENSIONS}
        res.evidence = evidence
        planner = Planner(evidence, constraints)
        evaluator = Evaluator(constraints)

        # Phase 3: plan + test loop (beam search over the configuration tree)
        beam = [Node(partial={}, depth=0, status="kept")]
        for depth, dim in enumerate(DIMENSIONS, start=1):
            children = []
            for parent in beam:
                for spec in planner.expand(parent.partial):
                    node = Node(partial=spec["partial"], depth=depth, thought=spec["thought"],
                                evidence=planner.node_evidence(spec["partial"]))
                    if spec["min_tokens"] > constraints.max_context_tokens:
                        node.status = "pruned"
                        node.reason = (f"heuristic: even the smallest remaining choice needs "
                                       f"{spec['min_tokens']:.0f} tokens > cap")
                    else:
                        node.config = planner.fill(node.partial)
                    children.append(node)
            res.tree.extend(children)
            to_test = [n for n in children if n.status == "open"]
            planned = sum(1 for n in to_test if not self.researcher.recall(
                n.config.key(), self.lab.dataset.name, "plan"))
            if self.budget.needs_approval(planned):
                decision = self.human.ask(
                    "high_impact_step", f"depth {depth} needs {planned} new experiments "
                    f"(> 50% of the {self.budget.max_evaluations}-run budget)",
                    ["approve", "deny"], default="approve")
                if decision == "deny":
                    res.stopped_reason = f"human denied the depth-{depth} step"
                    break
            for n in sorted(to_test, key=lambda n: -n.evidence):
                needs_run = not self.researcher.recall(n.config.key(), self.lab.dataset.name, "plan")
                if needs_run and self.budget.remaining <= 0:
                    n.status, n.reason = "skipped", "budget exhausted"
                    continue
                self._measure(n, "plan", "evaluations")
                if n.metrics:
                    n.score = evaluator.score(n.metrics, n.evidence)
            measured = [n for n in to_test if n.metrics]
            failed = [n for n in to_test if n.status == "failed"]
            if len(failed) >= 3:
                self.human.notify("repeated_failures", f"{len(failed)} experiments failed at depth {depth}")
            survivors, _ = evaluator.prune(measured)
            if not survivors and measured:
                # Observation -> re-plan: every branch broke a hard constraint with the default
                # fill. Before calling the request infeasible, retest this level with other
                # choices for the still-undecided retriever / re-ranker (cheapest re-ranker first).
                replanned = self._replan(planner, evaluator, children, constraints, res)
                measured += replanned
                survivors, _ = evaluator.prune(measured)
            if not survivors:
                soft = [n for n in measured if not evaluator.violations(n.metrics)]
                if soft:
                    best = max(soft, key=lambda n: n.score)
                    best.status, best.reason = "kept", "revived: best feasible node after whole level was pruned"
                    self.human.notify("level_collapsed", f"all depth-{depth} nodes fell below the score "
                                      f"threshold; continuing with {best.config.key()}")
                    survivors = [best]
                else:
                    suggestion = self._suggest_relaxation(children, constraints)
                    self.human.ask("infeasible", f"no configuration at depth {depth} meets the constraints; "
                                   f"suggestion: {suggestion}", [f"relax: {suggestion}", "stop"],
                                   default="stop")
                    res.stopped_reason = f"infeasible constraints ({suggestion})"
                    break
            # evidence conflict: literature favoured an option that measured clearly worse
            by_opt = {}
            for n in measured:
                by_opt.setdefault(n.partial[dim], []).append(n.metrics["accuracy"])
            if by_opt:
                best_acc = max(max(v) for v in by_opt.values())
                for opt, accs in by_opt.items():
                    if evidence[dim][opt]["score"] >= 0.65 and best_acc - max(accs) > 0.05:
                        self.human.notify("evidence_conflict",
                                          f"{dim}={opt} is supported by {evidence[dim][opt]['notes']} "
                                          f"but measured {best_acc - max(accs):.3f} lower accuracy on this dataset")
            survivors.sort(key=evaluator.tie_key, reverse=True)
            beam = survivors[: self.beam_width]
            for n in measured:
                if n in beam:
                    n.status = "kept"
                elif n.status == "open":
                    n.status, n.reason = "pruned", "outside beam"
        else:
            res.stopped_reason = "search complete"

        if not res.stopped_reason == "search complete":
            res.seconds = time.perf_counter() - t0
            res.experimenter_stats = {"failures": self.experimenter.failures,
                                      "recovered": self.experimenter.recovered}
            return res

        # Phase 4: validate finalists on the validation split
        finalists = []
        for n in beam[:2]:
            v = Node(partial=n.partial, depth=n.depth, config=n.config, evidence=n.evidence)
            self._measure(v, "validation", "validations")
            if v.metrics and not evaluator.violations(v.metrics):
                v.score = evaluator.utility(v.metrics)
                finalists.append(v)
        if not finalists:
            self.human.ask("infeasible", "finalists violate constraints on the validation split",
                           ["stop"], default="stop")
            res.stopped_reason = "finalists failed validation"
            res.seconds = time.perf_counter() - t0
            return res
        finalists.sort(key=lambda n: n.score, reverse=True)
        pick = finalists[0]
        if len(finalists) > 1:
            other = finalists[1]
            run_a, run_b = self.log.get(pick.run_id), self.log.get(other.run_id)
            res.confidence = evaluator.win_probability(pick.metrics, other.metrics, run_a["hits"], run_b["hits"])
            if res.confidence < CONFIDENCE_FLOOR:
                cheaper = min(finalists, key=lambda n: n.metrics["tokens"])
                choice = self.human.ask(
                    "low_confidence",
                    f"top two configurations are statistically close (P(best) = {res.confidence:.2f})",
                    [pick.config.key(), other.config.key()], default=cheaper.config.key())
                if choice == other.config.key():
                    pick, other = other, pick
            res.runner_up = self._summary(other)
        else:
            res.confidence = 1.0
        res.recommendation = self._summary(pick)
        res.seconds = time.perf_counter() - t0
        res.experimenter_stats = {"failures": self.experimenter.failures,
                                  "recovered": self.experimenter.recovered}
        return res

    @staticmethod
    def _summary(n: Node) -> dict:
        return {"config": n.config.key(), "params": n.config.as_dict(), "run_id": n.run_id,
                "metrics": n.metrics, "utility": n.score}
