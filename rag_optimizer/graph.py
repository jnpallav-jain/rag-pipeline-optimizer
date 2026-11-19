"""LangGraph controller: the Tree of Thought beam search (Checkpoint 4) as a state graph.

    clarify -> evidence -> expand -> test -> evaluate --(depth < 4)--> expand
                  |           |                  |
                  |           '--> human (approve/deny high-impact step)
                  |                              '--(depth 4 done)--> validate -> finalize -> END
                  '--> human (ambiguous goal / infeasible / low confidence) -> END or finalize

Graph state holds the control flow (depth, phase, pending human decision). The run's working
memory (search tree, beam, budget, agents) lives in a run context owned by the controller, so the
heavy objects (indexes, models) never pass through the checkpointer. Human checkpoints use
LangGraph `interrupt()` in interactive mode: the run pauses, the CLI asks the person, and the
graph resumes from the checkpoint with `Command(resume=choice)`.

Only the controller changes a node's status (single writer of shared state, Checkpoint 5).
Numbers come only from measured experiments; agents (rule-based or CrewAI) can shape thoughts,
fills, evidence and commentary, never scores or pruning (Checkpoint 4 mitigation).
"""
from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .agents.evaluator import CONFIDENCE_FLOOR, Evaluator
from .agents.planner import Planner
from .guardrails import GuardrailError, validate_request
from .human import Escalation
from .pipeline import DIMENSIONS


class GraphState(TypedDict, total=False):
    depth: int
    pending: dict | None      # human decision waiting: kind, reason, options, default, then
    decision: str | None
    done: bool


class RunContext:
    """Working memory of one run (not serialized)."""

    def __init__(self, request: dict):
        self.request = request
        self.t0 = time.perf_counter()
        self.res = None
        self.planner: Planner | None = None
        self.evaluator: Evaluator | None = None
        self.beam: list = []
        self.children: list = []
        self.to_test: list = []
        self.finalists: list = []


def build_graph(ctl) -> Any:
    """ctl: the Orchestrator (helpers, agents, budget, human gate)."""
    from .orchestrator import Node, Result

    def ctx() -> RunContext:
        return ctl._ctx

    # ------------------------------------------------------------ phase 1: clarify
    def clarify(state: GraphState) -> dict:
        c = ctx()
        try:
            constraints, problems = validate_request(**c.request)
        except GuardrailError as e:
            ctl.human.notify("policy_block", str(e))
            c.res = Result(None, stopped_reason=f"blocked by guardrail: {e}", escalations=ctl.human.log)
            return {"done": True}
        if problems:
            c.res = Result(None, stopped_reason="ambiguous goal: " + "; ".join(problems),
                           escalations=ctl.human.log)
            return {"pending": {"kind": "ambiguous_goal", "reason": "; ".join(problems),
                                "options": ["stop and ask the user to restate constraints"],
                                "default": "stop and ask the user to restate constraints",
                                "then": "stop"}}
        c.res = Result(constraints, budget=ctl.budget, escalations=ctl.human.log,
                       quarantined=ctl.kb.quarantined if ctl.kb else [])
        return {}

    # ------------------------------------------------------------ phase 2: evidence
    def evidence(state: GraphState) -> dict:
        c = ctx()
        ev = ctl.agents.research(ctl.researcher, c.res.constraints, c.res)
        c.res.evidence = ev
        c.planner = Planner(ev, c.res.constraints)
        c.evaluator = Evaluator(c.res.constraints)
        c.beam = [Node(partial={}, depth=0, status="kept")]
        return {"depth": 1}

    # ------------------------------------------------------------ phase 3: plan + test loop
    def expand(state: GraphState) -> dict:
        c, depth = ctx(), state["depth"]
        cons = c.res.constraints
        children = []
        for parent in c.beam:
            specs = c.planner.expand(parent.partial)
            for spec in specs:
                node = Node(partial=spec["partial"], depth=depth, thought=spec["thought"],
                            evidence=c.planner.node_evidence(spec["partial"]))
                if spec["min_tokens"] > cons.max_context_tokens:
                    node.status = "pruned"
                    node.reason = (f"heuristic: even the smallest remaining choice needs "
                                   f"{spec['min_tokens']:.0f} tokens > cap")
                else:
                    node.config = c.planner.fill(node.partial)
                children.append(node)
        # Planner agent: writes the Thought for each branch and may choose the fills for
        # still-undecided choices (validated against the allowed options and the token cap).
        ctl.agents.plan(c.planner, children, DIMENSIONS[depth - 1], depth, cons, c.res)
        c.res.tree.extend(children)
        c.children = children
        c.to_test = [n for n in children if n.status == "open"]
        planned = sum(1 for n in c.to_test if not ctl.researcher.recall(
            n.config.key(), ctl.lab.dataset.name, "plan"))
        if ctl.budget.needs_approval(planned):
            return {"pending": {"kind": "high_impact_step",
                                "reason": f"depth {depth} needs {planned} new experiments "
                                          f"(> 50% of the {ctl.budget.max_evaluations}-run budget)",
                                "options": ["approve", "deny"], "default": "approve",
                                "then": "approval"}}
        return {}

    def test(state: GraphState) -> dict:
        c = ctx()
        for n in sorted(c.to_test, key=lambda n: -n.evidence):
            needs_run = not ctl.researcher.recall(n.config.key(), ctl.lab.dataset.name, "plan")
            if needs_run and ctl.budget.remaining <= 0:
                n.status, n.reason = "skipped", "budget exhausted"
                continue
            ctl._measure(n, "plan", "evaluations")
            if n.metrics:
                n.score = c.evaluator.score(n.metrics, n.evidence)
        return {}

    def evaluate(state: GraphState) -> dict:
        c, depth = ctx(), state["depth"]
        dim, ev, evaluator = DIMENSIONS[depth - 1], c.res.evidence, c.evaluator
        measured = [n for n in c.to_test if n.metrics]
        failed = [n for n in c.to_test if n.status == "failed"]
        if len(failed) >= 3:
            ctl.human.notify("repeated_failures", f"{len(failed)} experiments failed at depth {depth}")
        survivors, _ = evaluator.prune(measured)
        if not survivors and measured:
            # Observation -> re-plan: every branch broke a hard constraint with the default fill.
            replanned = ctl._replan(c.planner, evaluator, c.children, c.res.constraints, c.res)
            measured += replanned
            survivors, _ = evaluator.prune(measured)
        if not survivors:
            soft = [n for n in measured if not evaluator.violations(n.metrics)]
            if soft:
                best = max(soft, key=lambda n: n.score)
                best.status, best.reason = "kept", "revived: best feasible node after whole level was pruned"
                ctl.human.notify("level_collapsed", f"all depth-{depth} nodes fell below the score "
                                 f"threshold; continuing with {best.config.key()}")
                survivors = [best]
            else:
                suggestion = ctl._suggest_relaxation(c.children, c.res.constraints)
                c.res.stopped_reason = f"infeasible constraints ({suggestion})"
                return {"pending": {"kind": "infeasible",
                                    "reason": f"no configuration at depth {depth} meets the constraints; "
                                              f"suggestion: {suggestion}",
                                    "options": [f"relax: {suggestion}", "stop"], "default": "stop",
                                    "then": "stop"}}
        # evidence conflict: literature favoured an option that measured clearly worse
        by_opt = {}
        for n in measured:
            by_opt.setdefault(n.partial[dim], []).append(n.metrics["accuracy"])
        if by_opt:
            best_acc = max(max(v) for v in by_opt.values())
            for opt, accs in by_opt.items():
                if ev[dim][opt]["score"] >= 0.65 and best_acc - max(accs) > 0.05:
                    ctl.human.notify("evidence_conflict",
                                     f"{dim}={opt} is supported by {ev[dim][opt]['notes']} "
                                     f"but measured {best_acc - max(accs):.3f} lower accuracy on this dataset")
        survivors.sort(key=evaluator.tie_key, reverse=True)
        c.beam = survivors[: ctl.beam_width]
        for n in measured:
            if n in c.beam:
                n.status = "kept"
            elif n.status == "open":
                n.status, n.reason = "pruned", "outside beam"
        # Evaluator (critic) agent: reviews this level's measured results against the evidence.
        ctl.agents.critique(dim, depth, measured, c.beam, ev, c.res)
        return {"depth": depth + 1}

    # ------------------------------------------------------------ phase 4: validate finalists
    def validate(state: GraphState) -> dict:
        c = ctx()
        c.res.stopped_reason = "search complete"
        finalists = []
        for n in c.beam[:2]:
            v = Node(partial=n.partial, depth=n.depth, config=n.config, evidence=n.evidence)
            ctl._measure(v, "validation", "validations")
            if v.metrics and not c.evaluator.violations(v.metrics):
                v.score = c.evaluator.utility(v.metrics)
                finalists.append(v)
        if not finalists:
            c.res.stopped_reason = "finalists failed validation"
            return {"pending": {"kind": "infeasible",
                                "reason": "finalists violate constraints on the validation split",
                                "options": ["stop"], "default": "stop", "then": "stop"}}
        finalists.sort(key=lambda n: n.score, reverse=True)
        c.finalists = finalists
        if len(finalists) > 1:
            pick, other = finalists
            run_a, run_b = ctl.log.get(pick.run_id), ctl.log.get(other.run_id)
            c.res.confidence = c.evaluator.win_probability(pick.metrics, other.metrics,
                                                           run_a["hits"], run_b["hits"])
            if c.res.confidence < CONFIDENCE_FLOOR:
                cheaper = min(finalists, key=lambda n: n.metrics["tokens"])
                return {"pending": {"kind": "low_confidence",
                                    "reason": f"top two configurations are statistically close "
                                              f"(P(best) = {c.res.confidence:.2f})",
                                    "options": [pick.config.key(), other.config.key()],
                                    "default": cheaper.config.key(), "then": "choose"}}
        else:
            c.res.confidence = 1.0
        return {"decision": None}

    def finalize(state: GraphState) -> dict:
        c = ctx()
        pick = c.finalists[0]
        other = c.finalists[1] if len(c.finalists) > 1 else None
        if other is not None and state.get("decision") == other.config.key():
            pick, other = other, pick
        if other is not None:
            c.res.runner_up = ctl._summary(other)
        c.res.recommendation = ctl._summary(pick)
        ctl.agents.summarize(c.res)
        return {"done": True}

    # ------------------------------------------------------------ human checkpoint
    def human(state: GraphState) -> dict:
        p = state["pending"]
        if ctl.human.mode == "interactive" and p["kind"] not in ctl.human.scripted:
            decision = interrupt({k: p[k] for k in ("kind", "reason", "options", "default")})
            if decision not in p["options"]:
                decision = p["default"]
            ctl.human.log.append(Escalation(p["kind"], p["reason"], p["options"], decision, True))
        else:
            decision = ctl.human.ask(p["kind"], p["reason"], p["options"], default=p["default"])
        update: dict = {"pending": None, "decision": decision}
        if p["then"] == "approval" and decision == "deny":
            ctx().res.stopped_reason = f"human denied the depth-{state['depth']} step"
            update["done"] = True
        elif p["then"] == "stop":
            update["done"] = True
        return update

    # ------------------------------------------------------------ routing
    def after_clarify(state: GraphState) -> str:
        if state.get("pending"):
            return "human"
        return END if state.get("done") else "evidence"

    def after_expand(state: GraphState) -> str:
        return "human" if state.get("pending") else "test"

    def after_evaluate(state: GraphState) -> str:
        if state.get("pending"):
            return "human"
        return "validate" if state["depth"] > len(DIMENSIONS) else "expand"

    def after_validate(state: GraphState) -> str:
        return "human" if state.get("pending") else "finalize"

    def after_human(state: GraphState) -> str:
        if state.get("done"):
            return END
        c = ctx()
        if c.finalists:                 # low-confidence choice made
            return "finalize"
        return "test"                   # high-impact step approved

    g = StateGraph(GraphState)
    for name, fn in [("clarify", clarify), ("evidence", evidence), ("expand", expand), ("test", test),
                     ("evaluate", evaluate), ("validate", validate), ("finalize", finalize),
                     ("human", human)]:
        g.add_node(name, fn)
    g.add_edge(START, "clarify")
    g.add_conditional_edges("clarify", after_clarify, ["human", "evidence", END])
    g.add_edge("evidence", "expand")
    g.add_conditional_edges("expand", after_expand, ["human", "test"])
    g.add_edge("test", "evaluate")
    g.add_conditional_edges("evaluate", after_evaluate, ["human", "validate", "expand"])
    g.add_conditional_edges("validate", after_validate, ["human", "finalize"])
    g.add_edge("finalize", END)
    g.add_conditional_edges("human", after_human, ["test", "finalize", END])
    return g.compile(checkpointer=InMemorySaver())


def run_graph(ctl, request: dict, prompt=None):
    """Invoke the graph; in interactive mode, resolve each interrupt with `prompt(payload)`."""
    ctl._ctx = RunContext(request)
    graph = ctl.graph
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 100}
    out = graph.invoke({"depth": 0, "pending": None, "decision": None, "done": False}, config)
    while out.get("__interrupt__"):
        payload = out["__interrupt__"][0].value
        choice = prompt(payload) if prompt else payload["default"]
        out = graph.invoke(Command(resume=choice), config)
    return ctl._ctx
