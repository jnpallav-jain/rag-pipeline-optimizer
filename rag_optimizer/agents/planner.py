"""Planner agent (Tree of Thought thought generator): expands a partial configuration into
one child per option of the next design choice, and fills undecided choices so each
child can be tested now."""
from __future__ import annotations

from ..guardrails import Constraints
from ..pipeline import DIMENSIONS, OPTIONS, TOKENS_PER_WORD, Config, estimated_min_tokens


class Planner:
    def __init__(self, evidence: dict[str, dict], constraints: Constraints):
        self.evidence = evidence  # dimension -> option -> {"score", "notes"}
        self.c = constraints
        self.overrides: dict = {}  # fills learned from observations (set after a successful re-plan)

    def prior(self, dimension: str):
        """Option the literature favours (first option on ties)."""
        opts = OPTIONS[dimension]
        return max(opts, key=lambda o: (self.evidence[dimension][o]["score"], -opts.index(o)))

    def fill(self, partial: dict) -> Config:
        """Complete a partial configuration for a sample experiment:
        evidence-preferred retriever/re-ranker, and the largest top-k that fits the token cap."""
        full = dict(partial)
        for dim, value in self.overrides.items():
            full.setdefault(dim, value)
        full.setdefault("chunk_size", self.prior("chunk_size"))
        full.setdefault("retriever", self.prior("retriever"))
        full.setdefault("rerank", self.prior("rerank"))
        if "top_k" not in full:
            fits = [k for k in OPTIONS["top_k"]
                    if full["chunk_size"] * k * TOKENS_PER_WORD <= self.c.max_context_tokens]
            full["top_k"] = max(fits) if fits else min(OPTIONS["top_k"])
        return Config(**full)

    def expand(self, partial: dict) -> list[dict]:
        depth = len(partial)
        dim = DIMENSIONS[depth]
        children = []
        for opt in OPTIONS[dim]:
            child = {**partial, dim: opt}
            ev = self.evidence[dim][opt]
            notes = ", ".join(ev["notes"]) or "no retrieved evidence"
            children.append({
                "partial": child,
                "dimension": dim,
                "option": opt,
                "thought": f"Try {dim}={opt} (evidence {ev['score']:.2f}; {notes}).",
                "min_tokens": estimated_min_tokens(child),
            })
        return children

    def node_evidence(self, partial: dict) -> float:
        vals = [self.evidence[d][v]["score"] for d, v in partial.items()]
        return sum(vals) / len(vals) if vals else 0.5
