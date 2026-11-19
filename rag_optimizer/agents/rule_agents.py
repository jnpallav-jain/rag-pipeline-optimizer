"""Rule-based agent team: the free, deterministic baseline (v2 behaviour).

The LangGraph controller calls the same four hooks on either team:
  research(researcher, constraints, res) -> evidence per design choice
  plan(planner, children, dim, depth, constraints, res) -> may rewrite thoughts / fills
  critique(dim, depth, measured, beam, evidence, res) -> may add notices
  summarize(res) -> may add a written summary to the result
"""
from __future__ import annotations

from ..pipeline import DIMENSIONS


class RuleAgents:
    name = "rule"

    def research(self, researcher, constraints, res) -> dict:
        return {d: researcher.evidence(d) for d in DIMENSIONS}

    def plan(self, planner, children, dim, depth, constraints, res) -> None:
        return None

    def critique(self, dim, depth, measured, beam, evidence, res) -> None:
        return None

    def summarize(self, res) -> None:
        return None
