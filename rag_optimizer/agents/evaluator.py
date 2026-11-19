"""Evaluator (critic) agent: scores candidates with a fixed rubric, prunes, breaks ties,
and estimates confidence in the final choice."""
from __future__ import annotations

import numpy as np

from ..guardrails import Constraints

WEIGHTS = {"accuracy": 0.50, "latency": 0.20, "cost": 0.15, "evidence": 0.15}
MIN_SCORE = 0.60          # absolute pruning threshold
MAX_GAP = 0.15            # prune if more than 15% below the best node at the same depth
TIE = 0.02                # scores within 0.02 count as a tie
CONFIDENCE_FLOOR = 0.80   # below this, the final choice goes to a human


class Evaluator:
    def __init__(self, constraints: Constraints):
        self.c = constraints

    def violations(self, m: dict) -> list[str]:
        out = []
        if m["tokens"] > self.c.max_context_tokens:
            out.append(f"context {m['tokens']:.0f} tokens > cap {self.c.max_context_tokens:.0f}")
        if m["p95_ms"] > self.c.max_p95_ms:
            out.append(f"p95 latency {m['p95_ms']:.2f} ms > cap {self.c.max_p95_ms:.2f}")
        return out

    def components(self, m: dict) -> dict:
        return {"accuracy": m["accuracy"],
                "latency": float(np.clip(1 - m["p95_ms"] / self.c.max_p95_ms, 0, 1)),
                "cost": float(np.clip(1 - m["tokens"] / self.c.max_context_tokens, 0, 1))}

    def score(self, m: dict, evidence: float) -> float:
        """Search-time rubric: measured performance plus research evidence."""
        comp = self.components(m)
        return sum(WEIGHTS[k] * comp[k] for k in comp) + WEIGHTS["evidence"] * evidence

    def utility(self, m: dict) -> float:
        """Final-selection utility: measured performance only (evidence excluded, so
        results on the user's data override priors from the literature)."""
        comp = self.components(m)
        total = sum(WEIGHTS[k] for k in comp)
        return sum(WEIGHTS[k] * comp[k] for k in comp) / total

    @staticmethod
    def tie_key(node) -> tuple:
        """Ties (within TIE) prefer lower latency, then fewer tokens, then stronger evidence."""
        return (round(node.score / TIE), -node.metrics["p95_ms"], -node.metrics["tokens"], node.evidence)

    def prune(self, nodes: list) -> tuple[list, list]:
        """Returns (survivors, pruned) with a reason set on each pruned node."""
        feasible = []
        for n in nodes:
            v = self.violations(n.metrics)
            if v:
                n.status, n.reason = "pruned", "; ".join(v)
            else:
                feasible.append(n)
        if not feasible:
            return [], nodes
        best = max(n.score for n in feasible)
        survivors = []
        for n in feasible:
            if n.score < MIN_SCORE:
                n.status, n.reason = "pruned", f"score {n.score:.3f} < {MIN_SCORE}"
            elif n.score < best * (1 - MAX_GAP):
                n.status, n.reason = "pruned", f"score {n.score:.3f} more than 15% below best {best:.3f}"
            else:
                survivors.append(n)
        return survivors, [n for n in nodes if n not in survivors]

    def win_probability(self, a: dict, b: dict, hits_a: list, hits_b: list,
                        n_boot: int = 2000, seed: int = 0) -> float:
        """Paired bootstrap over questions: P(utility(a) > utility(b))."""
        rng = np.random.default_rng(seed)
        ha, hb = np.array(hits_a), np.array(hits_b)
        ca, cb = self.components(a), self.components(b)
        w_acc = WEIGHTS["accuracy"]
        rest_a = WEIGHTS["latency"] * ca["latency"] + WEIGHTS["cost"] * ca["cost"]
        rest_b = WEIGHTS["latency"] * cb["latency"] + WEIGHTS["cost"] * cb["cost"]
        idx = rng.integers(0, len(ha), size=(n_boot, len(ha)))
        ua = w_acc * ha[idx].mean(axis=1) + rest_a
        ub = w_acc * hb[idx].mean(axis=1) + rest_b
        return float((ua > ub).mean() + 0.5 * (ua == ub).mean())
