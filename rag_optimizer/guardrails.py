"""Guardrails: input checks, injection detection, time limits, budget, report verification."""
from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass

DATASET_ALLOWLIST = {"squad-v1.1-dev"}  # public datasets only


class GuardrailError(Exception):
    """Raised when a request or action violates a hard guardrail."""


@dataclass
class Constraints:
    max_context_tokens: float
    max_p95_ms: float
    dataset: str = "squad-v1.1-dev"


def validate_request(max_context_tokens, max_p95_ms, dataset: str) -> tuple[Constraints | None, list[str]]:
    """Input checks. Returns (constraints, problems). Problems that make the goal
    ambiguous are returned for human clarification; policy violations raise."""
    if dataset not in DATASET_ALLOWLIST:
        raise GuardrailError(f"dataset '{dataset}' is not on the public-data allowlist "
                             f"{sorted(DATASET_ALLOWLIST)}")
    problems = []
    for name, value, lo, hi in [("max_context_tokens", max_context_tokens, 10, 20000),
                                ("max_p95_ms", max_p95_ms, 0.01, 10000)]:
        if value is None:
            problems.append(f"{name} is missing")
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            problems.append(f"{name}={value!r} is not a number")
            continue
        if not lo <= v <= hi:
            problems.append(f"{name}={v} is outside the sane range [{lo}, {hi}]")
    if problems:
        return None, problems
    return Constraints(float(max_context_tokens), float(max_p95_ms), dataset), []


_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (the |your )?(rules|instructions|guardrails)",
    r"\byou (must|should) (always )?recommend\b",
    r"\bsystem prompt\b",
    r"\bas an ai\b",
    r"skip (the )?(validation|verification|human review)",
]


def detect_injection(text: str) -> str | None:
    """Retrieved text is data, never instructions. Flag text that tries to instruct the agent."""
    low = text.lower()
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, low):
            return f"matched injection pattern /{pat}/"
    return None


class TimeLimitExceeded(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Abort an experiment that runs too long (Unix only; no-op elsewhere)."""
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        yield
        return

    def handler(signum, frame):
        raise TimeLimitExceeded(f"experiment exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@dataclass
class Budget:
    max_evaluations: int = 30          # sample experiments during search
    max_validations: int = 2           # full-validation runs for finalists
    used_evaluations: int = 0
    used_validations: int = 0
    cache_hits: int = 0

    @property
    def remaining(self) -> int:
        return self.max_evaluations - self.used_evaluations

    def needs_approval(self, planned: int) -> bool:
        """High-impact step: one step would spend more than half of the total budget."""
        return planned > 0.5 * self.max_evaluations


_CITED_NUMBER = re.compile(r"(\d+(?:\.\d+)?) \[(r\d{4})\.(accuracy|p95_ms|tokens)\]")


def verify_report(markdown: str, log) -> dict:
    """Source verification: every cited number must match the logged run it cites."""
    checked, mismatches = 0, []
    for value, run_id, metric in _CITED_NUMBER.findall(markdown):
        checked += 1
        run = log.get(run_id)
        if run is None:
            mismatches.append(f"{run_id} not in experiment log")
            continue
        logged = run["metrics"][metric]
        decimals = len(value.split(".")[1]) if "." in value else 0
        if round(logged, decimals) != float(value):
            mismatches.append(f"{value} != {run_id}.{metric}={logged}")
    return {"cited_numbers": checked, "mismatches": mismatches, "grounded": not mismatches}
