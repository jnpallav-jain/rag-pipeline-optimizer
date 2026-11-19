"""Human-in-the-loop gate. Every escalation is recorded; how it is resolved depends on mode:

- interactive: the LangGraph controller pauses with interrupt(); `prompt` asks on the terminal
- auto:        apply the documented default policy (used for batch evaluation) and record it
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Escalation:
    kind: str
    reason: str
    options: list[str]
    decision: str
    blocking: bool


@dataclass
class HumanGate:
    mode: str = "auto"
    log: list[Escalation] = field(default_factory=list)
    scripted: dict[str, str] = field(default_factory=dict)  # kind -> forced decision (tests)

    def ask(self, kind: str, reason: str, options: list[str], default: str,
            blocking: bool = True) -> str:
        if kind in self.scripted:
            decision = self.scripted[kind]
        elif self.mode == "interactive" and blocking:
            print(f"\n[HUMAN REVIEW: {kind}] {reason}")
            for i, o in enumerate(options, 1):
                print(f"  {i}. {o}")
            raw = input(f"Choose 1-{len(options)} (Enter = {default}): ").strip()
            decision = options[int(raw) - 1] if raw.isdigit() and 0 < int(raw) <= len(options) else default
        else:
            decision = default
        self.log.append(Escalation(kind, reason, options, decision, blocking))
        return decision

    def prompt(self, payload: dict) -> str:
        """Answer a LangGraph interrupt on the terminal (interactive mode)."""
        print(f"\n[HUMAN REVIEW: {payload['kind']}] {payload['reason']}")
        options = payload["options"]
        for i, o in enumerate(options, 1):
            print(f"  {i}. {o}")
        raw = input(f"Choose 1-{len(options)} (Enter = {payload['default']}): ").strip()
        return options[int(raw) - 1] if raw.isdigit() and 0 < int(raw) <= len(options) else payload["default"]

    def notify(self, kind: str, reason: str) -> None:
        self.log.append(Escalation(kind, reason, [], "noted", False))
        if self.mode == "interactive":
            print(f"\n[NOTICE: {kind}] {reason}")
