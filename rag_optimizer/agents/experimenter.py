"""Experimenter agent: the only role allowed to execute pipelines. Runs are time-limited,
retried once on failure, and always written to the experiment log."""
from __future__ import annotations

import time

from ..guardrails import time_limit
from ..memory import ExperimentLog
from ..pipeline import Config, Lab


class Experimenter:
    def __init__(self, lab: Lab, log: ExperimentLog, timeout_s: float = 120, retries: int = 1,
                 fault_injector=None):
        self.lab, self.log = lab, log
        self.timeout_s, self.retries = timeout_s, retries
        self.fault_injector = fault_injector  # tests only: callable(cfg, attempt) that may raise
        self.failures = 0
        self.recovered = 0

    def run(self, cfg: Config, split: str) -> dict:
        questions = self.lab.dataset.splits[split]
        errors = []
        for attempt in range(self.retries + 1):
            try:
                t0 = time.perf_counter()
                with time_limit(self.timeout_s):
                    if self.fault_injector:
                        self.fault_injector(cfg, attempt)
                    metrics = self.lab.run(cfg, questions)
                if attempt > 0:
                    self.recovered += 1
                hits = metrics.pop("hits")
                return self.log.add({"config": cfg.key(), "params": cfg.as_dict(),
                                     "dataset": self.lab.dataset.name, "split": split,
                                     "status": "ok", "metrics": metrics, "hits": hits,
                                     "seconds": round(time.perf_counter() - t0, 3),
                                     "attempts": attempt + 1})
            except Exception as exc:  # noqa: BLE001 - any failure is recorded, never hidden
                errors.append(f"{type(exc).__name__}: {exc}")
        self.failures += 1
        return self.log.add({"config": cfg.key(), "params": cfg.as_dict(),
                             "dataset": self.lab.dataset.name, "split": split,
                             "status": "failed", "errors": errors, "metrics": None})
