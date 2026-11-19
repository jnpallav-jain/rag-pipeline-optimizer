"""LLM reader: answers a question from retrieved chunks with Claude Haiku 5.5, and scores the
answer with the official SQuAD exact-match and F1 metrics.

Cost controls (the whole project has a hard $1 budget):
- every answer is cached on disk (keyed by model + prompt hash); a cached prompt is never re-sent
- a spend ledger adds up the token counts the API reports, at list prices, across all runs
- a call or batch is refused if its worst-case cost would take the ledger past the cap, and
  nothing new is sent once the ledger reaches the stop threshold
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data import ROOT
from .pipeline import normalize_answer

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6   # $ per token, Haiku 5.5 list price (prompts < 100K)
BATCH_DISCOUNT = 0.5                            # Message Batches API bills 50% of standard
MAX_TOKENS = 64                                 # answers are short spans
CAP_USD = 1.00                                  # hard project budget
STOP_USD = 0.90                                 # stop sending new requests here (headroom)
CACHE_PATH = ROOT / "results" / "llm_cache.jsonl"
LEDGER_PATH = ROOT / "results" / "llm_spend.json"

SYSTEM = ("You answer questions using only the provided context passages. Reply with the "
          "shortest phrase from the context that answers the question, with no explanation. "
          "If the context does not contain the answer, reply exactly: not found")


class BudgetExceeded(RuntimeError):
    pass


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) so the key never has to live in code."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_prompt(question: str, chunks: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return f"Context:\n{ctx}\n\nQuestion: {question}"


# ---------------------------------------------------------------- SQuAD metrics
def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    def one(p: str, g: str) -> float:
        pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
        common = sum((Counter(pt) & Counter(gt)).values())
        if not pt or not gt:
            return float(pt == gt)
        if common == 0:
            return 0.0
        prec, rec = common / len(pt), common / len(gt)
        return 2 * prec * rec / (prec + rec)
    return max(one(pred, g) for g in golds)


# ---------------------------------------------------------------- ledger + cache
@dataclass
class Ledger:
    path: Path = LEDGER_PATH

    def load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"total_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}

    def add(self, n_calls: int, tin: int, tout: int, usd: float) -> dict:
        d = self.load()
        d["total_usd"] = round(d["total_usd"] + usd, 9)
        d["calls"] += n_calls
        d["input_tokens"] += tin
        d["output_tokens"] += tout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=2))
        return d

    def check(self, worst_case_usd: float) -> None:
        spent = self.load()["total_usd"]
        if spent >= STOP_USD:
            raise BudgetExceeded(f"spend ${spent:.4f} reached the ${STOP_USD:.2f} stop threshold")
        if spent + worst_case_usd > CAP_USD:
            raise BudgetExceeded(f"spent ${spent:.4f} + worst case ${worst_case_usd:.4f} "
                                 f"would pass the ${CAP_USD:.2f} cap")


def cost(tin: int, tout: int, batch: bool) -> float:
    usd = tin * PRICE_IN + tout * PRICE_OUT
    return usd * (BATCH_DISCOUNT if batch else 1.0)


def worst_case(prompts: list[str], batch: bool) -> float:
    """Upper bound before sending: ~1 token per 3 characters (generous) plus full max_tokens."""
    tin = sum((len(SYSTEM) + len(p)) // 3 + 20 for p in prompts)
    return cost(tin, MAX_TOKENS * len(prompts), batch)


class Reader:
    def __init__(self, client=None, model: str = MODEL, cache_path: Path = CACHE_PATH,
                 ledger: Ledger | None = None):
        self._client = client
        self.model = model
        self.cache_path = cache_path
        self.ledger = ledger or Ledger()
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec

    @property
    def client(self):
        if self._client is None:
            import anthropic
            load_dotenv()
            self._client = anthropic.Anthropic()
        return self._client

    def key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{SYSTEM}\n{MAX_TOKENS}\n{prompt}".encode()).hexdigest()

    def _params(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": MAX_TOKENS, "system": SYSTEM,
                "thinking": {"type": "disabled"},  # allowed on Haiku 5.5 at default (medium) effort
                "messages": [{"role": "user", "content": prompt}]}

    def _store(self, rec: dict) -> dict:
        self.cache[rec["key"]] = rec
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    @staticmethod
    def _text(msg) -> tuple[str, str]:
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return re.sub(r"\s+", " ", text), msg.stop_reason

    # -------- one call (standard price); used for the pilot and the CLI demo
    def answer(self, prompt: str, tag: str = "") -> dict:
        k = self.key(prompt)
        if k in self.cache:
            return {**self.cache[k], "cached": True}
        self.ledger.check(worst_case([prompt], batch=False))
        t0 = time.perf_counter()
        msg = self.client.messages.create(**self._params(prompt))
        latency_ms = (time.perf_counter() - t0) * 1000
        text, stop = self._text(msg)
        tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
        usd = cost(tin, tout, batch=False)
        self.ledger.add(1, tin, tout, usd)
        return {**self._store({"key": k, "tag": tag, "model": self.model, "mode": "standard",
                               "answer": text, "stop_reason": stop, "input_tokens": tin,
                               "output_tokens": tout, "cost_usd": round(usd, 8),
                               "latency_ms": round(latency_ms, 1)}), "cached": False}

    # -------- many calls through the Message Batches API (50% price)
    def answer_batch(self, prompts: dict[str, str], poll_s: float = 20.0, log=print) -> dict[str, dict]:
        """prompts: {custom_id: prompt}. Returns {custom_id: record}. Cached prompts aren't sent."""
        out, todo = {}, {}
        for cid, p in prompts.items():
            k = self.key(p)
            if k in self.cache:
                out[cid] = {**self.cache[k], "cached": True}
            else:
                todo.setdefault(k, (cid, p))
        if not todo:
            return out
        self.ledger.check(worst_case([p for _, p in todo.values()], batch=True))
        requests = [{"custom_id": k[:64], "params": self._params(p)} for k, (_, p) in todo.items()]
        batch = self.client.messages.batches.create(requests=requests)
        log(f"batch {batch.id}: {len(requests)} requests submitted")
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            log(f"  {batch.processing_status}: {batch.request_counts.processing} processing")
            time.sleep(poll_s)
        by_short = {k[:64]: k for k in todo}
        n = tin_all = tout_all = 0
        usd_all = 0.0
        for res in self.client.messages.batches.results(batch.id):
            k = by_short[res.custom_id]
            if res.result.type != "succeeded":
                log(f"  {res.custom_id}: {res.result.type}")
                continue
            msg = res.result.message
            text, stop = self._text(msg)
            tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
            usd = cost(tin, tout, batch=True)
            n, tin_all, tout_all, usd_all = n + 1, tin_all + tin, tout_all + tout, usd_all + usd
            self._store({"key": k, "tag": todo[k][0], "model": self.model, "mode": "batch",
                         "batch_id": batch.id, "answer": text, "stop_reason": stop,
                         "input_tokens": tin, "output_tokens": tout, "cost_usd": round(usd, 8)})
        self.ledger.add(n, tin_all, tout_all, usd_all)
        for cid, p in prompts.items():
            k = self.key(p)
            if cid not in out and k in self.cache:
                out[cid] = {**self.cache[k], "cached": False}
        return out
