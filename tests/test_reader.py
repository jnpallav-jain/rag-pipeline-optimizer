"""Reader tests with a fake client: no API key, no network, no cost."""
from types import SimpleNamespace as NS

import pytest

from rag_optimizer import reader as rd
from rag_optimizer.reader import BudgetExceeded, Ledger, Reader, build_prompt, exact_match, f1


def _msg(text, tin=1000, tout=5):
    return NS(content=[NS(type="text", text=text)], stop_reason="end_turn",
              usage=NS(input_tokens=tin, output_tokens=tout))


class FakeMessages:
    def __init__(self):
        self.calls = 0
        self.batches = FakeBatches()

    def create(self, **params):
        assert params["model"] == "claude-haiku-5-5" and params["max_tokens"] == rd.MAX_TOKENS
        self.calls += 1
        return _msg("Denver Broncos")


class FakeBatches:
    def __init__(self):
        self.requests = []

    def create(self, requests):
        self.requests = requests
        return NS(id="msgbatch_test")

    def retrieve(self, _id):
        return NS(id=_id, processing_status="ended", request_counts=NS(processing=0))

    def results(self, _id):
        for r in self.requests:
            yield NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=_msg("1998")))


def make_reader(tmp_path):
    client = NS(messages=FakeMessages())
    return Reader(client=client, cache_path=tmp_path / "cache.jsonl",
                  ledger=Ledger(tmp_path / "spend.json")), client


def test_squad_metrics():
    assert exact_match("the Denver Broncos", ["Denver Broncos"]) == 1.0
    assert f1("Denver", ["Denver Broncos"]) == pytest.approx(2 / 3)
    assert f1("not found", ["Denver Broncos"]) == 0.0


def test_cache_prevents_repeat_calls_and_ledger_counts_tokens(tmp_path):
    r, client = make_reader(tmp_path)
    p = build_prompt("Who won Super Bowl 50?", ["The Denver Broncos won."])
    a = r.answer(p)
    b = r.answer(p)
    assert a["answer"] == "Denver Broncos" and not a["cached"] and b["cached"]
    assert client.messages.calls == 1
    led = r.ledger.load()
    assert led["calls"] == 1 and led["total_usd"] == pytest.approx(1000 * 0.10e-6 + 5 * 0.50e-6)
    # a fresh Reader reads the cache from disk
    r2 = Reader(client=client, cache_path=tmp_path / "cache.jsonl", ledger=Ledger(tmp_path / "spend.json"))
    assert r2.answer(p)["cached"] and client.messages.calls == 1


def test_budget_cap_refuses_before_sending(tmp_path):
    r, client = make_reader(tmp_path)
    r.ledger.add(0, 0, 0, rd.STOP_USD)
    with pytest.raises(BudgetExceeded):
        r.answer(build_prompt("q", ["c"]))
    assert client.messages.calls == 0


def test_worst_case_blocks_a_batch_that_could_pass_the_cap(tmp_path):
    r, client = make_reader(tmp_path)
    r.ledger.add(0, 0, 0, 0.85)
    huge = {str(i): build_prompt(f"q{i}", ["word " * 2000]) for i in range(5000)}
    with pytest.raises(BudgetExceeded):
        r.answer_batch(huge, poll_s=0)
    assert client.messages.batches.requests == []


def test_batch_path_bills_half_price_and_dedupes(tmp_path):
    r, client = make_reader(tmp_path)
    p = build_prompt("When?", ["In 1998."])
    out = r.answer_batch({"a": p, "b": p, "c": build_prompt("Other?", ["x"])}, poll_s=0, log=lambda *_: None)
    assert len(client.messages.batches.requests) == 2          # identical prompts sent once
    assert out["a"]["answer"] == out["b"]["answer"] == "1998"
    assert r.ledger.load()["total_usd"] == pytest.approx(2 * (1000 * 0.10e-6 + 5 * 0.50e-6) * 0.5)
