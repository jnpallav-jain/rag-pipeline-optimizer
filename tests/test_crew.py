"""CrewAI agent tests with a fake LLM: no API key, no network, no cost. They check that agent
outputs are validated, applied only where allowed, and that bad outputs fall back to rules."""
import json

from crewai.llms.base_llm import BaseLLM

from rag_optimizer.crew import CrewAgents, strip_uncited_numbers
from rag_optimizer.guardrails import Budget
from rag_optimizer.human import HumanGate
from rag_optimizer.memory import ExperimentLog, KnowledgeBase
from rag_optimizer.orchestrator import Orchestrator
from rag_optimizer.reader import Ledger
from tests.test_graph import FakeLab


_STATE = {"replies": {}, "last": {}}   # module-level: BaseLLM is a pydantic model


class FakeLLM(BaseLLM):
    """Answers each task type with canned JSON in CrewAI's ReAct 'Final Answer' format."""

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        text = json.dumps(messages) if not isinstance(messages, str) else messages
        reply = next((r for m, r in _STATE["replies"].items() if m in text), None)
        if reply is not None:
            _STATE["last"] = reply
        elif response_model is not None or "json" in text.lower():
            reply = _STATE["last"]       # CrewAI's follow-up call converting the answer to the schema
        else:
            reply = {}
        if response_model is not None:
            return json.dumps(reply)
        return f"Thought: I have what I need.\nFinal Answer: {json.dumps(reply)}"

    def supports_function_calling(self) -> bool:
        return False


GOOD = {
    "preparing evidence": {"items": [
        {"dimension": "retriever", "option": "bm25", "stance": 1, "strength": 0.8, "note_ids": ["kb05"]},
        {"dimension": "retriever", "option": "bge", "stance": 1, "strength": 0.9, "note_ids": ["kb_fake"]},
        {"dimension": "chunk_size", "option": "999", "stance": 1, "strength": 1.0, "note_ids": ["kb09"]}]},
    "Tree of Thought, depth": {"branches": [
        {"index": i, "thought": "Worth testing (kb05).", "fill": {"rerank": "overlap", "top_k": "99"}}
        for i in range(9)]},
    "You are the critic": {"assessment": "BM25 held up. It was 12% better.", "conflicts": [
        {"option": "bge", "note_ids": ["kb11"], "explanation": "Literature favours it, measured worse."}]},
    "Write a 3-5 sentence summary": {"summary": "Pick it. Recall is 0.999 here."},
}


def make(tmp_path, replies):
    _STATE["replies"], _STATE["last"] = replies, {}
    agents = CrewAgents(log_path=tmp_path / "log.jsonl", ledger=Ledger(tmp_path / "spend.json"),
                        llm_factory=lambda: FakeLLM(model="fake"), use_mcp=False)
    orch = Orchestrator(FakeLab(), ExperimentLog(tmp_path / "log.jsonl"), KnowledgeBase(), HumanGate(),
                        budget=Budget(), agents=agents)
    return orch, agents


def test_llm_agents_are_validated_and_applied(tmp_path):
    orch, agents = make(tmp_path, GOOD)
    res = orch.run(1000, 5.0)
    assert res.agents == "llm" and res.recommendation
    ev = res.evidence["retriever"]
    assert "kb05" in ev["bm25"]["notes"] and ev["bm25"]["score"] > 0.5
    assert ev["bge"]["score"] == 0.5                     # uncited (unknown note ID) item ignored
    assert all(v["score"] == 0.5 for v in res.evidence["chunk_size"].values())  # invalid option ignored
    kept = [n for n in res.tree if n.config and n.depth == 1]
    assert all(n.thought.startswith("[Planner agent]") for n in kept)
    assert any(n.config.rerank == "overlap" for n in kept)  # valid fill applied
    assert all(n.config.top_k in (1, 3, 5) for n in res.tree if n.config)  # invalid fill ignored
    texts = [n["text"] for n in res.agent_notes]
    assert "BM25 held up." in texts                       # sentence with an uncited number removed
    assert any(t.startswith("retriever=bge (kb11)") for t in texts)
    assert res.summary == "Pick it."                     # uncited number removed from summary
    assert not agents.fallbacks


def test_bad_outputs_fall_back_to_rules(tmp_path):
    orch, agents = make(tmp_path, {})                    # every reply is an empty object
    res = orch.run(1000, 5.0)
    assert res.recommendation is not None
    assert agents.fallbacks and all("rule-based agent used" in f for f in agents.fallbacks)
    assert any(n["agent"] == "fallback" for n in res.agent_notes)


def test_budget_cap_forces_fallback_without_calls(tmp_path):
    orch, agents = make(tmp_path, GOOD)
    agents.ledger.add(0, 0, 0, 0.95)
    res = orch.run(1000, 5.0)
    assert res.recommendation is not None and agents.calls == 0
    assert all("BudgetExceeded" in f for f in agents.fallbacks)


def test_strip_uncited_numbers():
    text, dropped = strip_uncited_numbers(
        "Use c128-bm25-k3-none: recall 0.885 [r0012.accuracy] (kb05). It is 12% faster. BM25 wins on p95.")
    assert dropped == 1 and "12%" not in text and "0.885 [r0012.accuracy]" in text
