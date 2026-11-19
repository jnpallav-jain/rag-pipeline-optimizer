"""Researcher agent: retrieves evidence for design options from the knowledge base and
finds earlier results in the experiment log."""
from __future__ import annotations

from ..memory import ExperimentLog, KnowledgeBase
from ..pipeline import OPTIONS

QUERIES = {
    "chunk_size": "chunk size for splitting documents into passages, precision versus context and token cost",
    "retriever": "which first-stage retriever works best for question answering passage retrieval: "
                 "BM25 lexical, TF-IDF, or dense sentence embeddings such as BGE",
    "top_k": "how many retrieved passages to put in the context, long contexts and more documents",
    "rerank": "re-ranking retrieved candidates and reciprocal rank fusion of lexical and dense rankers",
}


class Researcher:
    def __init__(self, kb: KnowledgeBase | None, log: ExperimentLog):
        self.kb, self.log = kb, log

    def evidence(self, dimension: str) -> dict[str, dict]:
        """For each option of a dimension: an evidence score in [0, 1] (0.5 = neutral)
        and the IDs of the notes it rests on. Without a knowledge base, all neutral."""
        out = {opt: {"score": 0.5, "notes": []} for opt in OPTIONS[dimension]}
        if self.kb is None:
            return out
        for note, _sim in self.kb.search(QUERIES[dimension]):
            for dim, option, stance, strength in note.supports:
                key = int(option) if dim in ("chunk_size", "top_k") else option
                if dim == dimension and key in out:
                    out[key]["score"] += 0.25 * stance * strength
                    out[key]["notes"].append(note.id)
        for v in out.values():
            v["score"] = min(1.0, max(0.0, v["score"]))
        return out

    def recall(self, config_key: str, dataset: str, split: str) -> dict | None:
        return self.log.find(config_key, dataset, split)
