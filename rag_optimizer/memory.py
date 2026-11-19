"""Long-term memory: the research knowledge base (retrieved semantically) and the
experiment log (searched by configuration and dataset metadata).

Short-term memory is the orchestrator's working state (the search tree), kept in
`orchestrator.SearchState`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .guardrails import detect_injection

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"


@dataclass
class Note:
    id: str
    title: str
    source: str
    body: str
    supports: list[tuple[str, str, int, float]] = field(default_factory=list)  # (dim, option, stance, strength)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}"


def parse_note(path: Path) -> Note:
    raw = path.read_text()
    m = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
    if not m:
        raise ValueError(f"{path.name}: missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    supports = []
    for item in filter(None, (s.strip() for s in meta.get("supports", "").split(";"))):
        target, stance, strength = item.rsplit(":", 2)
        dim, option = target.split("=")
        supports.append((dim, option, int(stance), float(strength)))
    return Note(meta["id"], meta["title"], meta["source"], m.group(2).strip(), supports)


class KnowledgeBase:
    """Semantic search over research notes (TF-IDF cosine, top-k with a relevance threshold).

    Notes whose text looks like an instruction to the agent (prompt injection) are
    quarantined at load time and never retrieved."""

    def __init__(self, directory: Path = KB_DIR, extra_paths: list[Path] | None = None,
                 threshold: float = 0.08, top_k: int = 4):
        paths = sorted(directory.glob("*.md")) + list(extra_paths or [])
        self.notes: list[Note] = []
        self.quarantined: list[tuple[str, str]] = []
        for p in paths:
            note = parse_note(p)
            reason = detect_injection(note.text)
            if reason:
                self.quarantined.append((note.id, reason))
            else:
                self.notes.append(note)
        self.threshold, self.top_k = threshold, top_k
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([n.text for n in self.notes])

    def search(self, query: str) -> list[tuple[Note, float]]:
        sims = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        order = np.argsort(-sims)[: self.top_k]
        return [(self.notes[i], float(sims[i])) for i in order if sims[i] >= self.threshold]


class ExperimentLog:
    """Append-only JSONL log of every experiment. Each run gets an ID that reports cite."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.records: list[dict] = []
        if path and path.exists():
            self.records = [json.loads(line) for line in path.read_text().splitlines() if line]

    def next_id(self) -> str:
        return f"r{len(self.records) + 1:04d}"

    def add(self, record: dict) -> dict:
        record = {"run_id": self.next_id(), **record}
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def find(self, config_key: str, dataset: str, split: str) -> dict | None:
        """Metadata-filtered lookup: only a run on the same dataset and split counts."""
        for r in reversed(self.records):
            if (r["config"] == config_key and r["dataset"] == dataset and r["split"] == split
                    and r["status"] == "ok"):
                return r
        return None

    def get(self, run_id: str) -> dict | None:
        return next((r for r in self.records if r["run_id"] == run_id), None)
