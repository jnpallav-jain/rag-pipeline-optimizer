"""Load SQuAD v1.1 dev as a retrieval benchmark.

Each Wikipedia article becomes one document (its paragraphs joined), so chunk size
genuinely changes what a retrieved unit contains. Questions are split into three
disjoint sets with a fixed seed:

- plan (300):        used by the agent for cheap sample experiments during search
- validation (1000): used by the agent to validate its two finalists
- test (2000):       never seen by the agent; used only for evaluation
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SQUAD_PATH = RAW / "squad-dev-v1.1.json"

SPLIT_SIZES = {"plan": 300, "validation": 1000, "test": 2000}
SEED = 7


@dataclass
class Question:
    qid: str
    text: str
    answers: list[str]
    article: str


@dataclass
class Dataset:
    name: str
    documents: dict[str, str]                     # article title -> full text
    splits: dict[str, list[Question]] = field(default_factory=dict)


def load_squad(path: Path = SQUAD_PATH) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    raw = json.loads(path.read_text())
    documents: dict[str, str] = {}
    questions: list[Question] = []
    for art in raw["data"]:
        title = art["title"]
        documents[title] = "\n\n".join(p["context"] for p in art["paragraphs"])
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                answers = sorted({a["text"] for a in qa["answers"]})
                questions.append(Question(qa["id"], qa["question"], answers, title))
    rng = random.Random(SEED)
    rng.shuffle(questions)
    splits, start = {}, 0
    for name, size in SPLIT_SIZES.items():
        splits[name] = questions[start:start + size]
        start += size
    return Dataset(name="squad-v1.1-dev", documents=documents, splits=splits)
