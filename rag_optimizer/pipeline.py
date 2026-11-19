"""The RAG retrieval pipeline under test: chunking, retrievers, re-ranking, metrics.

A configuration has four design choices (the agent's search space, 3 x 3 x 3 x 3 = 81):
  chunk_size  words per chunk (25% overlap)       64 | 128 | 256
  retriever   first-stage retriever              bm25 | tfidf | glove
  top_k       chunks passed to the generator      1 | 3 | 5
  rerank      second stage over top-20            none | rrf | overlap

Measured per configuration:
  accuracy    answer recall: share of questions whose gold answer string appears in the
              retrieved context (SQuAD normalization). Generation is out of scope; this
              is the upper bound on what a reader model could answer from the context.
  p95_ms      95th-percentile retrieval latency per query on the current machine
  tokens      mean context tokens sent to the generator (words x 1.3 estimate)
"""
from __future__ import annotations

import gzip
import re
import string
import time
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .data import RAW, Dataset, Question

OPTIONS = {
    "chunk_size": [64, 128, 256],
    "retriever": ["bm25", "tfidf", "glove"],
    "top_k": [1, 3, 5],
    "rerank": ["none", "rrf", "overlap"],
}
DIMENSIONS = list(OPTIONS)
TOKENS_PER_WORD = 1.3
CANDIDATES = 20  # first-stage depth for re-ranking

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "and", "or", "is",
    "was", "were", "are", "be", "been", "what", "which", "who", "whom", "when", "where", "why",
    "how", "did", "do", "does", "that", "this", "it", "its", "as", "into", "than", "then", "there",
}


@dataclass(frozen=True)
class Config:
    chunk_size: int
    retriever: str
    top_k: int
    rerank: str

    def key(self) -> str:
        return f"c{self.chunk_size}-{self.retriever}-k{self.top_k}-{self.rerank}"

    def as_dict(self) -> dict:
        return asdict(self)


def all_configs() -> list[Config]:
    return [Config(c, r, k, rr) for c in OPTIONS["chunk_size"] for r in OPTIONS["retriever"]
            for k in OPTIONS["top_k"] for rr in OPTIONS["rerank"]]


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def normalize_answer(s: str) -> str:
    """SQuAD official normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------- chunking
def chunk_documents(documents: dict[str, str], chunk_size: int, overlap: float = 0.25):
    """Fixed word windows with overlap. Returns (texts, titles)."""
    step = max(1, int(chunk_size * (1 - overlap)))
    texts, titles = [], []
    for title, text in documents.items():
        words = text.split()
        for start in range(0, max(1, len(words) - chunk_size + step), step):
            piece = words[start:start + chunk_size]
            if piece:
                texts.append(" ".join(piece))
                titles.append(title)
    return texts, titles


# ---------------------------------------------------------------- retrievers
class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(texts).tocsc().astype(np.float32)
        n = tf.shape[0]
        df = np.diff(tf.indptr)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        tf = tf.tocoo()
        data = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.matrix = sparse.csr_matrix((data, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        q.data[:] = 1.0
        return np.asarray((self.matrix @ q.T).todense()).ravel()


class TFIDF:
    def __init__(self, texts: list[str]):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   sublinear_tf=True)
        self.matrix = self.vec.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray((self.matrix @ self.vec.transform([query]).T).todense()).ravel()


@lru_cache(maxsize=1)
def _glove_vectors(vocab: frozenset) -> dict[str, np.ndarray]:
    path = RAW / "glove-wiki-gigaword-50.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    vecs = {}
    with gzip.open(path, "rt", encoding="utf8") as f:
        next(f)  # header line: "400000 50"
        for line in f:
            word, rest = line.split(" ", 1)
            if word in vocab:
                vecs[word] = np.asarray(rest.split(), dtype=np.float32)
    return vecs


class GloVeDense:
    """Dense retriever: IDF-weighted average of GloVe word vectors, cosine similarity."""

    def __init__(self, texts: list[str], vocab: frozenset):
        self.wv = _glove_vectors(vocab)
        tok = [tokenize(t) for t in texts]
        n = len(tok)
        df: dict[str, int] = {}
        for ts in tok:
            for w in set(ts):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.matrix = np.vstack([self._embed(ts) for ts in tok])

    def _embed(self, tokens: list[str]) -> np.ndarray:
        acc, wsum = np.zeros(50, dtype=np.float32), 0.0
        for w in tokens:
            if w in self.wv and w not in STOPWORDS:
                weight = self.idf.get(w, 1.0)
                acc += weight * self.wv[w]
                wsum += weight
        if wsum:
            acc /= wsum
        norm = np.linalg.norm(acc)
        return acc / norm if norm else acc

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self._embed(tokenize(query))


# ---------------------------------------------------------------- index + run
class Index:
    """All retrievers for one chunk size (built once, cached)."""

    def __init__(self, dataset: Dataset, chunk_size: int, vocab: frozenset):
        self.texts, self.titles = chunk_documents(dataset.documents, chunk_size)
        self.norm_texts = [normalize_answer(t) for t in self.texts]
        self.token_sets = [set(tokenize(t)) for t in self.texts]
        self.word_counts = np.array([len(t.split()) for t in self.texts])
        self.retrievers = {"bm25": BM25(self.texts), "tfidf": TFIDF(self.texts),
                           "glove": GloVeDense(self.texts, vocab)}

    def retrieve(self, query: str, cfg: Config) -> list[int]:
        primary = self.retrievers[cfg.retriever].scores(query)
        cand = np.argpartition(-primary, CANDIDATES)[:CANDIDATES]
        cand = cand[np.argsort(-primary[cand])]
        if cfg.rerank == "rrf":
            # Reciprocal rank fusion with a complementary retriever (lexical <-> dense)
            other = "glove" if cfg.retriever != "glove" else "bm25"
            sec = self.retrievers[other].scores(query)
            cand2 = np.argpartition(-sec, CANDIDATES)[:CANDIDATES]
            cand2 = cand2[np.argsort(-sec[cand2])]
            fused: dict[int, float] = {}
            for ranking in (cand, cand2):
                for r, idx in enumerate(ranking):
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + r + 1)
            order = sorted(fused, key=lambda i: -fused[i])
        elif cfg.rerank == "overlap":
            # Lexical cross-check: share of query content words present in the chunk
            q = {w for w in tokenize(query) if w not in STOPWORDS}
            order = sorted((int(i) for i in cand),
                           key=lambda i: (-(len(q & self.token_sets[i]) / max(1, len(q))),
                                          -primary[i]))
        else:
            order = [int(i) for i in cand]
        return order[:cfg.top_k]


class Lab:
    """Builds indexes lazily and runs configurations on question sets."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        vocab = set()
        for text in dataset.documents.values():
            vocab.update(tokenize(text))
        for qs in dataset.splits.values():
            for q in qs:
                vocab.update(tokenize(q.text))
        self.vocab = frozenset(vocab)
        self._indexes: dict[int, Index] = {}

    def index(self, chunk_size: int) -> Index:
        if chunk_size not in self._indexes:
            self._indexes[chunk_size] = Index(self.dataset, chunk_size, self.vocab)
        return self._indexes[chunk_size]

    def run(self, cfg: Config, questions: list[Question]) -> dict:
        idx = self.index(cfg.chunk_size)
        hits, lat, words = [], [], []
        for q in questions:
            t0 = time.perf_counter()
            ids = idx.retrieve(q.text, cfg)
            lat.append((time.perf_counter() - t0) * 1000)
            ctx = " ".join(idx.norm_texts[i] for i in ids)
            hits.append(any(normalize_answer(a) in ctx for a in q.answers if normalize_answer(a)))
            words.append(int(idx.word_counts[ids].sum()))
        hits_arr = np.array(hits, dtype=float)
        return {
            "accuracy": float(hits_arr.mean()),
            "p95_ms": float(np.percentile(lat, 95)),
            "tokens": float(np.mean(words) * TOKENS_PER_WORD),
            "n": len(questions),
            "hits": hits_arr.astype(int).tolist(),
        }


def estimated_min_tokens(partial: dict) -> float:
    """Optimistic lower bound on context tokens for a partial config (used to prune
    without running anything): smallest chunk x smallest k still possible."""
    c = partial.get("chunk_size", min(OPTIONS["chunk_size"]))
    k = partial.get("top_k", min(OPTIONS["top_k"]))
    return c * k * TOKENS_PER_WORD
