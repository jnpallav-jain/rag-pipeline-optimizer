"""The RAG retrieval pipeline under test: chunking, retrievers, re-ranking, metrics.

A configuration has four design choices (the agent's search space, 3 x 3 x 3 x 3 = 81):
  chunk_size  words per chunk (25% overlap)       64 | 128 | 256
  retriever   first-stage retriever              bm25 | tfidf | bge
  top_k       chunks passed to the generator      1 | 3 | 5
  rerank      second stage over top-20            none | rrf | overlap

Measured per configuration:
  accuracy    answer recall: share of questions whose gold answer string appears in the
              retrieved context (SQuAD normalization). Generation is out of scope; this
              is the upper bound on what a reader model could answer from the context.
  p95_ms      95th-percentile retrieval latency per query on the current machine (for bge it
              includes embedding the query on the CPU)
  tokens      mean context tokens sent to the generator (words x 1.3 estimate)
"""
from __future__ import annotations

import hashlib
import re
import string
import time
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .data import ROOT, Dataset, Question

OPTIONS = {
    "chunk_size": [64, 128, 256],
    "retriever": ["bm25", "tfidf", "bge"],
    "top_k": [1, 3, 5],
    "rerank": ["none", "rrf", "overlap"],
}
DIMENSIONS = list(OPTIONS)
TOKENS_PER_WORD = 1.3
CANDIDATES = 20  # first-stage depth for re-ranking
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d sentence encoder, run on CPU
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE v1.5 query instruction
INDEX_DIR = ROOT / "data" / "index"

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

    def top(self, query: str, n: int):
        return _top_from_scores(self.scores(query), n)


class TFIDF:
    def __init__(self, texts: list[str]):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   sublinear_tf=True)
        self.matrix = self.vec.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray((self.matrix @ self.vec.transform([query]).T).todense()).ravel()

    def top(self, query: str, n: int):
        return _top_from_scores(self.scores(query), n)


def _top_from_scores(scores: np.ndarray, n: int):
    """Ids of the n highest scores (best first) and their scores."""
    n = min(n, len(scores))
    ids = np.argpartition(-scores, n - 1)[:n]
    ids = ids[np.argsort(-scores[ids])]
    return ids, scores[ids]


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer  # heavy import, only when bge is used
    return SentenceTransformer(EMBED_MODEL, device="cpu")


class BGEDense:
    """Dense retriever: BGE sentence embeddings in a FAISS exact inner-product index (cosine,
    since vectors are normalized). Chunk embeddings are computed once and saved to data/index/.

    Query embeddings are cached per question so repeated experiments stay fast, but the time it
    took to embed the question is still charged to every query's latency (``extra_ms``)."""

    def __init__(self, texts: list[str], chunk_size: int):
        import faiss
        self.model = _encoder()
        digest = hashlib.sha1("\x1e".join(texts).encode()).hexdigest()[:12]
        path = INDEX_DIR / f"bge-small-c{chunk_size}-{digest}.faiss"
        if path.exists():
            self.index = faiss.read_index(str(path))
        else:
            vecs = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(path))
        self._queries: dict[str, tuple[np.ndarray, float]] = {}
        self.extra_ms = 0.0

    def embed_query(self, query: str) -> tuple[np.ndarray, float]:
        """Returns (vector, extra_ms). extra_ms is the stored embedding time on a cache hit and
        0 on a miss, because a miss is already inside the caller's timer."""
        if query in self._queries:
            vec, ms = self._queries[query]
            return vec, ms
        t0 = time.perf_counter()
        vec = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        self._queries[query] = (vec, (time.perf_counter() - t0) * 1000)
        return vec, 0.0

    def top(self, query: str, n: int):
        vec, extra_ms = self.embed_query(query)
        scores, ids = self.index.search(vec, n)
        self.extra_ms += extra_ms
        return ids[0], scores[0]


# ---------------------------------------------------------------- index + run
class Index:
    """All retrievers for one chunk size (built once, cached)."""

    def __init__(self, dataset: Dataset, chunk_size: int):
        self.texts, self.titles = chunk_documents(dataset.documents, chunk_size)
        self.norm_texts = [normalize_answer(t) for t in self.texts]
        self.token_sets = [set(tokenize(t)) for t in self.texts]
        self.word_counts = np.array([len(t.split()) for t in self.texts])
        self.retrievers = {"bm25": BM25(self.texts), "tfidf": TFIDF(self.texts),
                           "bge": BGEDense(self.texts, chunk_size)}

    def retrieve(self, query: str, cfg: Config) -> tuple[list[int], float]:
        """Returns (chunk ids, extra_ms): extra_ms is cached query-embedding time to add to the
        measured latency."""
        dense = self.retrievers["bge"]
        dense.extra_ms = 0.0
        cand, primary = self.retrievers[cfg.retriever].top(query, CANDIDATES)
        if cfg.rerank == "rrf":
            # Reciprocal rank fusion with a complementary retriever (lexical <-> dense)
            other = "bge" if cfg.retriever != "bge" else "bm25"
            cand2, _ = self.retrievers[other].top(query, CANDIDATES)
            fused: dict[int, float] = {}
            for ranking in (cand, cand2):
                for r, idx in enumerate(ranking):
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + r + 1)
            order = sorted(fused, key=lambda i: -fused[i])
        elif cfg.rerank == "overlap":
            # Lexical cross-check: share of query content words present in the chunk
            q = {w for w in tokenize(query) if w not in STOPWORDS}
            score = {int(i): float(p) for i, p in zip(cand, primary)}
            order = sorted(score, key=lambda i: (-(len(q & self.token_sets[i]) / max(1, len(q))),
                                                 -score[i]))
        else:
            order = [int(i) for i in cand]
        return order[:cfg.top_k], dense.extra_ms


class Lab:
    """Builds indexes lazily and runs configurations on question sets."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._indexes: dict[int, Index] = {}

    def index(self, chunk_size: int) -> Index:
        if chunk_size not in self._indexes:
            self._indexes[chunk_size] = Index(self.dataset, chunk_size)
        return self._indexes[chunk_size]

    def run(self, cfg: Config, questions: list[Question]) -> dict:
        idx = self.index(cfg.chunk_size)
        hits, lat, words = [], [], []
        for q in questions:
            t0 = time.perf_counter()
            ids, extra_ms = idx.retrieve(q.text, cfg)
            lat.append((time.perf_counter() - t0) * 1000 + extra_ms)
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
