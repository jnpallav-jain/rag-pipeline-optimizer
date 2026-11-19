# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 1000
- Max p95 retrieval latency: 0.5 ms

## Recommendation (validation split, 1,000 questions)
**`c256-bm25-k1-none`**: answer recall 0.809 [r0015.accuracy], context tokens 330.2 [r0015.tokens], p95 latency 0.42 [r0015.p95_ms] ms.

Runner-up `c256-bm25-k1-overlap`: answer recall 0.787 [r0016.accuracy], context tokens 330.9 [r0016.tokens], p95 latency 0.42 [r0016.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 1.00.
This recommendation is advisory: an engineer must approve it before production use.

## Evidence used (retrieved from the knowledge base)
- **kb03** SQuAD - questions written against the passage. Rajpurkar, P. et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP.
- **kb04** Sentence-BERT - trained sentence encoders beat averaged word vectors. Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
- **kb05** Dense Passage Retrieval needs training data; BM25 stronger on SQuAD. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- **kb06** Reciprocal Rank Fusion. Cormack, G., Clarke, C. and Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
- **kb07** Second-stage re-ranking of retrieved candidates. Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085.
- **kb08** Lost in the Middle - more context is not always better. Liu, N. F. et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL.
- **kb09** Chunk size trade-off (practitioner heuristic, no single source). Design heuristic; no peer-reviewed source. Treat as weak evidence.
- **kb11** BGE - general-purpose embedding models for dense retrieval. Xiao, S. et al. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE models). arXiv:2309.07597.

## Search tree (plan split, 300 questions)

| depth | configuration tested | score | status | note |
|---|---|---|---|---|
| 1 | `c64-bm25-k5-rrf` (r0002) | 0.623 | pruned | p95 latency 10.53 ms > cap 0.50 |
| 1 | `c128-bm25-k5-rrf` (r0001) | 0.592 | pruned | p95 latency 8.92 ms > cap 0.50 |
| 1 | `c256-bm25-k3-rrf` (r0003) | 0.536 | pruned | p95 latency 8.86 ms > cap 0.50 |
| 1 | `c64-bm25-k5-none` (r0004) | 0.611 | pruned | p95 latency 0.70 ms > cap 0.50 |
| 1 | `c128-bm25-k5-none` (r0005) | 0.573 | pruned | p95 latency 0.55 ms > cap 0.50 |
| 1 | `c256-bm25-k3-none` (r0006) | 0.569 | kept | revived: best feasible node after whole level was pruned |
| 2 | `c256-bm25-k3-none` (r0006, cached) | 0.593 | pruned | score 0.593 < 0.6 |
| 2 | `c256-tfidf-k3-none` (r0008) | 0.578 | pruned | score 0.578 < 0.6 |
| 2 | `c256-bge-k3-none` (r0007) | 0.492 | pruned | p95 latency 8.47 ms > cap 0.50 |
| 2 | `c256-bm25-k3-overlap` (r0009) | 0.596 | kept | revived: best feasible node after whole level was pruned |
| 2 | `c256-tfidf-k3-overlap` (r0010) | 0.579 | pruned | score 0.579 < 0.6 |
| 2 | `c256-bge-k3-overlap` (r0011) | 0.556 | pruned | p95 latency 8.49 ms > cap 0.50 |
| 3 | `c256-bm25-k1-overlap` (r0012) | 0.613 | kept |  |
| 3 | `c256-bm25-k3-overlap` (r0009, cached) | 0.593 | pruned | score 0.593 < 0.6 |
| 3 | `{'chunk_size': 256, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 1664 tokens > cap |
| 4 | `c256-bm25-k1-none` (r0014) | 0.630 | kept |  |
| 4 | `c256-bm25-k1-rrf` (r0013) | 0.562 | pruned | p95 latency 8.86 ms > cap 0.50 |
| 4 | `c256-bm25-k1-overlap` (r0012, cached) | 0.612 | kept |  |

Budget: 14/30 sample experiments, 2/2 validations, 3 reused from memory. Wall time 2.4 s.

## Human checkpoints and notices
- [notice] **replan**: all 3 branches at depth 1 broke a hard constraint; retesting with alternative fills [('bm25', 'none'), ('bge', 'none'), ('tfidf', 'none'), ('bm25', 'overlap'), ('bge', 'overlap'), ('tfidf', 'overlap'), ('bge', 'rrf'), ('tfidf', 'rrf')] -> noted
- [notice] **level_collapsed**: all depth-1 nodes fell below the score threshold; continuing with c256-bm25-k3-none -> noted
- [notice] **replan**: all 3 branches at depth 2 broke a hard constraint; retesting with alternative fills [('bm25', 'overlap'), ('bm25', 'rrf')] -> noted
- [notice] **level_collapsed**: all depth-2 nodes fell below the score threshold; continuing with c256-bm25-k3-overlap -> noted
- [notice] **evidence_conflict**: rerank=rrf is supported by ['kb06'] but measured 0.063 lower accuracy on this dataset -> noted

Source verification: 6 cited numbers, 0 mismatches.
