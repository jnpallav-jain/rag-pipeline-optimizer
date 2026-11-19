# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 1000
- Max p95 retrieval latency: 0.5 ms

## Recommendation (validation split, 1,000 questions)
**`c128-glove-k1-overlap`**: answer recall 0.604 [r0023.accuracy], context tokens 166.0 [r0023.tokens], p95 latency 0.13 [r0023.p95_ms] ms.

Runner-up `c128-glove-k3-overlap`: answer recall 0.706 [r0022.accuracy], context tokens 497.8 [r0022.tokens], p95 latency 0.12 [r0022.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 0.65 (low: a human reviewed this choice).
This recommendation is advisory: an engineer must approve it before production use.

## Evidence used (retrieved from the knowledge base)
- **kb01** The Probabilistic Relevance Framework - BM25 and Beyond. Robertson, S. and Zaragoza, H. (2009). Foundations and Trends in Information Retrieval 3(4).
- **kb03** SQuAD - questions written against the passage. Rajpurkar, P. et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP.
- **kb04** Sentence-BERT - averaged word vectors are a weak sentence embedding. Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
- **kb05** Dense Passage Retrieval needs training data; BM25 stronger on SQuAD. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- **kb06** Reciprocal Rank Fusion. Cormack, G., Clarke, C. and Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
- **kb07** Second-stage re-ranking of retrieved candidates. Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085.
- **kb08** Lost in the Middle - more context is not always better. Liu, N. F. et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL.
- **kb09** Chunk size trade-off (practitioner heuristic, no single source). Design heuristic; no peer-reviewed source. Treat as weak evidence.

## Search tree (plan split, 300 questions)

| depth | configuration tested | score | status | note |
|---|---|---|---|---|
| 1 | `c64-bm25-k5-rrf` (r0002) | 0.576 | pruned | p95 latency 2.42 ms > cap 0.50 |
| 1 | `c128-bm25-k5-rrf` (r0001) | 0.545 | pruned | p95 latency 1.37 ms > cap 0.50 |
| 1 | `c256-bm25-k3-rrf` (r0003) | 0.481 | pruned | p95 latency 1.38 ms > cap 0.50 |
| 1 | `c64-bm25-k5-none` (r0004) | 0.613 | pruned | p95 latency 1.84 ms > cap 0.50 |
| 1 | `c128-bm25-k5-none` (r0005) | 0.573 | pruned | p95 latency 1.47 ms > cap 0.50 |
| 1 | `c256-bm25-k3-none` (r0006) | 0.540 | pruned | p95 latency 1.03 ms > cap 0.50 |
| 1 | `c64-tfidf-k5-none` (r0007) | 0.601 | pruned | p95 latency 1.96 ms > cap 0.50 |
| 1 | `c128-tfidf-k5-none` (r0008) | 0.573 | pruned | p95 latency 1.76 ms > cap 0.50 |
| 1 | `c256-tfidf-k3-none` (r0009) | 0.533 | pruned | p95 latency 1.26 ms > cap 0.50 |
| 1 | `c64-glove-k5-none` (r0010) | 0.560 | pruned | score 0.560 < 0.6 |
| 1 | `c128-glove-k5-none` (r0011) | 0.562 | kept | revived: best feasible node after whole level was pruned |
| 1 | `c256-glove-k3-none` (r0012) | 0.510 | pruned | score 0.510 < 0.6 |
| 2 | `c128-bm25-k5-none` (r0005, cached) | 0.603 | pruned | p95 latency 1.47 ms > cap 0.50 |
| 2 | `c128-tfidf-k5-none` (r0008, cached) | 0.575 | pruned | p95 latency 1.76 ms > cap 0.50 |
| 2 | `c128-glove-k5-none` (r0011, cached) | 0.534 | pruned | score 0.534 < 0.6 |
| 2 | `c128-bm25-k5-overlap` (r0013) | 0.605 | pruned | p95 latency 1.56 ms > cap 0.50 |
| 2 | `c128-tfidf-k5-overlap` (r0014) | 0.575 | pruned | p95 latency 1.61 ms > cap 0.50 |
| 2 | `c128-glove-k5-overlap` (r0015) | 0.595 | kept | revived: best feasible node after whole level was pruned |
| 3 | `c128-glove-k1-overlap` (r0017) | 0.650 | kept |  |
| 3 | `c128-glove-k3-overlap` (r0016) | 0.655 | kept |  |
| 3 | `c128-glove-k5-overlap` (r0015, cached) | 0.598 | pruned | score 0.598 < 0.6 |
| 4 | `c128-glove-k3-none` (r0020) | 0.543 | pruned | score 0.543 < 0.6 |
| 4 | `c128-glove-k3-rrf` (r0018) | 0.538 | pruned | p95 latency 1.74 ms > cap 0.50 |
| 4 | `c128-glove-k3-overlap` (r0016, cached) | 0.659 | kept |  |
| 4 | `c128-glove-k1-none` (r0021) | 0.508 | pruned | score 0.508 < 0.6 |
| 4 | `c128-glove-k1-rrf` (r0019) | 0.457 | pruned | p95 latency 1.84 ms > cap 0.50 |
| 4 | `c128-glove-k1-overlap` (r0017, cached) | 0.656 | kept |  |

Budget: 21/30 sample experiments, 2/2 validations, 6 reused from memory. Wall time 5.7 s.

## Human checkpoints and notices
- [notice] **replan**: all 3 branches at depth 1 broke a hard constraint; retesting with alternative fills [('bm25', 'none'), ('tfidf', 'none'), ('glove', 'none'), ('bm25', 'overlap'), ('tfidf', 'overlap'), ('glove', 'overlap'), ('tfidf', 'rrf'), ('glove', 'rrf')] -> noted
- [notice] **level_collapsed**: all depth-1 nodes fell below the score threshold; continuing with c128-glove-k5-none -> noted
- [notice] **replan**: all 3 branches at depth 2 broke a hard constraint; retesting with alternative fills [('bm25', 'overlap'), ('bm25', 'rrf')] -> noted
- [notice] **level_collapsed**: all depth-2 nodes fell below the score threshold; continuing with c128-glove-k5-overlap -> noted
- [decision] **low_confidence**: top two configurations are statistically close (P(best) = 0.65) -> c128-glove-k1-overlap

Source verification: 6 cited numbers, 0 mismatches.
