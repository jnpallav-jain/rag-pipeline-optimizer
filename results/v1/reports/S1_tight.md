# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 200
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c128-bm25-k1-none`**: answer recall 0.773 [r0013.accuracy], context tokens 166.0 [r0013.tokens], p95 latency 1.35 [r0013.p95_ms] ms.

Runner-up `c128-bm25-k1-overlap`: answer recall 0.755 [r0014.accuracy], context tokens 166.1 [r0014.tokens], p95 latency 1.51 [r0014.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 1.00.
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
| 1 | `c64-bm25-k1-rrf` (r0002) | 0.541 | pruned | score 0.541 < 0.6 |
| 1 | `c128-bm25-k1-rrf` (r0001) | 0.526 | pruned | score 0.526 < 0.6 |
| 1 | `{'chunk_size': 256}` | - | pruned | heuristic: even the smallest remaining choice needs 333 tokens > cap |
| 1 | `c64-bm25-k1-none` (r0003) | 0.639 | kept |  |
| 1 | `c128-bm25-k1-none` (r0004) | 0.649 | kept |  |
| 2 | `c128-bm25-k1-none` (r0004, cached) | 0.679 | kept |  |
| 2 | `c128-tfidf-k1-none` (r0005) | 0.627 | kept |  |
| 2 | `c128-glove-k1-none` (r0007) | 0.428 | pruned | score 0.428 < 0.6 |
| 2 | `c64-bm25-k1-none` (r0003, cached) | 0.674 | kept |  |
| 2 | `c64-tfidf-k1-none` (r0006) | 0.625 | pruned | outside beam |
| 2 | `c64-glove-k1-none` (r0008) | 0.452 | pruned | score 0.452 < 0.6 |
| 3 | `c128-bm25-k1-none` (r0004, cached) | 0.665 | kept |  |
| 3 | `{'chunk_size': 128, 'retriever': 'bm25', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 499 tokens > cap |
| 3 | `{'chunk_size': 128, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 832 tokens > cap |
| 3 | `c64-bm25-k1-none` (r0003, cached) | 0.663 | kept |  |
| 3 | `{'chunk_size': 64, 'retriever': 'bm25', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 250 tokens > cap |
| 3 | `{'chunk_size': 64, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 416 tokens > cap |
| 3 | `c128-tfidf-k1-none` (r0005, cached) | 0.623 | kept |  |
| 3 | `{'chunk_size': 128, 'retriever': 'tfidf', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 499 tokens > cap |
| 3 | `{'chunk_size': 128, 'retriever': 'tfidf', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 832 tokens > cap |
| 4 | `c128-bm25-k1-none` (r0004, cached) | 0.658 | kept |  |
| 4 | `c128-bm25-k1-rrf` (r0001, cached) | 0.541 | pruned | score 0.541 < 0.6 |
| 4 | `c128-bm25-k1-overlap` (r0009) | 0.659 | kept |  |
| 4 | `c64-bm25-k1-none` (r0003, cached) | 0.657 | kept |  |
| 4 | `c64-bm25-k1-rrf` (r0002, cached) | 0.565 | pruned | score 0.565 < 0.6 |
| 4 | `c64-bm25-k1-overlap` (r0010) | 0.645 | pruned | outside beam |
| 4 | `c128-tfidf-k1-none` (r0005, cached) | 0.620 | pruned | outside beam |
| 4 | `c128-tfidf-k1-rrf` (r0011) | 0.516 | pruned | score 0.516 < 0.6 |
| 4 | `c128-tfidf-k1-overlap` (r0012) | 0.641 | pruned | outside beam |

Budget: 12/30 sample experiments, 2/2 validations, 10 reused from memory. Wall time 6.6 s.

## Human checkpoints and notices
- [notice] **replan**: all 2 branches at depth 1 broke a hard constraint; retesting with alternative fills [('bm25', 'none'), ('tfidf', 'none'), ('glove', 'none'), ('bm25', 'overlap'), ('tfidf', 'overlap'), ('glove', 'overlap'), ('tfidf', 'rrf'), ('glove', 'rrf')] -> noted
- [notice] **evidence_conflict**: rerank=rrf is supported by ['kb06'] but measured 0.233 lower accuracy on this dataset -> noted

Source verification: 6 cited numbers, 0 mismatches.
