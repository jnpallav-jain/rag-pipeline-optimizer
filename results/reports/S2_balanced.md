# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 400
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c64-bm25-k1-none`**: answer recall 0.689 [r0012.accuracy], context tokens 83.0 [r0012.tokens], p95 latency 1.78 [r0012.p95_ms] ms.

Runner-up `c64-bm25-k3-overlap`: answer recall 0.845 [r0011.accuracy], context tokens 249.3 [r0011.tokens], p95 latency 2.21 [r0011.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 0.59 (low: a human reviewed this choice).
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
| 1 | `c64-bm25-k3-rrf` (r0002) | 0.636 | kept |  |
| 1 | `c128-bm25-k1-rrf` (r0001) | 0.531 | pruned | score 0.531 < 0.6 |
| 1 | `c256-bm25-k1-rrf` (r0003) | 0.539 | pruned | score 0.539 < 0.6 |
| 2 | `c64-bm25-k3-rrf` (r0002, cached) | 0.671 | kept |  |
| 2 | `c64-tfidf-k3-rrf` (r0004) | 0.595 | pruned | score 0.595 < 0.6 |
| 2 | `c64-glove-k3-rrf` (r0005) | 0.591 | pruned | score 0.591 < 0.6 |
| 3 | `c64-bm25-k1-rrf` (r0006) | 0.623 | kept |  |
| 3 | `c64-bm25-k3-rrf` (r0002, cached) | 0.665 | kept |  |
| 3 | `{'chunk_size': 64, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 416 tokens > cap |
| 4 | `c64-bm25-k3-none` (r0008) | 0.699 | pruned | outside beam |
| 4 | `c64-bm25-k3-rrf` (r0002, cached) | 0.663 | pruned | outside beam |
| 4 | `c64-bm25-k3-overlap` (r0007) | 0.728 | kept |  |
| 4 | `c64-bm25-k1-none` (r0010) | 0.690 | kept |  |
| 4 | `c64-bm25-k1-rrf` (r0006, cached) | 0.623 | pruned | outside beam |
| 4 | `c64-bm25-k1-overlap` (r0009) | 0.691 | kept |  |

Budget: 10/30 sample experiments, 2/2 validations, 4 reused from memory. Wall time 7.4 s.

## Human checkpoints and notices
- [notice] **evidence_conflict**: rerank=rrf is supported by ['kb06'] but measured 0.127 lower accuracy on this dataset -> noted
- [decision] **low_confidence**: top two configurations are statistically close (P(best) = 0.59) -> c64-bm25-k1-none

Source verification: 6 cited numbers, 0 mismatches.
