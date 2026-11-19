# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 2000
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c128-bm25-k3-overlap`**: answer recall 0.881 [r0022.accuracy], context tokens 498.2 [r0022.tokens], p95 latency 1.32 [r0022.p95_ms] ms.

Runner-up `c128-bm25-k5-none`: answer recall 0.909 [r0023.accuracy], context tokens 829.6 [r0023.tokens], p95 latency 1.45 [r0023.p95_ms] ms.

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
| 1 | `c64-bm25-k5-rrf` (r0002) | 0.720 | kept |  |
| 1 | `c128-bm25-k5-rrf` (r0001) | 0.743 | kept |  |
| 1 | `c256-bm25-k5-rrf` (r0003) | 0.692 | kept |  |
| 2 | `c128-bm25-k5-rrf` (r0001, cached) | 0.773 | kept |  |
| 2 | `c128-tfidf-k5-rrf` (r0004) | 0.733 | kept |  |
| 2 | `c128-glove-k5-rrf` (r0007) | 0.707 | pruned | outside beam |
| 2 | `c64-bm25-k5-rrf` (r0002, cached) | 0.756 | kept |  |
| 2 | `c64-tfidf-k5-rrf` (r0005) | 0.727 | pruned | outside beam |
| 2 | `c64-glove-k5-rrf` (r0008) | 0.702 | pruned | outside beam |
| 2 | `c256-bm25-k5-rrf` (r0003, cached) | 0.728 | pruned | outside beam |
| 2 | `c256-tfidf-k5-rrf` (r0006) | 0.681 | pruned | outside beam |
| 2 | `c256-glove-k5-rrf` (r0009) | 0.669 | pruned | outside beam |
| 3 | `c128-bm25-k1-rrf` (r0012) | 0.658 | pruned | outside beam |
| 3 | `c128-bm25-k3-rrf` (r0010) | 0.759 | kept |  |
| 3 | `c128-bm25-k5-rrf` (r0001, cached) | 0.756 | kept |  |
| 3 | `c64-bm25-k1-rrf` (r0013) | 0.642 | pruned | score 0.642 more than 15% below best 0.759 |
| 3 | `c64-bm25-k3-rrf` (r0011) | 0.718 | pruned | outside beam |
| 3 | `c64-bm25-k5-rrf` (r0002, cached) | 0.741 | kept |  |
| 3 | `c128-tfidf-k1-rrf` (r0015) | 0.626 | pruned | score 0.626 more than 15% below best 0.759 |
| 3 | `c128-tfidf-k3-rrf` (r0014) | 0.716 | pruned | outside beam |
| 3 | `c128-tfidf-k5-rrf` (r0004, cached) | 0.726 | pruned | outside beam |
| 4 | `c128-bm25-k3-none` (r0017) | 0.778 | pruned | outside beam |
| 4 | `c128-bm25-k3-rrf` (r0010, cached) | 0.757 | pruned | outside beam |
| 4 | `c128-bm25-k3-overlap` (r0016) | 0.812 | kept |  |
| 4 | `c128-bm25-k5-none` (r0019) | 0.795 | kept |  |
| 4 | `c128-bm25-k5-rrf` (r0001, cached) | 0.756 | pruned | outside beam |
| 4 | `c128-bm25-k5-overlap` (r0018) | 0.797 | kept |  |
| 4 | `c64-bm25-k5-none` (r0021) | 0.791 | pruned | outside beam |
| 4 | `c64-bm25-k5-rrf` (r0002, cached) | 0.741 | pruned | outside beam |
| 4 | `c64-bm25-k5-overlap` (r0020) | 0.793 | pruned | outside beam |

Budget: 21/30 sample experiments, 2/2 validations, 9 reused from memory. Wall time 10.9 s.

## Human checkpoints and notices
- [notice] **evidence_conflict**: rerank=rrf is supported by ['kb06'] but measured 0.060 lower accuracy on this dataset -> noted

Source verification: 6 cited numbers, 0 mismatches.
