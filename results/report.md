# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 400
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c64-bm25-k3-overlap`**: answer recall 0.845 [r0015.accuracy], context tokens 249.3 [r0015.tokens], p95 latency 2.09 [r0015.p95_ms] ms.

Runner-up `c64-bm25-k3-none`: answer recall 0.834 [r0014.accuracy], context tokens 249.2 [r0014.tokens], p95 latency 2.07 [r0014.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 0.96.
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
| 1 | `c64-bm25-k3-rrf` (r0002) | 0.632 | kept |  |
| 1 | `c128-bm25-k1-rrf` (r0001) | 0.581 | pruned | score 0.581 < 0.6 |
| 1 | `c256-bm25-k1-rrf` (r0003) | 0.559 | pruned | score 0.559 < 0.6 |
| 2 | `c64-bm25-k3-rrf` (r0002, cached) | 0.667 | kept |  |
| 2 | `c64-tfidf-k3-rrf` (r0004) | 0.620 | kept |  |
| 2 | `c64-glove-k3-rrf` (r0005) | 0.585 | pruned | score 0.585 < 0.6 |
| 3 | `c64-bm25-k1-rrf` (r0006) | 0.614 | kept |  |
| 3 | `c64-bm25-k3-rrf` (r0002, cached) | 0.660 | kept |  |
| 3 | `{'chunk_size': 64, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 416 tokens > cap |
| 3 | `c64-tfidf-k1-rrf` (r0007) | 0.577 | pruned | score 0.577 < 0.6 |
| 3 | `c64-tfidf-k3-rrf` (r0004, cached) | 0.623 | kept |  |
| 3 | `{'chunk_size': 64, 'retriever': 'tfidf', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 416 tokens > cap |
| 4 | `c64-bm25-k3-none` (r0009) | 0.721 | kept |  |
| 4 | `c64-bm25-k3-rrf` (r0002, cached) | 0.659 | pruned | outside beam |
| 4 | `c64-bm25-k3-overlap` (r0008) | 0.695 | kept |  |
| 4 | `c64-bm25-k1-none` (r0011) | 0.684 | kept |  |
| 4 | `c64-bm25-k1-rrf` (r0006, cached) | 0.614 | pruned | outside beam |
| 4 | `c64-bm25-k1-overlap` (r0010) | 0.675 | pruned | outside beam |
| 4 | `c64-tfidf-k3-none` (r0013) | 0.661 | pruned | outside beam |
| 4 | `c64-tfidf-k3-rrf` (r0004, cached) | 0.626 | pruned | outside beam |
| 4 | `c64-tfidf-k3-overlap` (r0012) | 0.671 | pruned | outside beam |

Budget: 13/30 sample experiments, 2/2 validations, 6 reused from memory. Wall time 16.9 s.

## Human checkpoints and notices
- [notice] **evidence_conflict**: rerank=rrf is supported by ['kb06'] but measured 0.127 lower accuracy on this dataset -> noted

Source verification: 6 cited numbers checked against the experiment log, 0 mismatches.
