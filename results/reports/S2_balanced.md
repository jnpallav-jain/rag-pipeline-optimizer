# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 400
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c128-bm25-k1-none`**: answer recall 0.773 [r0020.accuracy], context tokens 166.0 [r0020.tokens], p95 latency 0.53 [r0020.p95_ms] ms.

Runner-up `c128-bm25-k1-overlap`: answer recall 0.755 [r0019.accuracy], context tokens 166.1 [r0019.tokens], p95 latency 0.55 [r0019.p95_ms] ms.

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
| 1 | `c64-bm25-k3-rrf` (r0002) | 0.573 | pruned | p95 latency 10.51 ms > cap 5.00 |
| 1 | `c128-bm25-k1-rrf` (r0001) | 0.556 | pruned | p95 latency 9.05 ms > cap 5.00 |
| 1 | `c256-bm25-k1-rrf` (r0003) | 0.470 | pruned | p95 latency 8.96 ms > cap 5.00 |
| 1 | `c64-bm25-k3-none` (r0004) | 0.725 | kept |  |
| 1 | `c128-bm25-k1-none` (r0005) | 0.740 | kept |  |
| 1 | `c256-bm25-k1-none` (r0006) | 0.685 | kept |  |
| 2 | `c128-bm25-k1-none` (r0005, cached) | 0.758 | kept |  |
| 2 | `c128-tfidf-k1-none` (r0008) | 0.720 | pruned | outside beam |
| 2 | `c128-bge-k1-none` (r0007) | 0.473 | pruned | p95 latency 8.37 ms > cap 5.00 |
| 2 | `c64-bm25-k3-none` (r0004, cached) | 0.749 | kept |  |
| 2 | `c64-tfidf-k3-none` (r0011) | 0.732 | kept |  |
| 2 | `c64-bge-k3-none` (r0009) | 0.550 | pruned | p95 latency 9.93 ms > cap 5.00 |
| 2 | `c256-bm25-k1-none` (r0006, cached) | 0.710 | pruned | outside beam |
| 2 | `c256-tfidf-k1-none` (r0012) | 0.674 | pruned | outside beam |
| 2 | `c256-bge-k1-none` (r0010) | 0.391 | pruned | p95 latency 8.47 ms > cap 5.00 |
| 3 | `c128-bm25-k1-none` (r0005, cached) | 0.748 | kept |  |
| 3 | `{'chunk_size': 128, 'retriever': 'bm25', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 499 tokens > cap |
| 3 | `{'chunk_size': 128, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 832 tokens > cap |
| 3 | `c64-tfidf-k1-none` (r0014) | 0.700 | pruned | outside beam |
| 3 | `c64-tfidf-k3-none` (r0011, cached) | 0.735 | kept |  |
| 3 | `{'chunk_size': 64, 'retriever': 'tfidf', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 416 tokens > cap |
| 3 | `c64-bm25-k1-none` (r0013) | 0.726 | pruned | outside beam |
| 3 | `c64-bm25-k3-none` (r0004, cached) | 0.746 | kept |  |
| 3 | `{'chunk_size': 64, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 416 tokens > cap |
| 4 | `c128-bm25-k1-none` (r0005, cached) | 0.743 | kept |  |
| 4 | `c128-bm25-k1-rrf` (r0001, cached) | 0.565 | pruned | p95 latency 9.05 ms > cap 5.00 |
| 4 | `c128-bm25-k1-overlap` (r0016) | 0.756 | kept |  |
| 4 | `c64-tfidf-k3-none` (r0011, cached) | 0.732 | kept |  |
| 4 | `c64-tfidf-k3-rrf` (r0017) | 0.575 | pruned | p95 latency 10.49 ms > cap 5.00 |
| 4 | `c64-tfidf-k3-overlap` (r0018) | 0.742 | pruned | outside beam |
| 4 | `c64-bm25-k3-none` (r0004, cached) | 0.741 | pruned | outside beam |
| 4 | `c64-bm25-k3-rrf` (r0002, cached) | 0.595 | pruned | p95 latency 10.51 ms > cap 5.00 |
| 4 | `c64-bm25-k3-overlap` (r0015) | 0.750 | pruned | outside beam |

Budget: 18/30 sample experiments, 2/2 validations, 11 reused from memory. Wall time 5.6 s.

## Human checkpoints and notices
- [notice] **replan**: all 3 branches at depth 1 broke a hard constraint; retesting with alternative fills [('bm25', 'none'), ('bge', 'none'), ('tfidf', 'none'), ('bm25', 'overlap'), ('bge', 'overlap'), ('tfidf', 'overlap'), ('bge', 'rrf'), ('tfidf', 'rrf')] -> noted

Source verification: 6 cited numbers, 0 mismatches.
