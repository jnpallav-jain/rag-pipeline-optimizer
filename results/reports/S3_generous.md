# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 1000
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c64-tfidf-k3-overlap`**: answer recall 0.849 [r0024.accuracy], context tokens 249.2 [r0024.tokens], p95 latency 0.71 [r0024.p95_ms] ms.

Runner-up `c64-bm25-k3-overlap`: answer recall 0.845 [r0025.accuracy], context tokens 249.3 [r0025.tokens], p95 latency 0.84 [r0025.p95_ms] ms.

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
| 1 | `c64-bm25-k5-rrf` (r0002) | 0.623 | pruned | p95 latency 9.23 ms > cap 5.00 |
| 1 | `c128-bm25-k5-rrf` (r0001) | 0.592 | pruned | p95 latency 9.73 ms > cap 5.00 |
| 1 | `c256-bm25-k3-rrf` (r0003) | 0.536 | pruned | p95 latency 9.68 ms > cap 5.00 |
| 1 | `c64-bm25-k5-none` (r0004) | 0.781 | kept |  |
| 1 | `c128-bm25-k5-none` (r0005) | 0.751 | kept |  |
| 1 | `c256-bm25-k3-none` (r0006) | 0.724 | kept |  |
| 2 | `c64-bm25-k5-none` (r0004, cached) | 0.806 | kept |  |
| 2 | `c64-tfidf-k5-none` (r0011) | 0.783 | kept |  |
| 2 | `c64-bge-k5-none` (r0008) | 0.616 | pruned | p95 latency 8.56 ms > cap 5.00 |
| 2 | `c128-bm25-k5-none` (r0005, cached) | 0.769 | pruned | outside beam |
| 2 | `c128-tfidf-k5-none` (r0009) | 0.756 | kept |  |
| 2 | `c128-bge-k5-none` (r0007) | 0.558 | pruned | p95 latency 9.19 ms > cap 5.00 |
| 2 | `c256-bm25-k3-none` (r0006, cached) | 0.748 | pruned | outside beam |
| 2 | `c256-tfidf-k3-none` (r0012) | 0.724 | pruned | outside beam |
| 2 | `c256-bge-k3-none` (r0010) | 0.492 | pruned | p95 latency 9.30 ms > cap 5.00 |
| 3 | `c64-bm25-k1-none` (r0014) | 0.745 | pruned | outside beam |
| 3 | `c64-bm25-k3-none` (r0013) | 0.801 | kept |  |
| 3 | `c64-bm25-k5-none` (r0004, cached) | 0.795 | kept |  |
| 3 | `c64-tfidf-k1-none` (r0018) | 0.719 | pruned | outside beam |
| 3 | `c64-tfidf-k3-none` (r0016) | 0.791 | kept |  |
| 3 | `c64-tfidf-k5-none` (r0011, cached) | 0.778 | pruned | outside beam |
| 3 | `c128-tfidf-k1-none` (r0017) | 0.753 | pruned | outside beam |
| 3 | `c128-tfidf-k3-none` (r0015) | 0.776 | pruned | outside beam |
| 3 | `c128-tfidf-k5-none` (r0009, cached) | 0.749 | pruned | outside beam |
| 4 | `c64-tfidf-k3-none` (r0016, cached) | 0.788 | pruned | outside beam |
| 4 | `c64-tfidf-k3-rrf` (r0021) | 0.631 | pruned | p95 latency 9.09 ms > cap 5.00 |
| 4 | `c64-tfidf-k3-overlap` (r0023) | 0.799 | kept |  |
| 4 | `c64-bm25-k5-none` (r0004, cached) | 0.792 | pruned | outside beam |
| 4 | `c64-bm25-k5-rrf` (r0002, cached) | 0.639 | pruned | p95 latency 9.23 ms > cap 5.00 |
| 4 | `c64-bm25-k5-overlap` (r0022) | 0.793 | kept |  |
| 4 | `c64-bm25-k3-none` (r0013, cached) | 0.795 | pruned | outside beam |
| 4 | `c64-bm25-k3-rrf` (r0019) | 0.651 | pruned | p95 latency 9.09 ms > cap 5.00 |
| 4 | `c64-bm25-k3-overlap` (r0020) | 0.805 | kept |  |

Agents: rule. Budget: 23/30 sample experiments, 2/2 validations, 10 reused from memory. Wall time 4.6 s.

## Human checkpoints and notices
- [notice] **replan**: all 3 branches at depth 1 broke a hard constraint; retesting with alternative fills [('bm25', 'none'), ('bge', 'none'), ('tfidf', 'none'), ('bm25', 'overlap'), ('bge', 'overlap'), ('tfidf', 'overlap'), ('bge', 'rrf'), ('tfidf', 'rrf')] -> noted

Source verification: 6 cited numbers, 0 mismatches.
