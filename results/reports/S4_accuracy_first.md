# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 2000
- Max p95 retrieval latency: 20.0 ms

## Recommendation (validation split, 1,000 questions)
**`c64-bm25-k5-none`**: answer recall 0.874 [r0023.accuracy], context tokens 415.4 [r0023.tokens], p95 latency 0.81 [r0023.p95_ms] ms.

Runner-up `c64-bm25-k3-none`: answer recall 0.834 [r0022.accuracy], context tokens 249.2 [r0022.tokens], p95 latency 0.83 [r0022.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 0.99.
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
| 1 | `c64-bm25-k5-rrf` (r0002) | 0.763 | kept |  |
| 1 | `c128-bm25-k5-rrf` (r0001) | 0.758 | kept |  |
| 1 | `c256-bm25-k5-rrf` (r0003) | 0.684 | kept |  |
| 2 | `c64-bm25-k5-rrf` (r0002, cached) | 0.787 | kept |  |
| 2 | `c64-tfidf-k5-rrf` (r0008) | 0.767 | pruned | outside beam |
| 2 | `c64-bge-k5-rrf` (r0005) | 0.774 | kept |  |
| 2 | `c128-bm25-k5-rrf` (r0001, cached) | 0.777 | kept |  |
| 2 | `c128-tfidf-k5-rrf` (r0006) | 0.753 | pruned | outside beam |
| 2 | `c128-bge-k5-rrf` (r0004) | 0.765 | pruned | outside beam |
| 2 | `c256-bm25-k5-rrf` (r0003, cached) | 0.709 | pruned | outside beam |
| 2 | `c256-tfidf-k5-rrf` (r0009) | 0.689 | pruned | outside beam |
| 2 | `c256-bge-k5-rrf` (r0007) | 0.697 | pruned | outside beam |
| 3 | `c64-bm25-k1-rrf` (r0013) | 0.700 | pruned | outside beam |
| 3 | `c64-bm25-k3-rrf` (r0011) | 0.778 | kept |  |
| 3 | `c64-bm25-k5-rrf` (r0002, cached) | 0.776 | kept |  |
| 3 | `c64-bge-k1-rrf` (r0015) | 0.697 | pruned | outside beam |
| 3 | `c64-bge-k3-rrf` (r0014) | 0.770 | kept |  |
| 3 | `c64-bge-k5-rrf` (r0005, cached) | 0.767 | pruned | outside beam |
| 3 | `c128-bm25-k1-rrf` (r0012) | 0.718 | pruned | outside beam |
| 3 | `c128-bm25-k3-rrf` (r0010) | 0.775 | pruned | outside beam |
| 3 | `c128-bm25-k5-rrf` (r0001, cached) | 0.764 | pruned | outside beam |
| 4 | `c64-bm25-k5-none` (r0020) | 0.845 | kept |  |
| 4 | `c64-bm25-k5-rrf` (r0002, cached) | 0.779 | pruned | outside beam |
| 4 | `c64-bm25-k5-overlap` (r0018) | 0.845 | pruned | outside beam |
| 4 | `c64-bm25-k3-none` (r0017) | 0.836 | kept |  |
| 4 | `c64-bm25-k3-rrf` (r0011, cached) | 0.778 | pruned | outside beam |
| 4 | `c64-bm25-k3-overlap` (r0016) | 0.845 | kept |  |
| 4 | `c64-bge-k3-none` (r0021) | 0.736 | pruned | outside beam |
| 4 | `c64-bge-k3-rrf` (r0014, cached) | 0.772 | pruned | outside beam |
| 4 | `c64-bge-k3-overlap` (r0019) | 0.770 | pruned | outside beam |

Agents: rule. Budget: 21/30 sample experiments, 2/2 validations, 9 reused from memory. Wall time 4.9 s.

## Human checkpoints and notices
- none

Source verification: 6 cited numbers, 0 mismatches.
