# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 1000
- Max p95 retrieval latency: 0.5 ms

**No recommendation.** finalists failed validation

## Evidence used (retrieved from the knowledge base)
- **kb01** The Probabilistic Relevance Framework - BM25 and Beyond. Robertson, S. and Zaragoza, H. (2009). Foundations and Trends in Information Retrieval 3(4).
- **kb03** SQuAD - questions written against the passage. Rajpurkar, P. et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP.
- **kb05** Dense Passage Retrieval needs training data; BM25 stronger on SQuAD. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- **kb06** Reciprocal Rank Fusion. Cormack, G., Clarke, C. and Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
- **kb08** Lost in the Middle - more context is not always better. Liu, N. F. et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL.
- **kb09** Chunk size trade-off (practitioner heuristic, no single source). Design heuristic; no peer-reviewed source. Treat as weak evidence.
- **kb10** Retrieval-augmented generation. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- **kb11** BGE - general-purpose embedding models for dense retrieval. Xiao, S. et al. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE models). arXiv:2309.07597.

## Search tree (plan split, 300 questions)

| depth | configuration tested | score | status | note |
|---|---|---|---|---|
| 1 | `c64-bm25-k5-none` (r0001) | 0.611 | pruned | p95 latency 0.97 ms > cap 0.50 |
| 1 | `c128-bm25-k5-none` (r0002) | 0.562 | pruned | p95 latency 0.56 ms > cap 0.50 |
| 1 | `c256-bm25-k3-none` (r0003) | 0.551 | pruned | score 0.551 < 0.6 |
| 1 | `c64-tfidf-k5-none` (r0004) | 0.601 | pruned | p95 latency 0.68 ms > cap 0.50 |
| 1 | `c128-tfidf-k5-none` (r0005) | 0.562 | pruned | p95 latency 0.54 ms > cap 0.50 |
| 1 | `c256-tfidf-k3-none` (r0006) | 0.557 | kept | revived: best feasible node after whole level was pruned |
| 2 | `c256-bm25-k3-none` (r0003, cached) | 0.564 | kept | revived: best feasible node after whole level was pruned |
| 2 | `c256-tfidf-k3-none` (r0006, cached) | 0.557 | pruned | score 0.557 < 0.6 |
| 2 | `c256-bge-k3-none` (r0007) | 0.468 | pruned | p95 latency 8.91 ms > cap 0.50 |
| 2 | `c256-bm25-k3-overlap` (r0008) | 0.554 | pruned | p95 latency 0.54 ms > cap 0.50 |
| 2 | `c256-tfidf-k3-overlap` (r0009) | 0.552 | pruned | score 0.552 < 0.6 |
| 2 | `c256-bge-k3-overlap` (r0010) | 0.532 | pruned | p95 latency 8.91 ms > cap 0.50 |
| 3 | `c256-bm25-k1-overlap` (r0011) | 0.575 | pruned | p95 latency 0.71 ms > cap 0.50 |
| 3 | `c256-bm25-k3-overlap` (r0008, cached) | 0.555 | pruned | p95 latency 0.54 ms > cap 0.50 |
| 3 | `{'chunk_size': 256, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 1664 tokens > cap |
| 3 | `c256-bm25-k1-none` (r0012) | 0.605 | kept |  |
| 3 | `c256-bm25-k3-none` (r0003, cached) | 0.564 | pruned | score 0.564 < 0.6 |
| 4 | `c256-bm25-k1-none` (r0012, cached) | 0.603 | kept |  |
| 4 | `c256-bm25-k1-rrf` (r0013) | 0.548 | pruned | p95 latency 9.37 ms > cap 0.50 |
| 4 | `c256-bm25-k1-overlap` (r0011, cached) | 0.574 | pruned | p95 latency 0.71 ms > cap 0.50 |

## Agent reasoning (CrewAI agents; scores and pruning stay in code)

- **Researcher** (evidence): 10 cited evidence items from the knowledge base
- **Planner** (depth 1 (chunk_size)): wrote 3 thoughts; changed 0 fills
- **Evaluator** (depth 1 (chunk_size)): (assessment removed: it quoted uncited numbers)
- **Evaluator** (conflict): chunk_size=64 (kb09): The literature does not predict this result, so the measured ranking conflicts with the note's neutral rating.
- **Planner** (depth 2 (retriever)): wrote 3 thoughts; changed 0 fills
- **Evaluator** (depth 2 (retriever)): The BM25 and TF-IDF variants land close together, with BM25 scoring slightly higher, while the BGE dense retriever is clearly worse, with latency far above the cap and recall and score both lower than the sparse options. Overlap reranking did not help: it raised recall slightly but pushed latency over the cap, and the TF-IDF overlap variant scored lower than its plain counterpart.
- **Planner** (depth 3 (top_k)): wrote 2 thoughts; changed 0 fills
- **Evaluator** (depth 3 (top_k)): The k=3 and k=5 variants either exceeded the latency cap or fell below the score threshold, so the trade-off is clearly recall for latency and token budget. The measured results do not show a case where the literature-favoured option clearly failed while another option was clearly better on the same measurement.
- **Planner** (depth 4 (rerank)): wrote 3 thoughts; changed 0 fills
- **Evaluator** (depth 4 (rerank)): Adding the rrf reranker lowered recall and pushed p95 latency far above the cap, so it was pruned, and the overlap reranker also failed the latency cap despite a slightly lower score. Within this level, the reranking options did not improve on the no-rerank baseline, and rrf was measured clearly worse on recall, latency and score.
- **Evaluator** (conflict): rerank=rrf (kb06): The literature note kb06 favours rrf as a reranking choice, but the measured run r0013 was pruned: its recall fell below the no-rerank baseline, its p95 latency was far above the cap, and its score was lower. This is a direct conflict between the cited evidence and the measurement.
- Thought for kept branch `c256-tfidf-k3-none`: Larger c256 chunks give more context per hit; kb09 gives no discriminating signal. Re-planned fill: {'chunk_size': 256, 'retriever': 'tfidf', 'top_k': 3, 'rerank': 'none'}.
- Thought for kept branch `c256-bm25-k3-none`: With chunk_size 256 and top_k 3 the context stays well under the cap, and rerank none keeps latency within budget.
- Thought for kept branch `c256-bm25-k1-none`: Testing top_k=1 with bm25 at chunk size 256 is worth checking because it keeps context tokens far below the cap, and note kb10 gave it a moderate score. Overlap re-ranking is cheap and avoids the dense-retriever latency cost, so it fits the p95 latency cap. Re-planned fill: {'chunk_size': 256, 'retriever': 'bm25', 'top_k': 1, 'rerank': 'none'}.
- Thought for kept branch `c256-bm25-k1-none`: No rerank keeps latency lowest and avoids added cost; no notes support it either way (no evidence beyond the baseline score), so it serves as the control branch.

Agents: llm. Budget: 13/30 sample experiments, 1/2 validations, 6 reused from memory. Wall time 38.8 s.

## Human checkpoints and notices
- [notice] **replan**: all 3 branches at depth 1 broke a hard constraint; retesting with alternative fills [('tfidf', 'none'), ('bge', 'none'), ('bm25', 'overlap'), ('tfidf', 'overlap'), ('bge', 'overlap'), ('bm25', 'rrf'), ('tfidf', 'rrf'), ('bge', 'rrf')] -> noted
- [notice] **level_collapsed**: all depth-1 nodes fell below the score threshold; continuing with c256-tfidf-k3-none -> noted
- [notice] **replan**: all 3 branches at depth 2 broke a hard constraint; retesting with alternative fills [('bm25', 'overlap'), ('bm25', 'rrf')] -> noted
- [notice] **level_collapsed**: all depth-2 nodes fell below the score threshold; continuing with c256-bm25-k3-none -> noted
- [notice] **replan**: all 2 branches at depth 3 broke a hard constraint; retesting with alternative fills [('bm25', 'none'), ('bm25', 'rrf')] -> noted
- [decision] **infeasible**: finalists violate constraints on the validation split -> stop

Source verification: 0 cited numbers, 0 mismatches.
