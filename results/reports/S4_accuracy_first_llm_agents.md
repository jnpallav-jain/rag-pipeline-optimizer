# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 2000
- Max p95 retrieval latency: 20.0 ms

## Recommendation (validation split, 1,000 questions)
**`c64-bm25-k3-overlap`**: answer recall 0.845 [r0026.accuracy], context tokens 249.3 [r0026.tokens], p95 latency 0.79 [r0026.p95_ms] ms.

Runner-up `c64-bm25-k5-none`: answer recall 0.874 [r0025.accuracy], context tokens 415.4 [r0025.tokens], p95 latency 0.85 [r0025.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 0.67 (low: a human reviewed this choice).
This recommendation is advisory: an engineer must approve it before production use.

## Summary (Planner agent, CrewAI + Claude Haiku 5.5)
Numbers here are only those cited from the experiment log; sentences with any other number are removed before the report is written.

The ranking follows the composite score rather than recall alone, since larger top_k values scored higher on recall while the k=5 and k=3 bm25 settings stayed within the p95 latency and context-token caps. The 'rrf' variants were pruned despite higher recall because their p95 latency was well above the 'none' and 'overlap' options, and the measurements contain no rerank-stage evidence to support the literature's preference for rerank.

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
| 1 | `c64-bm25-k5-none` (r0002) | 0.827 | kept |  |
| 1 | `c128-bm25-k5-none` (r0001) | 0.824 | kept |  |
| 1 | `c256-bm25-k3-overlap` (r0003) | 0.802 | kept |  |
| 2 | `c64-bm25-k5-rrf` (r0005) | 0.765 | kept |  |
| 2 | `c64-tfidf-k5-rrf` (r0008) | 0.755 | kept |  |
| 2 | `c64-bge-k5-rrf` (r0011) | 0.739 | kept |  |
| 2 | `c128-bm25-k5-rrf` (r0004) | 0.750 | pruned | outside beam |
| 2 | `c128-tfidf-k5-rrf` (r0006) | 0.736 | pruned | outside beam |
| 2 | `c128-bge-k5-rrf` (r0010) | 0.726 | pruned | outside beam |
| 2 | `c256-bm25-k3-rrf` (r0007) | 0.728 | pruned | outside beam |
| 2 | `c256-tfidf-k3-rrf` (r0009) | 0.717 | pruned | outside beam |
| 2 | `c256-bge-k3-rrf` (r0012) | 0.705 | pruned | outside beam |
| 3 | `c64-tfidf-k1-none` (r0017) | 0.738 | pruned | outside beam |
| 3 | `c64-tfidf-k3-none` (r0014) | 0.825 | pruned | outside beam |
| 3 | `c64-tfidf-k5-none` (r0016) | 0.828 | kept |  |
| 3 | `c64-bm25-k1-none` (r0015) | 0.761 | pruned | outside beam |
| 3 | `c64-bm25-k3-none` (r0013) | 0.830 | kept |  |
| 3 | `c64-bm25-k5-none` (r0002, cached) | 0.842 | kept |  |
| 3 | `c64-bge-k1-none` (r0020) | 0.637 | pruned | score 0.637 more than 15% below best 0.842 |
| 3 | `c64-bge-k3-none` (r0018) | 0.713 | pruned | score 0.713 more than 15% below best 0.842 |
| 3 | `c64-bge-k5-none` (r0019) | 0.731 | pruned | outside beam |
| 4 | `c64-bm25-k5-none` (r0002, cached) | 0.840 | kept |  |
| 4 | `c64-bm25-k5-rrf` (r0005, cached) | 0.763 | pruned | outside beam |
| 4 | `c64-bm25-k5-overlap` (r0023) | 0.838 | kept |  |
| 4 | `c64-bm25-k3-none` (r0013, cached) | 0.827 | pruned | outside beam |
| 4 | `c64-bm25-k3-rrf` (r0021) | 0.758 | pruned | outside beam |
| 4 | `c64-bm25-k3-overlap` (r0022) | 0.836 | kept |  |
| 4 | `c64-tfidf-k5-none` (r0016, cached) | 0.827 | pruned | outside beam |
| 4 | `c64-tfidf-k5-rrf` (r0008, cached) | 0.757 | pruned | outside beam |
| 4 | `c64-tfidf-k5-overlap` (r0024) | 0.829 | pruned | outside beam |

## Agent reasoning (CrewAI agents; scores and pruning stay in code)

- **Researcher** (evidence): 10 cited evidence items from the knowledge base
- **Planner** (depth 1 (chunk_size)): wrote 3 thoughts; changed 3 fills
- **Evaluator** (depth 1 (chunk_size)): All three chunk_size options were kept with no pruning or failures.
- **Evaluator** (conflict): chunk_size=64 (kb09): 
- **Evaluator** (conflict): chunk_size=256 (kb09): 
- **Planner** (depth 2 (retriever)): wrote 9 thoughts; changed 3 fills
- **Evaluator** (depth 2 (retriever)): The retriever ranking therefore follows the composite score rather than recall alone.
- **Planner** (depth 3 (top_k)): wrote 9 thoughts; changed 9 fills
- **Evaluator** (depth 3 (top_k)): At this depth, larger top_k values measured higher on answer recall and on the combined score, with the k=5 setting (bm25, c64, no reranker) scoring highest among the kept nodes and the k=3 setting close behind. The k=1 settings measured clearly lower on recall and were pruned, while the k=5 and k=3 bm25 settings stayed within the cap on p95 latency and context tokens.
- **Planner** (depth 4 (rerank)): wrote 9 thoughts; changed 0 fills
- **Evaluator** (depth 4 (rerank)): The 'rrf' variants were pruned as outside the beam despite higher recall, and their p95 latency was well above the 'none' and 'overlap' options. The literature favours 'rrf' (note kb06), but the measurements do not show a rerank stage, so the evidence for 'rerank' itself is missing.
- **Evaluator** (conflict): rerank=rrf (kb06): Their p95 latency was also far above the none and overlap options. This conflicts with the literature's preference for rrf.
- **guardrail** (summary): removed 2 sentence(s) with uncited numbers
- Thought for kept branch `c64-bm25-k5-none`: chunk_size 64 is worth testing because it keeps context tokens low under the cap and note kb09 gives it some support, though its evidence score is weaker. Pairing with bm25, top_k 5 and no reranking keeps p95 latency under the cap.
- Thought for kept branch `c128-bm25-k5-none`: chunk_size 128 has the strongest evidence score in note kb09, making it the leading candidate for recall. bm25 with top_k 5 and no reranking fits the context cap and avoids the extra latency of rrf.
- Thought for kept branch `c256-bm25-k3-overlap`: chunk_size 256 is tested as a larger-context contrast, supported only by note kb09 with a lower score. Context tokens at top_k 5 may approach the cap, so top_k 3 keeps it safely within the limit while using overlap reranking, which is cheap in latency.
- Thought for kept branch `c64-bm25-k5-rrf`: BM25 is the strongest retriever signal here, supported by notes kb01, kb03 and kb05, so a small chunk with BM25 is worth testing first. Pairing it with top_k 5 and rrf re-ranking keeps recall high, while the context cap is respected at this chunk size.
- Thought for kept branch `c64-tfidf-k5-rrf`: TF-IDF has weaker evidence (only kb03), so it is worth testing mainly as a lexical baseline at the smaller chunk size. Using top_k 5 with rrf gives a fair comparison against BM25 under the same settings.
- Thought for kept branch `c64-bge-k5-rrf`: BGE dense retrieval has modest evidence (kb05, kb11) and may capture semantic matches BM25 misses at a small chunk size. Testing with top_k 5 and rrf is the natural dense configuration, though latency should be watched.
- Thought for kept branch `c64-tfidf-k5-none`: top_k=5 with tfidf is supported by kb08 but with a lower score than top_k=3, and it raises context size. Rerank is set to none so the latency cap is not exceeded.
- Thought for kept branch `c64-bm25-k3-none`: top_k=3 with bm25 is backed by kb08 and tests whether bm25 improves on tfidf at the best-supported k. Rerank is set to none to respect the latency cap.
- Thought for kept branch `c64-bm25-k5-none`: top_k=5 with bm25 is backed by kb08 and tests whether a larger sparse pool helps recall despite the context cost. Rerank is set to none to respect the latency cap.
- Thought for kept branch `c64-bm25-k5-none`: Testing rerank=none gives a no-overhead baseline; evidence is weak (no notes), so it serves as the control against which reranking gains are measured.
- Thought for kept branch `c64-bm25-k5-overlap`: Overlap reranking is a cheap lexical option with no supporting notes; testing it fills the gap between no reranking and the dense-based rrf at low latency cost.
- Thought for kept branch `c64-bm25-k3-overlap`: Top_k 3 with overlap reranking checks a low-context, low-latency lexical reranker; no evidence, so it is a cheap exploratory branch.

Agents: llm. Budget: 24/30 sample experiments, 2/2 validations, 6 reused from memory. Wall time 52.6 s.

## Human checkpoints and notices
- [decision] **low_confidence**: top two configurations are statistically close (P(best) = 0.67) -> c64-bm25-k3-overlap

Source verification: 6 cited numbers, 0 mismatches.
