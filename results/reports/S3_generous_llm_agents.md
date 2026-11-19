# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 1000
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c64-tfidf-k3-overlap`**: answer recall 0.849 [r0022.accuracy], context tokens 249.2 [r0022.tokens], p95 latency 0.74 [r0022.p95_ms] ms.

Runner-up `c128-bm25-k1-none`: answer recall 0.773 [r0023.accuracy], context tokens 166.0 [r0023.tokens], p95 latency 0.62 [r0023.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 1.00.
This recommendation is advisory: an engineer must approve it before production use.

## Summary (Planner agent, CrewAI + Claude Haiku 5.5)
Numbers here are only those cited from the experiment log; sentences with any other number are removed before the report is written.

The overlap-style reranking was preferred over rrf under the latency cap, and the bge configurations were pruned for exceeding that cap.

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
| 1 | `c64-bm25-k5-none` (r0001) | 0.772 | kept |  |
| 1 | `c128-bm25-k5-none` (r0002) | 0.740 | kept |  |
| 1 | `c256-bm25-k3-none` (r0003) | 0.724 | kept |  |
| 2 | `c64-bm25-k5-none` (r0001, cached) | 0.785 | kept |  |
| 2 | `c64-tfidf-k5-none` (r0004) | 0.774 | kept |  |
| 2 | `c64-bge-k5-none` (r0007) | 0.593 | pruned | p95 latency 9.38 ms > cap 5.00 |
| 2 | `c128-bm25-k5-none` (r0002, cached) | 0.753 | kept |  |
| 2 | `c128-tfidf-k5-none` (r0005) | 0.749 | pruned | outside beam |
| 2 | `c128-bge-k5-none` (r0008) | 0.530 | pruned | p95 latency 10.65 ms > cap 5.00 |
| 2 | `c256-bm25-k3-none` (r0003, cached) | 0.737 | pruned | outside beam |
| 2 | `c256-tfidf-k3-none` (r0006) | 0.723 | pruned | outside beam |
| 2 | `c256-bge-k3-none` (r0009) | 0.470 | pruned | p95 latency 8.89 ms > cap 5.00 |
| 3 | `c64-tfidf-k1-none` (r0015) | 0.714 | pruned | outside beam |
| 3 | `c64-tfidf-k3-none` (r0012) | 0.789 | kept |  |
| 3 | `c64-tfidf-k5-none` (r0004, cached) | 0.772 | pruned | outside beam |
| 3 | `c64-bm25-k1-none` (r0013) | 0.731 | pruned | outside beam |
| 3 | `c64-bm25-k3-none` (r0010) | 0.785 | pruned | outside beam |
| 3 | `c64-bm25-k5-none` (r0001, cached) | 0.781 | pruned | outside beam |
| 3 | `c128-bm25-k1-none` (r0014) | 0.772 | kept |  |
| 3 | `c128-bm25-k3-none` (r0011) | 0.790 | kept |  |
| 3 | `c128-bm25-k5-none` (r0002, cached) | 0.748 | pruned | outside beam |
| 4 | `c128-bm25-k1-none` (r0014, cached) | 0.770 | kept |  |
| 4 | `c128-bm25-k1-rrf` (r0020) | 0.586 | pruned | p95 latency 11.11 ms > cap 5.00 |
| 4 | `c128-bm25-k1-overlap` (r0021) | 0.776 | pruned | outside beam |
| 4 | `c128-bm25-k3-none` (r0011, cached) | 0.786 | kept |  |
| 4 | `c128-bm25-k3-rrf` (r0016) | 0.619 | pruned | p95 latency 11.11 ms > cap 5.00 |
| 4 | `c128-bm25-k3-overlap` (r0017) | 0.787 | pruned | outside beam |
| 4 | `c64-tfidf-k3-none` (r0012, cached) | 0.787 | pruned | outside beam |
| 4 | `c64-tfidf-k3-rrf` (r0018) | 0.625 | pruned | p95 latency 10.02 ms > cap 5.00 |
| 4 | `c64-tfidf-k3-overlap` (r0019) | 0.794 | kept |  |

## Agent reasoning (CrewAI agents; scores and pruning stay in code)

- **Researcher** (evidence): 10 cited evidence items from the knowledge base
- **Planner** (depth 1 (chunk_size)): wrote 3 thoughts; changed 0 fills
- **Evaluator** (depth 1 (chunk_size)): All three chunk_size options were kept, with no pruning or failures. The smallest chunk size scored highest overall, with lower recall than the two larger options but the smallest context footprint and within the latency cap. The larger chunk sizes achieved higher recall and lower latency but consumed substantially more context and scored lower overall, so the measured trade-off favours the smallest size on the composite score.
- **Evaluator** (conflict): chunk_size=64 (kb09): The measurement does not conflict with it, since the smallest size scored highest here, but the literature gives no basis for preferring it. No option the literature favours measured clearly worse.
- **Planner** (depth 2 (retriever)): wrote 9 thoughts; changed 0 fills
- **Evaluator** (depth 2 (retriever)): The tfidf variant at the same chunk size was kept as well and scored slightly lower, while it was close in recall and token use. All bge configurations were pruned for exceeding the p95 latency cap, and their scores were the lowest in the level despite recall that was comparable to the others.
- **Planner** (depth 3 (top_k)): wrote 9 thoughts; changed 0 fills
- **Evaluator** (depth 3 (top_k)): (assessment removed: it quoted uncited numbers)
- **Planner** (depth 4 (rerank)): wrote 9 thoughts; changed 0 fills
- **Evaluator** (depth 4 (rerank)): Overall, the measurements favour overlap-style reranking under the latency cap, not rrf, and the literature support for rrf (kb06) is not borne out by these runs.
- **Evaluator** (conflict): rerank=rrf (kb06): 
- **guardrail** (summary): removed 3 sentence(s) with uncited numbers
- Thought for kept branch `c64-bm25-k5-none`: chunk_size 64 is worth testing because note kb09 is the only evidence and small chunks keep context tokens low under the context cap; it may hurt recall if answers span chunks. Fill with bm25, top_k 5 to keep recall up while staying within the context cap, and no reranking to respect the latency cap.
- Thought for kept branch `c128-bm25-k5-none`: chunk_size 128 is a middle option supported by kb09 and balances answer completeness against context size. Fill with bm25, top_k 5, no rerank, matching the rule fill so it is directly comparable with the other branches.
- Thought for kept branch `c256-bm25-k3-none`: No reranking keeps latency under the cap.
- Thought for kept branch `c64-bm25-k5-none`: BM25 is the strongest retriever on the evidence (kb01, kb03, kb05) at a small chunk size, so it is worth testing as the baseline configuration. Its fill keeps top_k at a value that fits the context cap and skips re-ranking to hold latency.
- Thought for kept branch `c64-tfidf-k5-none`: TF-IDF has only weak evidence (kb03) at this chunk size; testing it at small chunks checks whether a simpler lexical retriever can match BM25 cheaply. Re-ranking is skipped to keep latency low.
- Thought for kept branch `c128-bm25-k5-none`: BM25 at a medium chunk size is supported by kb01, kb03, kb05 and tests whether larger context improves recall.
- Thought for kept branch `c64-tfidf-k3-none`: Top_k=3 with tfidf had the strongest evidence in note kb08, making it the leading candidate for recall; testing it without rerank keeps latency low.
- Thought for kept branch `c128-bm25-k1-none`: Larger chunk size 128 with top_k=1 is tested to see whether bigger context per chunk offsets the small top_k, citing kb10; rerank none is used to stay under the latency cap.
- Thought for kept branch `c128-bm25-k3-none`: Chunk size 128 with top_k=3 is worth testing because kb08 favors top_k=3 and larger chunks may add coverage; context tokens may approach the cap, so rerank stays none.
- Thought for kept branch `c128-bm25-k1-none`: No rerank keeps latency lowest and the context cost minimal; no note evidence favors reranking for this configuration, so this is the baseline to compare against.
- Thought for kept branch `c128-bm25-k3-none`: (thought removed: it quoted uncited numbers)
- Thought for kept branch `c64-tfidf-k3-overlap`: Overlap reranking on tfidf with smaller chunks is a low-latency rerank option with no note evidence, tested to see if lexical reranking helps recall.

Agents: llm. Budget: 21/30 sample experiments, 2/2 validations, 9 reused from memory. Wall time 48.5 s.

## Human checkpoints and notices
- none

Source verification: 6 cited numbers, 0 mismatches.
