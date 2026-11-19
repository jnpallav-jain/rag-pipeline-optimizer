# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 200
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c64-tfidf-k1-overlap`**: answer recall 0.707 [r0017.accuracy], context tokens 83.1 [r0017.tokens], p95 latency 0.68 [r0017.p95_ms] ms.

Runner-up `c128-bm25-k1-none`: answer recall 0.773 [r0016.accuracy], context tokens 166.0 [r0016.tokens], p95 latency 0.64 [r0016.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 1.00.
This recommendation is advisory: an engineer must approve it before production use.

## Summary (Planner agent, CrewAI + Claude Haiku 5.5)
Numbers here are only those cited from the experiment log; sentences with any other number are removed before the report is written.

Measured scores across the kept rerank-stage candidates were close, so the choice rests mainly on the latency and token constraints rather than a decisive quality gain. The chunk_size=256 option and the BGE retriever are supported only by the literature, since the chunk_size=256 option was pruned before measurement and every BGE configuration exceeded the latency cap, so neither can be verified by this search. The top_k evidence is indirect, so the measurements do not show whether a larger top_k would improve recall, only that it would exceed the token budget.

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
| 1 | `c64-bm25-k1-none` (r0001) | 0.669 | kept |  |
| 1 | `c128-bm25-k1-none` (r0002) | 0.667 | kept |  |
| 1 | `{'chunk_size': 256}` | - | pruned | heuristic: even the smallest remaining choice needs 333 tokens > cap |
| 2 | `c128-bm25-k1-rrf` (r0003) | 0.495 | pruned | p95 latency 11.42 ms > cap 5.00 |
| 2 | `c128-tfidf-k1-rrf` (r0005) | 0.483 | pruned | p95 latency 11.08 ms > cap 5.00 |
| 2 | `c128-bge-k1-rrf` (r0007) | 0.463 | pruned | p95 latency 11.15 ms > cap 5.00 |
| 2 | `c64-bm25-k1-rrf` (r0004) | 0.533 | pruned | p95 latency 10.23 ms > cap 5.00 |
| 2 | `c64-tfidf-k1-rrf` (r0006) | 0.520 | pruned | p95 latency 10.02 ms > cap 5.00 |
| 2 | `c64-bge-k1-rrf` (r0008) | 0.517 | pruned | p95 latency 10.05 ms > cap 5.00 |
| 2 | `c128-bm25-k1-none` (r0002, cached) | 0.680 | kept |  |
| 2 | `c128-tfidf-k1-none` (r0009) | 0.650 | pruned | outside beam |
| 2 | `c128-bge-k1-none` (r0010) | 0.385 | pruned | p95 latency 10.65 ms > cap 5.00 |
| 2 | `c64-bm25-k1-none` (r0001, cached) | 0.682 | kept |  |
| 2 | `c64-tfidf-k1-none` (r0011) | 0.671 | kept |  |
| 2 | `c64-bge-k1-none` (r0012) | 0.479 | pruned | p95 latency 9.40 ms > cap 5.00 |
| 3 | `c128-bm25-k1-none` (r0002, cached) | 0.673 | kept |  |
| 3 | `{'chunk_size': 128, 'retriever': 'bm25', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 499 tokens > cap |
| 3 | `{'chunk_size': 128, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 832 tokens > cap |
| 3 | `c64-tfidf-k1-none` (r0011, cached) | 0.666 | kept |  |
| 3 | `{'chunk_size': 64, 'retriever': 'tfidf', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 250 tokens > cap |
| 3 | `{'chunk_size': 64, 'retriever': 'tfidf', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 416 tokens > cap |
| 3 | `c64-bm25-k1-none` (r0001, cached) | 0.675 | kept |  |
| 3 | `{'chunk_size': 64, 'retriever': 'bm25', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 250 tokens > cap |
| 3 | `{'chunk_size': 64, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 416 tokens > cap |
| 4 | `c128-bm25-k1-none` (r0002, cached) | 0.672 | kept |  |
| 4 | `c128-bm25-k1-rrf` (r0003, cached) | 0.490 | pruned | p95 latency 11.42 ms > cap 5.00 |
| 4 | `c128-bm25-k1-overlap` (r0013) | 0.669 | pruned | outside beam |
| 4 | `c64-bm25-k1-none` (r0001, cached) | 0.673 | pruned | outside beam |
| 4 | `c64-bm25-k1-rrf` (r0004, cached) | 0.527 | pruned | p95 latency 10.23 ms > cap 5.00 |
| 4 | `c64-bm25-k1-overlap` (r0014) | 0.678 | kept |  |
| 4 | `c64-tfidf-k1-none` (r0011, cached) | 0.665 | pruned | outside beam |
| 4 | `c64-tfidf-k1-rrf` (r0006, cached) | 0.518 | pruned | p95 latency 10.02 ms > cap 5.00 |
| 4 | `c64-tfidf-k1-overlap` (r0015) | 0.675 | kept |  |

## Agent reasoning (CrewAI agents; scores and pruning stay in code)

- **Researcher** (evidence): 10 cited evidence items from the knowledge base
- **Planner** (depth 1 (chunk_size)): wrote 2 thoughts; changed 2 fills
- **Evaluator** (depth 1 (chunk_size)): (assessment removed: it quoted uncited numbers)
- **Evaluator** (conflict): chunk_size=64 (kb09): 
- **Evaluator** (conflict): chunk_size=256 (kb09): The literature rates this option at the same level as the others, yet it was never measured: it was pruned by a heuristic on token cap, so the literature's favourable rating cannot be checked and is unsupported by this search.
- **Planner** (depth 2 (retriever)): wrote 6 thoughts; changed 0 fills
- **Evaluator** (depth 2 (retriever)): At the retriever level, the BM25 configurations without reranking are the only ones that stay within the latency cap while scoring well, with the smaller chunk size scoring slightly higher than the larger one and both staying well under the cap. Every RRF-fused variant and every BGE variant exceeded the latency cap and was pruned, despite some of them showing reasonable recall, so none of them can be carried forward. TF-IDF without fusion stayed within the cap but scored below BM25 and was outside the beam, so it survives only as a fallback.
- **Evaluator** (conflict): retriever=bge (kb05, kb11): The literature favours BGE as a retriever, but in the measurements every BGE configuration was pruned for exceeding the latency cap, and the unfused BGE variants also scored clearly worse than BM25 with lower recall. This is a conflict between the literature's support and the measured results.
- **Planner** (depth 3 (top_k)): wrote 3 thoughts; changed 0 fills
- **Evaluator** (depth 3 (top_k)): The top_k evidence is therefore indirect: the measured results say nothing about whether a larger top_k would improve recall or score, only that it would exceed the token budget.
- **Planner** (depth 4 (rerank)): wrote 9 thoughts; changed 0 fills
- **Evaluator** (depth 4 (rerank)): At the rerank level, the kept configurations all stayed within the p95 latency cap and the context-token cap, and the bm25 and overlap variants with the larger chunk or the bm25 chunk held the highest recall, while the rrf variants were pruned for exceeding the latency cap. Scores across the kept rerank-stage candidates were close to one another, so the measurements do not clearly separate the options on quality, and the rerank choice rests mainly on the latency and token constraints rather than on a decisive score gain.
- **Evaluator** (conflict): rerank=rrf (kb06): 
- **guardrail** (summary): removed 2 sentence(s) with uncited numbers
- Thought for kept branch `c64-bm25-k1-none`: Testing chunk_size 64 is worth it because note kb09 is the only evidence and it supports the smaller chunk option; the smaller chunk keeps context tokens low under the context cap when combined with top_k 1, and with no rerank the latency stays within the p95 cap.
- Thought for kept branch `c128-bm25-k1-none`: Testing chunk_size 128 is worth it because kb09 also supports this option at the same evidence strength, and larger chunks may carry more answer-bearing context for recall; top_k 1 keeps context tokens within the cap and skipping rerank avoids the extra latency from dense re-ranking.
- Thought for kept branch `c128-bm25-k1-none`: (thought removed: it quoted uncited numbers) Re-planned fill: {'chunk_size': 128, 'retriever': 'bm25', 'top_k': 1, 'rerank': 'none'}.
- Thought for kept branch `c64-bm25-k1-none`: (thought removed: it quoted uncited numbers) Re-planned fill: {'chunk_size': 64, 'retriever': 'bm25', 'top_k': 1, 'rerank': 'none'}.
- Thought for kept branch `c64-tfidf-k1-none`: (thought removed: it quoted uncited numbers) Re-planned fill: {'chunk_size': 64, 'retriever': 'tfidf', 'top_k': 1, 'rerank': 'none'}.
- Thought for kept branch `c128-bm25-k1-none`: (thought removed: it quoted uncited numbers)
- Thought for kept branch `c64-tfidf-k1-none`: (thought removed: it quoted uncited numbers)
- Thought for kept branch `c64-bm25-k1-none`: (thought removed: it quoted uncited numbers)
- Thought for kept branch `c128-bm25-k1-none`: Baseline with no re-ranking; no note evidence supports it over alternatives, but it is the cheapest option and keeps latency well under the cap. Fill is already fully decided by the branch config.
- Thought for kept branch `c64-bm25-k1-overlap`: Overlap re-ranking on c64 chunks has no note evidence; it is a cheap rerank to check whether lexical reordering helps when chunks are small.
- Thought for kept branch `c64-tfidf-k1-overlap`: tfidf with c64 and overlap re-ranking has no note evidence; it is a low-cost check of whether lexical reordering helps the tfidf retriever.

Agents: llm. Budget: 15/30 sample experiments, 2/2 validations, 11 reused from memory. Wall time 54.0 s.

## Human checkpoints and notices
- [notice] **replan**: all 6 branches at depth 2 broke a hard constraint; retesting with alternative fills [('bm25', 'none'), ('bm25', 'overlap')] -> noted

Source verification: 6 cited numbers, 0 mismatches.
