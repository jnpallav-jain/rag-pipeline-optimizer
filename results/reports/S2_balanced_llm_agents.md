# RAG Pipeline Optimizer report

## Request
- Dataset: `squad-v1.1-dev` (public, allowlisted)
- Max context tokens: 400
- Max p95 retrieval latency: 5.0 ms

## Recommendation (validation split, 1,000 questions)
**`c256-bm25-k1-none`**: answer recall 0.809 [r0010.accuracy], context tokens 330.2 [r0010.tokens], p95 latency 0.51 [r0010.p95_ms] ms.

Runner-up `c256-tfidf-k1-overlap`: answer recall 0.789 [r0011.accuracy], context tokens 330.8 [r0011.tokens], p95 latency 0.46 [r0011.p95_ms] ms.

Confidence that the recommendation beats the runner-up: 0.96.
This recommendation is advisory: an engineer must approve it before production use.

## Summary (Planner agent, CrewAI + Claude Haiku 5.5)
Numbers here are only those cited from the experiment log; sentences with any other number are removed before the report is written.

The runner-up, c256-tfidf-k1-overlap, is slightly faster at 0.46 [r0011.p95_ms] ms p95 and uses a comparable 330.8 [r0011.tokens] context tokens, but its answer recall is lower at 0.789 [r0011.accuracy]. The BM25 retriever with no reranker was kept over TF-IDF with overlap reranking because it scored highest on recall while remaining within the latency cap, whereas the rrf reranking variants were pruned for breaching that cap without a measured recall gain.

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
| 1 | `c64-bm25-k3-rrf` (r0001) | 0.573 | pruned | p95 latency 10.15 ms > cap 5.00 |
| 1 | `c128-bm25-k1-rrf` (r0002) | 0.544 | pruned | p95 latency 11.17 ms > cap 5.00 |
| 1 | `c256-bm25-k1-none` (r0003) | 0.684 | kept |  |
| 2 | `c256-bm25-k1-none` (r0003, cached) | 0.698 | kept |  |
| 2 | `c256-tfidf-k1-none` (r0004) | 0.658 | kept |  |
| 2 | `c256-bge-k1-none` (r0005) | 0.368 | pruned | p95 latency 8.92 ms > cap 5.00 |
| 3 | `c256-bm25-k1-none` (r0003, cached) | 0.691 | kept |  |
| 3 | `{'chunk_size': 256, 'retriever': 'bm25', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 998 tokens > cap |
| 3 | `{'chunk_size': 256, 'retriever': 'bm25', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 1664 tokens > cap |
| 3 | `c256-tfidf-k1-none` (r0004, cached) | 0.655 | kept |  |
| 3 | `{'chunk_size': 256, 'retriever': 'tfidf', 'top_k': 3}` | - | pruned | heuristic: even the smallest remaining choice needs 998 tokens > cap |
| 3 | `{'chunk_size': 256, 'retriever': 'tfidf', 'top_k': 5}` | - | pruned | heuristic: even the smallest remaining choice needs 1664 tokens > cap |
| 4 | `c256-bm25-k1-none` (r0003, cached) | 0.689 | kept |  |
| 4 | `c256-bm25-k1-rrf` (r0006) | 0.478 | pruned | p95 latency 9.34 ms > cap 5.00 |
| 4 | `c256-bm25-k1-overlap` (r0007) | 0.682 | kept |  |
| 4 | `c256-tfidf-k1-none` (r0004, cached) | 0.656 | pruned | outside beam |
| 4 | `c256-tfidf-k1-rrf` (r0008) | 0.462 | pruned | p95 latency 9.32 ms > cap 5.00 |
| 4 | `c256-tfidf-k1-overlap` (r0009) | 0.673 | kept |  |

## Agent reasoning (CrewAI agents; scores and pruning stay in code)

- **Researcher** (evidence): 10 cited evidence items from the knowledge base
- **Planner** (depth 1 (chunk_size)): wrote 3 thoughts; changed 1 fills
- **Evaluator** (depth 1 (chunk_size)): (assessment removed: it quoted uncited numbers)
- **Planner** (depth 2 (retriever)): wrote 3 thoughts; changed 3 fills
- **Evaluator** (depth 2 (retriever)): At the retriever level, the BM25 configuration scored highest and was kept, with recall above the other kept option, latency within the cap, and context tokens in the same range as the alternatives. TF-IDF was kept as well but scored lower, with lower recall and slightly higher latency that is still within the cap. The BGE configuration was pruned: its recall was the lowest measured, its latency exceeded the cap, and its score was the weakest of the three.
- **Evaluator** (conflict): retriever=bge (kb05, kb11): The literature cites bge with a relatively low score, yet the measurements place it below the other retrievers on score and recall and it was pruned for latency. This is a case where the literature's support for bge is not reflected in the measured results, although the literature's low score for bge is consistent with the measured outcome.
- **Planner** (depth 3 (top_k)): wrote 2 thoughts; changed 2 fills
- **Evaluator** (depth 3 (top_k)): The top_k=3 and top_k=5 branches were never measured; they were pruned by a token-budget heuristic before any run, so the measurements say nothing about their recall, latency or score. The literature favours top_k=3 (kb08) and top_k=5 (kb08), but the search gives no evidence for or against them, and the kb10 note for top_k=1 is consistent with the measured results.
- **Planner** (depth 4 (rerank)): wrote 6 thoughts; changed 0 fills
- **Evaluator** (depth 4 (rerank)): At the rerank depth, the configurations that used no reranker or the overlap option stayed within the latency cap and were kept, with the bm25 no-rerank setup scoring highest among them. The rrf variants on bm25 and tfidf both breached the latency cap and were pruned, despite reaching lower recall and lower scores than the kept options. Overall, the measurements show rrf reranking as the costly choice here, with no measured gain in answer recall to offset its latency.
- **Evaluator** (conflict): rerank=rrf (kb06): This is a clear conflict between the literature's support for rrf and the measured results under the latency constraint.
- **guardrail** (summary): removed 2 sentence(s) with uncited numbers
- Thought for kept branch `c256-bm25-k1-none`: chunk_size 256 gives fuller context per hit but risks exceeding the context cap at higher top_k; evidence is limited to note kb09. Fill with bm25 and top_k 1 to stay within the context cap, with no reranking to protect the latency cap.
- Thought for kept branch `c256-bm25-k1-none`: BM25 has the strongest evidence so far, supported by notes kb01, kb03 and kb05, so it is worth testing as the lead retriever. Fill top_k=1 keeps context within the cap at chunk_size 256, and rerank 'none' avoids the extra latency of rrf.
- Thought for kept branch `c256-tfidf-k1-none`: TF-IDF has weaker evidence, supported only by note kb03, but it is a cheap lexical baseline worth checking against BM25. Fill top_k=1 and rerank 'none' keep it within both caps.
- Thought for kept branch `c256-bm25-k1-none`: Re-ranking with rrf uses the dense retriever and adds latency, so the cheaper 'none' fill keeps this branch within the p95 latency cap while still testing the bm25 baseline.
- Thought for kept branch `c256-tfidf-k1-none`: Using 'none' for rerank avoids the extra latency of the dense retriever that rrf requires, keeping the branch within the caps.
- Thought for kept branch `c256-bm25-k1-none`: Testing rerank none gives the cheapest baseline with BM25 at chunk size 256 and top_k 1; no note evidence favors it, but it keeps latency well under the cap.
- Thought for kept branch `c256-bm25-k1-overlap`: Overlap reranking is a lightweight lexical option with no note evidence; it is worth a cheap check against the baseline with BM25.
- Thought for kept branch `c256-tfidf-k1-overlap`: Overlap reranking on TF-IDF is a low-cost option with no note evidence, worth testing to complete the rerank comparison for this retriever.

Agents: llm. Budget: 9/30 sample experiments, 2/2 validations, 5 reused from memory. Wall time 44.4 s.

## Human checkpoints and notices
- none

Source verification: 9 cited numbers, 0 mismatches.
