# Plan: v2 — modern embeddings, a vector index and an LLM reader

Status: **complete Nov 19, 2026 — all 5 steps done; v2 pushed as `1cabea7`.**

## Why

v1 optimizes only the retrieval half of RAG. It uses GloVe (2014) as its only dense retriever,
an in-memory numpy index, and no LLM; the score is *answer recall* (is the gold answer string in
the retrieved context). That is a sound upper bound, but it leaves two questions open that an
interviewer will ask:

1. Does a modern embedding model change the picture (v1: BM25 wins on SQuAD)?
2. Does the configuration with the best retrieval recall also give the best *answers*? Research
   in the knowledge base (`kb08_lost_middle.md`) says more context can hurt the reader even when
   recall goes up. v1 cannot test that; v2 can.

v2 answers both, and keeps what makes v1 good: every number traceable to a logged run,
reproducible results, cheap searches.

## What does not change

- The agents stay deterministic Python (Planner, Researcher, Experimenter, Evaluator). The LLM is
  the *reader inside the pipeline being tested*, not the agent's brain. This keeps the search
  reproducible and free. (An LLM planner is possible later; out of scope here.)
- The search loop still optimizes answer recall on the 300-question plan split; it never calls the
  LLM. Generation runs only after the search, on a small set of finalist configurations.
- The test split (2,000 questions) stays unseen by the agent.
- v1 results are kept for comparison in `results/v1/`.

## Stack choice

| part | choice | why |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` via `sentence-transformers` (384-d, runs on CPU, free) | Strong small open model; no API key; different from Project 1, which uses Voyage embeddings |
| Vector index | FAISS `IndexFlatIP` (exact cosine), saved to `data/index/` | Real vector index without a server. A DB like Chroma adds client overhead that would distort the p95-latency metric the agent optimizes. Project 1 already shows pgvector |
| Reader LLM | `claude-haiku-5-5` via the `anthropic` SDK, thinking off, short answers (`max_tokens` ~64) | Pallav's choice (Nov 19): cheapest current model; SQuAD answers are short spans |
| Answer metrics | SQuAD exact match (EM) and token F1 (free, standard); **no LLM judge by default** | Keeps calls down; F1 already gives partial credit for differently phrased answers |

**Budget: under $1 for the whole project, all API calls included** (Pallav, Nov 19). Enforced
in code, not just planned (see Step 3).

## Steps (each one is a separate go/no-go)

### Step 1 — Modern dense retriever + FAISS — ✅ done Nov 19
- **Result:** GloVe removed; `bge-small-en-v1.5` + FAISS `IndexFlatIP` built (indexes cached in
  `data/index/`, gitignored). 16/16 tests pass. FAISS + PyTorch segfaulted together on macOS;
  fixed by running OpenMP single-threaded (`rag_optimizer/__init__.py`). First reading, 300 plan
  questions, chunk 128, top-3: recall BM25 0.897, TF-IDF 0.863, BGE 0.813; with RRF BM25 0.917,
  BGE 0.917. BGE p95 ≈ 8 ms (query embedding on CPU) vs BM25 0.6 ms.
- **Do:** replace the GloVe retriever with `bge` (embed chunks once, store in FAISS, embed each
  query at search time; query latency includes the embedding time). Still **81 configurations**.
  RRF fuses lexical retrievers with `bge` instead of GloVe.
- **Why:** answers question 1; adds real embeddings + vector index to the project.
- **Files:** `rag_optimizer/pipeline.py`, `requirements.txt`, `scripts/download_data.py` (model
  download), `eval/run_eval.py`, `eval/grid.py`, `eval/summarize.py`, `docs/make_charts.py`, new
  `rag_optimizer/knowledge_base/kb11_bge.md` (citation for the new option), `tests/test_agent.py`.
- **Watch:** query embedding on CPU costs a few ms (estimate), so `bge` may fail the 5 ms
  scenarios. That is a real trade-off the agent should surface, not a bug. `torch` is a large
  install (several hundred MB, estimate).
- **You do:** `pip install -r requirements.txt`, `python scripts/download_data.py`.
- **Done when:** all tests pass; `bge` configs appear in the grid.

### Step 2 — Re-run the retrieval evaluation — ✅ done Nov 19
- **Result** (`results/headline_metrics.json`, test split, metric answer recall / utility regret):
  agent within 0.01 of exhaustive best 64% of runs (v1 60%), within 0.02 88% (v1 76%), mean regret
  0.0101 (v1 0.0115), 17.2 of 81 experiments, 150/150 report numbers verified, safety tests as
  designed (injection quarantined, 9/9 faults recovered). Retriever means over their 27 configs:
  BM25 0.851, TF-IDF 0.843, BGE 0.832. Best config `c256-bm25-k5-rrf` 0.940 (BM25+BGE fusion) vs
  `c256-bm25-k5-none` 0.939, at 10.1 ms vs 0.55 ms p95. The agent never picks BGE: under 5 ms caps
  BGE and every RRF config are infeasible, and at 20 ms (S4) it gains ≤0.001. New finding: the
  no-knowledge-base ablation now beats the full agent (regret 0.0037 vs 0.0101): the research notes
  favour dense retrieval and RRF, which don't pay off on SQuAD. S4 latency cap raised 5 → 20 ms so
  BGE could compete at all. Runtime 7 min 53 s.
- **Do:** move v1 outputs to `results/v1/`; re-run `eval/grid.py` (108 configs), `eval/run_eval.py`
  and `eval/summarize.py`; regenerate charts.
- **Why:** every headline number in the README changes with the bigger search space.
- **Files:** `results/**`, `docs/*.png`.
- **You do:** nothing (runs locally, no API cost; roughly 20–30 min, estimate).
- **Done when:** new `results/headline_metrics.json`; I report v1 vs v2 side by side.

### Step 3 — LLM reader (generation), behind a flag — ✅ done Nov 19
- **Result:** 20-call pilot (`c128-bm25-k3-none`, plan questions): 894 input / 7.6 output tokens
  per call as billed, $0.0019 total, median latency 0.63 s, EM 0.85. 21/21 tests pass.
- **Do:** new `rag_optimizer/reader.py`: prompt = question + retrieved chunks, short extractive
  answer. Every response is cached on disk (`results/llm_cache.jsonl`, keyed by model + prompt
  hash) so reruns are free and reproducible. A spend ledger sums the API's `usage` token counts
  at Haiku 5.5 prices across *all* runs; any call that would take the total past **$1.00** is
  refused (the run stops at $0.90 to leave headroom). Tests use a fake client, so the tests still
  need no API key. Off by default: the CLI runs exactly as before unless `--generate` is passed.
- **Files:** new `rag_optimizer/reader.py`, `rag_optimizer/cli.py`, `requirements.txt`
  (`anthropic`), new `tests/test_reader.py`.
- **You do:** have an Anthropic API key available (`ANTHROPIC_API_KEY` or `ant auth login`).
- **Done when:** a 20-call pilot runs; I report measured tokens and cost per call, replacing
  the estimates below.

### Step 4 — End-to-end evaluation — ✅ done Nov 19
- **Result** (`results/generation_summary.json`, 200 test questions, Haiku 5.5, Batch API): top-k
  sweep at `c256-bm25-*-rrf` k=1/3/5: recall 0.855/0.935/0.950, EM 0.705/0.750/0.775, F1
  0.788/0.844/0.847, EM when retrieved 0.825/0.802/0.816 (flat: no lost-in-the-middle at ≤2.7K
  tokens). Recall and EM rank the 6 configs in the same order (one EM tie). Recall overstates EM by
  12–19 points; at k=5, 24 of 45 misses overlap the gold span. Billed tokens ≈ 1.6x the agent's
  words x 1.3 proxy. **Project LLM spend $0.0725** over 1,220 calls (projected $0.056 for the batch).
- **Do:** new `eval/run_generation.py`. For a fixed 200-question sample of the test split, run
  the reader on at most 6 distinct configs: the top-k sweep (k = 1, 3, 5 at the best
  chunk/retriever, which tests "more context hurts the reader") plus up to 3 agent picks from
  the scenarios (deduplicated). **At most 1,200 calls**, sent through the Batch API (50% off).
  Score EM/F1 locally. Calls already in the cache are never re-sent.
- **Why:** answers question 2 — does retrieval recall predict answer quality, and does adding
  chunks hurt the reader?
- **Files:** new `eval/run_generation.py`, `results/generation_*.json|md`, `eval/summarize.py`.
- **You do:** approve the cost before the full run (I'll show the pilot-based number first).
- **Done when:** a table of recall vs EM/F1 per config, with cost and latency.

### Step 5 — Write-up — ✅ done Nov 19 (reviewed by Pallav, pushed `1cabea7`)
- **Do:** update README (architecture diagram, new results, honest reading, limitations), add the
  project to the source of truth if you want it tracked, commit and push.
- **Files:** `README.md`, `docs/architecture.png`, `../ai-engineer-source-of-truth.md` (if wanted).
- **You do:** review the README before I push.

## Cost estimate for Step 4 (estimate — the pilot in Step 3 replaces it)

Model: `claude-haiku-5-5` at $0.10 input / $0.50 output per million tokens (Anthropic list price,
prompts under 100K tokens). Assumptions: ~1,000 input tokens per call (v1 context ranges ~80–1,660
tokens by the words x 1.3 estimate, plus a ~150-token prompt); ~30 output tokens (short answer,
thinking off).

| run | calls | input | output | standard | with Batch API |
|---|---|---|---|---|---|
| Step 3 pilot (non-batch, also measures reader latency) | 20 | ~20K | ~0.6K | ~$0.002 | — |
| Step 4 main run | ≤1,200 | ~1.2M | ~36K | ~$0.14 | **~$0.07** |
| **Total** | **≤1,220** | | | ~$0.14 | **~$0.08** |

That leaves more than 10x headroom under the $1 cap, which covers a rerun if something breaks
(the cache makes most reruns free anyway). If the pilot measures more than ~3x these token
counts, I stop and report before Step 4. Prompt caching won't help: the shared prompt is shorter
than the minimum cacheable length and the context differs per question.

## Decisions for you

1. ~~Reader model~~ — **decided Nov 19: Haiku 5.5.**
2. ~~Spend cap~~ — **decided Nov 19: under $1 total, as few calls as possible.**
3. ~~Track in the source of truth?~~ — **decided Nov 19: yes, as a side project.** It doesn't
   replace Projects 1–4 and stays small: finish every step above, nothing more.
4. ~~Keep GloVe?~~ — **decided Nov 19: no. `bge` replaces GloVe**, so the search space stays at
   3 x 3 x 3 x 3 = 81 and the agent code doesn't change shape. GloVe is dropped from the
   download script; the prompt-injection fixture now targets a `bge` config.

## Risks

- Bigger search space: the agent's regret and "within 0.01" rates may drop; reported honestly.
- `bge` latency on CPU may push it out of tight-latency scenarios.
- SQuAD favours lexical matching, so BM25 may still win retrieval. That's a finding, not a failure;
  a second dataset is out of scope for v2.

---

# Plan v3: build the agents with LangGraph, CrewAI and MCP (draft, Nov 19)

Status: **approved Nov 19 ("go for v3, don't stop until you need my input"); Steps 1–5 done Nov 19.**

## Why

The capstone must use the frameworks taught in the course, and Checkpoints 4–5 committed to
them (CP4 §2d: CrewAI Planner and Evaluator, a framework controller, an MCP server for shared
state). v2 implements the CP4 *algorithm* faithfully (beam 3, depth 4, 30-node budget, rubric
50/20/15/15, 0.60 floor, 15% gap, 0.02 ties, backtracking) but in plain Python with rule-based
agents. v3 keeps that algorithm and moves it onto the frameworks.

## Design

| piece | framework | what it does |
|---|---|---|
| Controller (orchestrator) | **LangGraph** `StateGraph` | Nodes: clarify → gather evidence → expand → test → evaluate/prune → (loop until depth 4) → validate finalists → report. Conditional edges for backtracking, re-planning and stopping. Human checkpoints with LangGraph `interrupt()` and a checkpointer, so an interactive run pauses and resumes |
| Researcher, Planner, Evaluator (critic) | **CrewAI** agents on Claude Haiku 5.5 | Researcher gathers cited evidence; Planner writes each Thought (why each of the 3 options, which untested choices to fill) and the final report; Evaluator reviews the measured scores, flags evidence that conflicts with experiments and checks citations |
| Experimenter | CrewAI tool, no LLM | Runs the pipeline; an LLM adds nothing to running code, so it stays a tool (keeps cost down) |
| Shared state | **MCP server** (`mcp` Python SDK, FastMCP) | Exposes `search_research`, `search_experiments`, `run_experiment`, `get_tree` to the CrewAI agents (read-only except the Experimenter's log write) |

Rules kept from CP4/CP6: **numbers come only from measured experiments.** LLM agents reason and
write, but scores, pruning and constraint checks stay in code, and every number in a report is
still verified against the experiment log. If an LLM answer is invalid, refused or over budget,
the run falls back to the rule-based policy for that step and records it (CP6 fallback logic).

Two modes: `--agents llm` (CrewAI, the capstone system) and `--agents rule` (v2 policy, now
running inside the same LangGraph graph). Rule mode is the free, repeatable baseline and the
ablation; tests use it plus a fake LLM.

## Steps (each a separate go/no-go)

1. ✅ **LangGraph controller, rule mode.** Done: `graph.py`; identical decisions to the v2 loop on
   13 deterministic scenarios incl. low-confidence and denied steps (`tests/test_graph.py`).
   Original plan: Rewrite `orchestrator.run` as a graph that calls the
   existing agent code. Check it picks the same configurations as v2 on all five scenarios.
   Files: new `rag_optimizer/graph.py`, `orchestrator.py`, `human.py`, `cli.py`, tests,
   `requirements.txt` (`langgraph`). $0.
2. ✅ **MCP server.** Done: 3 read-only tools (`run_experiment` left out: the Experimenter is
   a tool the graph calls directly), stdio test incl. injection filter. Original plan: New `rag_optimizer/mcp_server.py` with the four tools; test it with an MCP
   client. $0.
3. ✅ **CrewAI agents.** Done: Researcher, Planner, Evaluator critic; fake-LLM tests. CrewAI 1.15
   sends `budget_tokens: null` with disabled thinking (API 400): worked around. Original plan: New `rag_optimizer/crew.py`: the three LLM agents, structured (Pydantic)
   outputs, fallback, and their token usage added to the existing $1 spend ledger. One smoke
   run (~$0.005, estimate). Files: `crew.py`, `graph.py`, `reader.py` (shared ledger), tests,
   `requirements.txt` (`crewai`, `mcp`).
4. ✅ **Evaluate LLM mode vs rule mode.** Done (`results/llm_agents_eval.json`): CrewAI regret
   S1 0.0131 / S2 0.0673 / S3 0.0000 / S4 0.0129 / S5 no pick (finalists over the 0.5 ms cap on
   validation) vs rules 0.0074 / 0.0084 / 0.0031 / 0.0061 / 0.0025; 0 fallbacks; ~$0.006 and
   ~45 s per run vs $0 and ~4 s. Rule suite re-run through LangGraph: regret 0.0055, within 0.01
   92% (v2 0.0101 / 64%; same decisions given same measurements, so the change is timing noise).
   Safety in LLM mode passed. Project LLM spend $0.161 of $1. Original plan: 5 scenarios × 2 repeats + the prompt-injection and
   human-checkpoint tests in LLM mode; the full 100-run suite stays in rule mode. Compare regret,
   experiments, time and cost. Estimate ~$0.05–0.10.
5. ◐ **Write-up.** README, diagram, push done Nov 19; the 7.1 report template is next.

## Cost (estimate)

~9 Haiku calls per LLM-mode run (Planner and Evaluator once per depth, Researcher, report) at
~3K input / ~300 output tokens (CrewAI prompts are verbose) ≈ $0.004 per run. ~12 evaluation runs
plus development ≈ $0.10. Already spent $0.0725, so the project stays under the **$1 cap**, which
the code enforces.

## Risks

- CrewAI is a large dependency; Python 3.13 support must be checked in Step 3.
- Haiku 5.5 rejects non-default `temperature`; CrewAI/LiteLLM must not send it.
- LLM mode varies between runs, so its results are reported as ranges over repeats.
- Estimated effort ~8–10 h.
