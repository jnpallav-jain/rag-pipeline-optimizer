# End-to-end results: retrieval + Claude Haiku 5.5 reader

200 questions sampled (seed 7) from the held-out test split. Recall is the share of questions whose gold answer appears in the retrieved context; EM and F1 are the official SQuAD metrics on the model's answer. *EM when retrieved* counts only questions whose answer was in the context, so it measures the reader alone.

| config | why | recall | EM | F1 | EM when retrieved | 'not found' | input tokens | cost |
|---|---|---|---|---|---|---|---|---|
| `c256-bm25-k1-rrf` | top-k sweep (k=1) | 0.855 | 0.705 | 0.788 | 0.825 | 0.155 | 624 | $0.0067 |
| `c256-bm25-k3-rrf` | top-k sweep (k=3) | 0.935 | 0.750 | 0.844 | 0.802 | 0.070 | 1685 | $0.0173 |
| `c256-bm25-k5-rrf` | top-k sweep (k=5) | 0.950 | 0.775 | 0.847 | 0.816 | 0.070 | 2739 | $0.0279 |
| `c128-bm25-k1-none` | agent pick, S1_tight | 0.815 | 0.690 | 0.742 | 0.847 | 0.195 | 361 | $0.0040 |
| `c64-tfidf-k3-overlap` | agent pick, S3_generous | 0.845 | 0.705 | 0.769 | 0.834 | 0.160 | 498 | $0.0054 |
| `c128-bm25-k3-none` | agent pick, S4_accuracy_first | 0.885 | 0.730 | 0.817 | 0.825 | 0.105 | 894 | $0.0094 |

Total LLM spend for the project so far: $0.0725 over 1220 calls (1,378,059 input / 10,745 output tokens, as reported by the API). Hard cap $1.00.
