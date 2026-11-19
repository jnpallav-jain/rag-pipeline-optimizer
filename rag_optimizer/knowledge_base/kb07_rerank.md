---
id: kb07
title: Second-stage re-ranking of retrieved candidates
source: Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085.
supports: rerank=overlap:+1:0.3
---
Re-ranking the top candidates from a first-stage retriever with a stronger relevance model
substantially improved passage ranking on MS MARCO. The evidence is for learned cross-encoder
re-rankers; a lightweight lexical re-ranker is only a cheap stand-in, so this support is weak.
