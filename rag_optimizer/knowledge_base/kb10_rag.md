---
id: kb10
title: Retrieval-augmented generation
source: Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
supports:
---
RAG combines a retriever with a generator so answers are conditioned on retrieved passages.
The generator can only use what the retriever returns, so whether the retrieved context contains
the answer is an upper bound on answer quality and a useful measure of retrieval quality.
