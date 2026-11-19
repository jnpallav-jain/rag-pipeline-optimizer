---
id: kb02
title: BEIR - heterogeneous zero-shot retrieval benchmark
source: Thakur, N. et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks.
supports: retriever=bm25:+1:0.8
---
Across 18 retrieval datasets, BM25 proved a robust zero-shot baseline for passage retrieval. Many
dense retrievers trained on one domain underperformed BM25 when applied to other domains,
while re-ranking with a cross-encoder gave the best results at higher computational cost.
