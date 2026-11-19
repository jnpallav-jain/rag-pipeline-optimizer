---
id: kb04
title: Sentence-BERT - averaged word vectors are a weak sentence embedding
source: Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
supports: retriever=glove:-1:0.7
---
Averaging static word embeddings such as GloVe gives a simple sentence embedding, but it scored
well below trained sentence encoders on semantic similarity benchmarks. Untrained averaged
embeddings are best treated as a weak dense baseline for retrieval.
