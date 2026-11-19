---
id: kb04
title: Sentence-BERT - trained sentence encoders beat averaged word vectors
source: Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
supports: retriever=bge:+1:0.4
---
Averaging static word embeddings such as GloVe gives a simple sentence embedding, but it scored
well below sentence encoders trained with a siamese objective on semantic similarity benchmarks.
Trained sentence encoders are the stronger choice for dense retrieval with embeddings.
