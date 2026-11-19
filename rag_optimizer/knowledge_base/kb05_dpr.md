---
id: kb05
title: Dense Passage Retrieval needs training data; BM25 stronger on SQuAD
source: Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
supports: retriever=bm25:+1:0.8; retriever=glove:-1:0.5
---
A dense retriever trained on question-passage pairs outperformed BM25 on top-20 passage retrieval
accuracy for several open-domain QA datasets. On SQuAD, however, BM25 performed better, which the
authors attribute to the high lexical overlap between SQuAD questions and their passages.
Dense retrieval gains depend on training the encoder for the task.
