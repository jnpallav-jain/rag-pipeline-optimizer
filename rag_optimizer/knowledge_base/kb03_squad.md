---
id: kb03
title: SQuAD - questions written against the passage
source: Rajpurkar, P. et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP.
supports: retriever=bm25:+1:0.5; retriever=tfidf:+1:0.4
---
SQuAD questions were written by crowdworkers while reading the Wikipedia paragraph that contains
the answer. As a result, questions often share many words with the answer sentence, so lexical
retrievers such as BM25 and TF-IDF tend to find the right passage for this dataset.
