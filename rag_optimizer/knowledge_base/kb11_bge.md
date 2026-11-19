---
id: kb11
title: BGE - general-purpose embedding models for dense retrieval
source: Xiao, S. et al. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE models). arXiv:2309.07597.
supports: retriever=bge:+1:0.6
---
The BGE embedding models are trained with large-scale contrastive pre-training followed by
fine-tuning on labelled query-passage pairs, and ranked near the top of the MTEB retrieval
benchmark when released. Dense sentence embeddings from such general-purpose models transfer
across domains much better than untrained word-vector averages; the small variant trades some
accuracy for faster embedding of questions and passages.
