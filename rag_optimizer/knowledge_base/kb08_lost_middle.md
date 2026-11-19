---
id: kb08
title: Lost in the Middle - more context is not always better
source: Liu, N. F. et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL.
supports: top_k=3:+1:0.4; top_k=5:-1:0.2
---
Language models used relevant information best when it appeared at the beginning or end of the
context and worse when it was in the middle. Adding more retrieved documents gave diminishing
returns for question answering, so a moderate number of retrieved passages is a sensible default.
