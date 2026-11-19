---
id: kb06
title: Reciprocal Rank Fusion
source: Cormack, G., Clarke, C. and Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
supports: rerank=rrf:+1:0.6
---
Reciprocal rank fusion combines several ranked lists by summing 1/(k + rank) for each document.
In the original experiments it outperformed the individual rankers it fused. Hybrid search
commonly fuses a lexical ranker with a dense ranker this way. Fusion works best when the
rankers being combined are each reasonably strong.
