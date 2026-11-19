---
id: kb09
title: Chunk size trade-off (practitioner heuristic, no single source)
source: Design heuristic; no peer-reviewed source. Treat as weak evidence.
supports: chunk_size=128:+1:0.3
---
Smaller chunks make retrieval more precise but can split an answer from its context. Larger
chunks keep more context but raise the number of tokens sent to the generator, and with it the
cost and latency. A medium chunk size is a common starting point to be tested, not assumed.
