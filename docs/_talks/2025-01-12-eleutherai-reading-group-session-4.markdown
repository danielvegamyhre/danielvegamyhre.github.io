---
layout: post
title:  "EleutherAI ML Perf Reading Group: Ring Attention"
date:   2025-01-12 12:45:51 -0700
categories: ml performance
---

For session 4 of the EleutherAI ML Scalability & Performance reading group, I gave a presentation covering the seminal paper "Ring Attention with Blockwise Transformers for Near-Infinite Context Length."

I also cover 2 key pieces of prior work which provide the foundation for ring attention, to understand what the limitations were of those prior approaches and how ring attention built on them to unlock massive gains in max sequence length for transformer models.

Papers:

1. [Sequence Parallelism: Long Sequence Training from System Perspective](https://aclanthology.org/2023.acl-long.134.pdf)
2. [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370)
3. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

Note: you may have to disable ad blocker for the YouTube player to render correctly. Alternatively, you can watch the recording directly on YouTube [here](https://www.youtube.com/watch?v=fC9L8J7dVFI).

<iframe width="560" height="315" src="https://www.youtube.com/embed/fC9L8J7dVFI?si=Ay7q1fhkpXJ7HaXx" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>