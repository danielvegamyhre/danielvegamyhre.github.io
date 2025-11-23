---
layout: post
title:  "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
date:   2024-12-14 12:45:51 -0700
categories: ml performance
---

For session 2 of the EleutherAI ML Scalability & Performance reading group, I co-presented a talk on the paper "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." 

Another member of the reading group (Ben) presented an overview of the theory, and I presented my Triton kernel implementation of Flash Attention 2. 

The code can be found [here](https://github.com/danielvegamyhre/ML-scalability-and-performance/blob/main/session_2/flash_attention.py).


Papers:

1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2307.08691)

2. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

Note: you may have to disable ad blocker for the YouTube player to render correctly. Alternatively, you can watch the recording directly on YouTube [here](https://youtu.be/Lys0TpsLIEc?si=WPOQmsTo09gYjXRR&t=1739).

<iframe width="560" height="315" src="https://www.youtube.com/embed/Lys0TpsLIEc?si=Wr3kMxElfSa_Phkg&amp;start=1739" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>