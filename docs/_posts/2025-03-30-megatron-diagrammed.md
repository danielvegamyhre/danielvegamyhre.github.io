---
layout: post
title:  "A diagrammed walkthrough of Megatron-style tensor parallelism"
date:   2025-03-30 12:45:51 -0700
categories: ml performance
---
<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>

The paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) is a seminal work in ML performance research, and is a must-read for anyone working in this domain. It introduces tensor parallelism as a new technique which paritions the computation of certain transformer layers across accelerators such that: 

1) The activations are smaller, reducing peak memory usage and allowing larger models to be trained. Activations often dominate peak memory usage in very large models, so reducing activation memory required to train larger models is important.

2) The activations remain sharded for as long as possible before synchronizing (which must be done to ensure the mathematical integrity of the training process), to minimize this communication overhead between devices, which can slow down training and become a bottleneck.

The paper itself provides some diagrams to help the reader understand the mathematics presented, but I found them to be a bit lacking in conveying the nuances of why certain sharding strategies are better than others, and overall my own understanding became much more concrete when I diagrammed everything out myself. Since this was helpful for me, I'm hoping it may be helpful for others as well.

The paper presents sharding strategies for 4 types of layers: the MLP block, the multi-head attention layer (MHA), the input embeddings and the output embeddings. As in the paper, we'll start with the MLP block.

## Sharding the MLP blocks

At the time this paper was written, the standard MLP block following the attention layers in transformers consisted of 2 linear layers with a non-linearity between them [^1]. This can be represented mathematically like so:

$$
    Y = GeLU(XA) 
$$

$$
    O = YB
$$

where `X` are our input activations, `A` is the first linear projection, and `B` is the second linear projection.

Fundamentally, these linear layers will each require 3 GEMM operations: one in the forward pass, and two in the backward pass:

$$
    O = XW
$$

$$ 
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial O}W
$$

$$
\frac{\partial L}{\partial W} = \left[\frac{\partial L}{\partial O} \right]^T X
$$

The output size of these GEMM operations will be based on the dimensions of the input `X` and the weights `W`:

$$
    X \in \mathbb{R}^{M \times K}, \quad W \in \mathbb{R}^{K \times N} \quad \Rightarrow \quad XW \in \mathbb{R}^{M \times N}
$$

 In the linear projections of the MLP blocks in the GPT-2 style transformer models studied in this paper, `M`, `K`, and `N` are *very large* (by modern standards they're actually small, but let's read the paper in the context it was written in). 
 
 Thus, storing all of the intermediate output activations of these GEMMs will be extremely memory intensive. Unless we find a way to reduce this excessive activation memory, we'll be unable to do research on larger models, due to the current physical limits of HBM capacity on GPUs/TPUs (in this paper, the authors used NVIDIA V100s with 32GB of HBM).

Thus is born the motivation for the authors to explore reducing activation memory by *sharding* the matrices involved in these GEMMS across multiple devices. By sharding the computation across devices, each device holds smaller sub-matrices and thus produces smaller  activations. 

<!-- Fundamentally, these linear layers will each require a GEMM operation. These GEMM operations are computationally expensive - in asymptotic notation, the time complexity of a 2D GEMM operation between matrices of shape `(M,K)` and `(K,N)` is $$O(M \cdot K \cdot N)$$. Put another way, the computation required to compute the GEMM grows *non-linearly* with respect to the matrix dimensions. If we increase `M` by 1, the number of FMA (fused multiply add operations used in dot prdocuts) we need to perform grows by `K * N`. 

Given the MLP blocks in large transformer models are often have very large dimensions, $$ M \cdot K \cdot N $$ can be very large. Furthermore, we usually have a batch dimension on our inputs, meaning our matmul shapes will be:

`Y = (batch size, sequence length, hidden dim) @ (hidden_dim, 4 * hidden_dim) = (batch size, sequence length, 4 * hidden_dim)`.

In this paper, they used GPT-2 for experiments with:

- Batch size: 512
- Sequence length: 1024
- Hidden size: 3072

This model is small by modern standards, so you can multiply these out yourself and see this will require *a lot* of compute - and this is just for the forward pass!

The backward pass for a linear layer requires 2 more GEMMs:

$$ 
\frac{\partial L}{\partial X} = GW
$$

$$
\frac{\partial L}{\partial W} = G^T X
$$

where `G` is our upstream gradient, `X` is our input activations and `W` is our weights.

Suffice to say, a lot of compute is required for this! -->

### 1st GEMM - the bad option

There are a couple of ways we could shard `X` and `A` to reduce the size of output activation. One obvious way is to shard `X` column-wise and `A` row-wise:

$$
    \mathbf{X} = [X_1, X_2], \quad \mathbf{A} = 
    \begin{bmatrix} 
    A_1 \\ 
    A_2 
    \end{bmatrix}
$$

Conceptually, the math above can be visualized like so:

<img src="/images/megatron-diagrams/MLP-1st-GEMM-bad-option-stacked-layout.png" alt="MLP-1st-GEMM-bad-option" style="width: 100%">

As shown in the diagram above, this option is not ideal because to compute the *complete* results of any output element in the output matrix, we would need to sum the *partial* results on each accelerator. This means we already would need an all-reduce operation across all N devices in the tensor parallel group - after only doing the 1st GEMM in the MLP layer! Since we're trying to minimize the communication overhead by keeping these computations independent on each device for as long as possible, this is probably not ideal, so we should evaluate other options.

However, you might ask: do we necessarily *have* to all-reduce here? Why can't we keep the partial results on each device, continue on with applying GeLU to each set of activations individually, do the GEMM for the next linear layer, and then combine these partial outputs via all-reduce at the end?

The answer is because we need this *partioned* version of the activation function (left above) to be mathematically equivalent to the original, *non-sharded* version of the activation function. Otherwise, the integrity of the numerics will be comprised and we'll run into things like convergence problems, training instability, and so on. In other words: the math will be wrong.

To specific, we can't perform the GeLU non-linearity on the partial results and sum later because non-linearities like GeLU do not have the distributive property:

$$
    GeLU(X_1 A_1) + GeLU(X_2 A_2) \ne GeLU(X_1 A_1 + X_2 A_2) 
$$

Here is an example demonstrating this with a simpler non-linearity (ReLU), and scalar values instead of matrices:

$$
\text{ReLU}\left( (-1 \cdot 2) + (1 \cdot 1) \right) = \text{ReLU}(-2 + 1) = \text{ReLU}(-1) = 0

$$

vs

$$
    ReLU(-1 \cdot 2) + ReLU(1 \cdot 1) = ReLU(-2) + ReLU(1) = 0+1 = 1 
$$

### 1st GEMM - the good option

So, given that we'd have to perform an all-reduce immediately to maintain mathematical fidelity with the non-sharded computation, let's examine another option: *not sharding* the input activations `X`, and sharding the weight matrix `A` column-wise:


$$
    [Y_1, Y_2] = [GeLU(XA_1), GeLU(XA_2)]
$$

<img src="/images/megatron-diagrams/MLP-1st-GEMM-good-option-stacked-layout.png" alt="MLP-1st-GEMM-bad-option" style="width: 100%">

With this approach, there are some immediately obvious benefits:

1) The output activations are still smaller than the original non-sharded version by a factor of $$\frac{1}{\text{number of accelerators}}$$ - nice!

2) Since we have *complete* results for each element of the output matrix on a given device, no summation/reduction operations are necessary and we can apply GeLU directly to these outputs, while maintaining mathematical fidelity with the non-sharded computation - super nice! :fire:

With this approach, the activations from the first linear layer now stay paritioned column-wise through the GeLU and pass into the 2nd GEMM.

### 2nd GEMM

For this final step in the MLP block, there's no way to avoid synchronization any further:

- Given the activations $$Y$$ are sharded column-wise and the activations must be the left operand in the next GEMM $$O = YB$$, we can only shard the weights row-wise, so that the number of columns in the left operand (activations) match the number of rows in the right operand (weights) on each device, so we can complete a standard dot product operation. However, the resulting output matrices $$ [O_1, O_2] = [Y_1 B_1, Y_2 B_2] $$ will contain *partial* results that must be summed across devices before going through the next layer - dropout.

- Matrix multiplication does not have the communiative property ($$AB \ne BA$$). Therefore, we can't swap around the order of our GEMM operands to make the current column-wise sharding of the activations more favorable, as the mathematics would diverge from the original, non-sharded computation.

Between the two options of sharding the weight matrix $$B$$ row-wise or column-wise 

The sharded activations flow directly through the 2nd GEMM, where the weights $$B$$ of the 2nd linear weight matrix are sharded row-wise across devices. 

$$ 
    [O_1, O_2] = [Y_1 B_1, Y_2 B_2]
$$

Now we have a shard of the complete outputs of the MLP block on each device. We must now (finally) perform an all-reduce to get complete MLP block outputs on each device, in order to go through the dropout layer next. 


<img src="/images/megatron-diagrams/2nd-GEMM-stacked-layout.png" alt="MLP-1st-GEMM-bad-option" style="width: 100%">

It's important to remember when we do a collective in the forward pass, we'll need to peform the *inverse* of the collective in the backward pass, to propagate the gradient to all relevant inputs, or reduce the gradient from all relevant outputs.

In this case, the all-reduce operation in the forward-pass will become a identity operation (i.e., a no-op) of the upstream gradient across devices.

Conversely, since our input activations to the MLP block were not partitioned in the forward pass (i.e., identity operator), this will become an all-reduce in the backward pass when we need to propagate the gradients from each shard of the computation through to the previous layer. This way our reduced (summed) gradients are exactly equivalent to the gradients of a non-partitioned version of this MLP block.

**Interesting side note**: In the Megatron paper, the dropout computation is duplicated on each device (i.e., not sharded). The authors don't say much about why this is, I assume it's because these layers are cheap computationally and it was not clear (at the time) that attempting to shard these layers would have a favorable "memory reduction vs communication overhead" trade-off. However, this sets the stage for a future paper, [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), which observed that layers in the non-TP regions of the transformer (namely dropout and layer normalization) do not require much computation but *do* require a lot of activation memory, making them a potentially juicy target for optimization. They also observed the computation for these layers can be performed *independently along the sequence dimension* without violating the mathematics - meaning theoretically, they can shard along the sequence dimension and potentially reduce activation memory per device, thus avoiding the need to recompute activations in the backward pass to train larger models. If you're interested in this, I presented this paper at the Eleuther AI ML Scalability & Performance reading group, which you can check out the recording for [here](https://danielvegamyhre.github.io/ml/performance/2025/03/23/eleutherai-reading-group-session-9.html).

Anyway, now that we understand how the MLP block is sharded and *why*, we're ready to move onto the mult-head attention layer.

## Sharding the multi-head self-attention layers

Coming soon...

[^1]: Nowadays, FFNs with a slightly different structure are often used (see [Llama3](https://arxiv.org/abs/2407.21783) models as an example).

