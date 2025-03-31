---
layout: post
title:  "A diagrammed walkthrough of Megatron-style tensor parallelism"
date:   2025-03-30 12:45:51 -0700
categories: ml performance
---
<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>

In this post, I attempt to provide a detailed walkthrough of Megatron-style tensor parallelism, with diagrams to help make the concepts and mathematics more digestible. The target audience for this post is readers who are already familiar with ML and the transformer architecture, who wish to deepen their understanding of tensor parallelism.

The paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) is a seminal work in ML performance research, and is a must-read for anyone working in this domain. It introduces tensor parallelism as a new technique which paritions the computation of certain transformer layers across accelerators such that: 

1) The activations are smaller, reducing peak memory usage and allowing larger models to be trained. Activations often dominate peak memory usage in very large models, so reducing activation memory required to train larger models is important.

2) The activations remain sharded for as long as possible before synchronizing (which must be done to ensure the mathematical integrity of the training process), to minimize this communication overhead between devices, which can slow down training and become a bottleneck.

This post will be divided into 4 sections, with some broken down into more digestable sub-sections:
1. [MLP blocks](#mlp-blocks)
    - [1st GEMM of the MLP block forward pass - the bad option](#1st-gemm-of-the-mlp-forward-pass-the-bad-option-)
    - [1st GEMM of the MLP block forward pass - the good option](#1st-gemm-of-the-mlp-forward-pass-the-good-option-)
    - [2nd GEMM of the MLP block](#2nd-gemm-of-the-mlp-forward-pass)
    - [Optional attention review](#optional-attention-review)
2. [Attention layers](#attention-layers)
3. [Input embeddings](#input-embeddings)
4. [Output embeddings and fused cross-entropy loss](#fourth-examplehttpwwwfourthexamplecom)

## MLP blocks

At the time this paper was written, the standard MLP block following the attention layers in transformers consisted of 2 linear layers with a non-linearity between them [^1]. This can be represented mathematically like so:

$$
    Y = GeLU(XA) 
$$

$$
    O = YB
$$

where `X` are our input activations, `A` is the first linear projection, and `B` is the second linear projection.

Fundamentally, these linear layers will *each* require 3 GEMM operations.

One in the forward pass: 

$$
    C = AB
$$

And two in the backward pass:

$$ 
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}B
$$

$$
\frac{\partial L}{\partial B} = \left[\frac{\partial L}{\partial C} \right]^T A
$$

The output size of these GEMM operations will be based on the dimensions of the inputs `A` and the weights `B`:

$$
    A \in \mathbb{R}^{M \times K}, \quad B \in \mathbb{R}^{K \times N} \quad \Rightarrow \quad AB \in \mathbb{R}^{M \times N}
$$

 In the linear projections of the MLP blocks in the GPT-2 style transformer models studied in this paper, `M`, `K`, and `N` are *very large* (by modern standards they're actually small, but let's read the paper in the context it was written in). 
 
 Thus, storing all of the intermediate output activations of these GEMMs will be extremely memory intensive. Unless we find a way to reduce this excessive activation memory, we'll be unable to do research on larger models, due to the current physical limits of HBM capacity on GPUs/TPUs (in this paper, the authors used NVIDIA V100s with 32GB of HBM).

Thus is born the motivation for the authors to explore reducing activation memory by *sharding* the matrices involved in these GEMMS across multiple devices. By sharding the computation across devices, each device holds smaller sub-matrices and thus produces smaller  activations. 

**Note**: the authors focused on the forward pass when describing the paritioning scheme of the MLP block, so we'll do the same here, but the same concepts outlined below apply to the backward pass as well.

#### **1st GEMM of the MLP forward pass**: *the bad option* ❌


There are a couple of ways we could shard `X` and `A` to reduce the size of output activation. One obvious way is to shard `X` column-wise and `A` row-wise. For example, sharding `X` and `A` across a tensor parallel group of 2 devices:

$$
    \mathbf{X} = [X_1, X_2], \quad \mathbf{A} = 
    \begin{bmatrix} 
    A_1 \\ 
    A_2 
    \end{bmatrix}
$$

Conceptually, the math above can be visualized like so (**same-colored arrows** represent dot products occuring locally on a device):

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

#### **1st GEMM of the MLP forward pass**: *the good option* ✅

So, given that we'd have to perform an all-reduce immediately to maintain mathematical fidelity with the non-sharded computation, let's examine another option: *not sharding* the input activations `X`, and sharding the weight matrix `A` column-wise:


$$
    [Y_1, Y_2] = [GeLU(XA_1), GeLU(XA_2)]
$$

<img src="/images/megatron-diagrams/MLP-1st-GEMM-good-option-stacked-layout.png" alt="MLP-1st-GEMM-bad-option" style="width: 100%">

With this approach, there are some immediately obvious benefits:

1) The output activations are still smaller than the original non-sharded version by a factor of $$\frac{1}{\text{number of accelerators}}$$ - nice!

2) Since we have *complete* results for each element of the output matrix on a given device, no summation/reduction operations are necessary and we can apply GeLU directly to these outputs, while maintaining mathematical fidelity with the non-sharded computation - super nice! :fire:

With this approach, the activations from the first linear layer now stay paritioned column-wise through the GeLU and pass into the 2nd GEMM.

#### **2nd GEMM of the MLP forward pass**

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

Now, you may notice that the dropout layer (and layer norm, which is not pictured) are performing redundant computation on every device: after the all-reduce, all devices have the same output activations, and thus applying dropout and layer norm to them will be identical. These particular layers were not the subject of this paper, but resolving this inefficiency was later explored in a subsequent paper by NVIDIA [^2].

To recap:
- In the MLP block forward pass, we do only one all-reduce at the end, before the dropout layer. This becomes an identity op in the backward pass.
- In the forward pass, the input activations are not sharded, so this becomes an all-reduce in the backward pass.
- In total, for each MLP block in the transformer, there will be a total of 2 all-reduces: one in the forward pass, and one in the backward pass.

Now that we understand how the MLP block is sharded and *why*, we're ready to move onto the mult-head attention layer.

### Optional attention review

As a reminder, the multi-head self-attention layer involves the following steps:

1. Our input activations $$X$$ are projected through parameter matrices $$W_i^Q$$, $$W_i^K$$, and $$W_i^V$$ to get our queries, keys, and values for each attention head (denoted by the "i" subscript).

$$
    Q_i = XW_i^Q
$$

$$
    K_i = XW_i^K
$$ 

$$
    V_i = XW_i^V
$$

Where:
- $$X \in \mathbb{R}^{B \times S \times H}$$ are the input activations
- $$W_i^Q \in \mathbb{R}^{d_{\text{hidden}} \times d_q}$$ is the query parameter matrix for head $$i$$.
- $$W_i^K \in \mathbb{R}^{d_{\text{hidden}} \times d_{kv}}$$ is the key parameter matrix for head $$i$$.
- $$W_i^V \in \mathbb{R}^{d_{\text{hidden}} \times d_{kv}}$$ is the value parameter matrix for head $$i$$.
- $$d_q$$ is the dimension of queries
- $$d_{kv}$$ is the dimension of keys and values
- $$d_{\text{hidden}}$$ is the hidden dimension

Our queries, keys, and values are used for scaled dot product attention for each attention head:

$$
\text{head}_i(Q, K, V) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_{kv}}}\right)V_i
$$

Finally, we concatenate our attention head outputs and project them back to the hidden dimension:

$$
\text{MultiHead}(\text{head}_1, \ldots, \text{head}_h) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

Where:
- $$W^O \in \mathbb{R}^{d_{kv} \times d_{\text{hidden}}}$$ is the output projection matrix

And now all of our tokens position in embedding space has been updated by aggegregating the weighted updates provided by each other token in the context, with weights based on the relevance of each token.


## Sharding the multi-head self-attention layers

*Important note*: In this paper, the authors explore how to parallelize a *vanilla* multi-head attention layer. However, it is important to note that the sharding scheme described here is fairly generic and is composable with attention variants such as MQA, GQA, and [ring attention](https://danielvegamyhre.github.io/ml/performance/2025/01/12/eleutherai-reading-group-session-4.html), and MLA. There may be slight differences, such as in MQA needing to do redundant projections of $$K$$ and $$V$$ on each device before applying TP. 

So, how do we parallelize the attention layer? It is actually somewhat straight-forward. The reason for this is that MHA has the convenient property that the computation within each attention head is completely self-contained. This makes it easy to parallelize across the `num_heads` dimension.

Thus we can shard the $$W^Q$$ $$W^K$$  $$W^V$$ parameter matrices column-wise across the `num_heads` dimension, as shown in the diagram below. These will operate on our non-sharded input activations which will be coming a previous layer norm layer, which as mentioned previously, is NOT partioned in any way, so each device has duplicate activations from this layer present on it at this point:

<img src="/images/megatron-diagrams/QKV_projections.png" alt="QKV projections" style="width: auto;">

Now that we have our $$Q_i$$, $$K_i$$, and $$V_i$$ projections for each head, we can perform scaled dot product attention for each head locally on each device to get our attention activations for that head, $$Y_i$$:

<img src="/images/megatron-diagrams/attention.png" alt="sdpa" style="width: auto;">

With the attention activtions for each head, we can now pass through the final linear projection $$W^O$$, which has been partitioned *row-wise* across devices. As shown in the [attention review](#optional-attention-review) above, the $$W^O$$ projection is normally applied to the **concatenated** attention heads in the typical unsharded computation. So to maintain mathematical fidelity with the unsharded computation, we now need to all-reduce the outputs before proceeding with the dropout layer:

<img src="/images/megatron-diagrams/attention-linear-output.png" alt="sdpa" style="width: auto;">

One natural question may arise at this point: why are we doing an all-reduce here and not an all-gather here? We parallelized along the `num_heads` dimension, and normally in single-device training we concatenate the attention heads, so wouldn't the analgous thing to do in mult-device training be to all-gataher the head outputs together? 

Let's look  carefully at the shapes involved in the computation to figure out why this is.

The local linear projection for each head on each device is:

$$O_i = Y_i W_i^O$$

Where:

- $$Y_i \in \mathbb{R}^{B \times S \times d_{kv}}$$
are the attention activations.

- $$W_i^O \in \mathbb{R}^{d_{kv} \times d_{hidden}}$$ is the linear output projection.

So the dimensions of the output $$O_i$$ will be $$ \mathbb{R}^{B \times S \times d_{hidden}} $$. Now we can ask ourselves, how would this output shape be different for all the attention heads, instead of just one? Answer: it wouldn't, none of the dimensions are dependent on the number of heads. So what the output projection is doing is basically performing an aggregated projection using the information in all the attention heads to project the tokens back into the hidden dimension. The diagram below helps illustrate this:


<img src="/images/megatron-diagrams/why-not-all-gather-attention-heads.png" alt="sdpa" style="width: auto;">

So by parallelizing along the `num_heads` dimension, we will have the same shaped output activation on multiple devices, each containing only *partial* results. Therefore, we need to all-reduce to aggregate the results (i.e., the updates to our tokens' positions in embedding space as dictated by the aggregated updates present in the attention head outputs).

...and that's it! To recape:

- In the attention layer forward pass, we do only one all-reduce in the forward pass, at the end of the attention layer. This becomes an identity operator (no-op) in the backward pass.
- In the attention layer forward pass, our input activations are not sharded, so this becomes an all-reduce in the backward pass, for the same reasons as described in the MLP section.
- In total, for each attention layer in the model, we'll have 2 all-reduces: one in the forward pass and one in the backward pass.

## Input embeddings

The way the input embeddings are sharded is a bit unintuitive. To review, the input embeddings are a matrix $$ E_{input} \in \mathbb{R}^{V \times {H}}$$ where $$V$$ is the vocabulary dimension and $$H$$ is the hidden dimension. The embedding matrix basically stores learnable parameters representing the tokens "original" position in embedding space, before any updates to its position (based on the surrounding context) are applied through the various attention layers.

Sharding the input and output embedding matrices is beneficial because the vocabulary size can be quite large (in the paper, it was 50,127 and padded to be 51,200 - the next multiple of 8 - for more efficient GEMMs on the hardware). Since the hidden dimension in this paper is 3072 for the largest model tested (GPT-2 8.2B) the size of the input embedding is:

 51,200 tokens * 3072 hidden dimension size = 157,286,400 * 2 bytes per parameter in bfloat16 = 314,572,800 bytes or ~315MB. This can even be much larger in modern models [^3], so this is good motivation to shard the embedding matrix if possible, to reduce memory pressure and allow other useful things to use that memory.

To shard this embedding matrix, we can do so either row-wise (along the hidden dimension) or column-wise (along the vocabulary dimension).

Sharding along the hidden dimension would require doing an all-gather before going through layer norm (which normalizes along the full hidden dimension). This is not ideal, since all-gather requires moving more data around between devices than all-reduce, which we've gotten away with using so far. 

For this reason, it turns that by sharding along the vocabulary dimension, we *can* get away with just using an all-reduce. However, it is a bit unintuitive how this works, so let's take it step by step.

Sharding along the vocabulary dimension would result in each device having a subset of the token embeddings. At first this seems problematic: our full raw input tokens will arrive on each device in parallel, with shape $$T \in \mathbb{R}^{S \times V}$$ where $$S$$ is the sequence length and $$V$$ is the vocabulary size. How could we handle tokens whose embeddings do not exist on the local device?

We can handle this by simply assigning `0` as the embedding for any token whose embedding does not exist on the device. This scalar `0` can broadcast along the embedding dimension. 

So the output of each input embedding on the i-th device is of shape $$Y_i \in \mathbb{R}^{B \times S \times H}$$, where some of the tokens along the sequence dimension `S` have a scalar 0 which broadcasts along the hidden dimension.

Then we can do an all-reduce to get the full token embeddings on each device, with the same shape, but no empty 0 vectors!

This is shown in the diagram below:


<img src="/images/megatron-diagrams/input-embedding.png" alt="sdpa" style="width: auto;">

To make things even more concrete, let's take a look at the PyTorch implementation of [_MaskPartial](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_ops/_embedding_ops.py#L70).

## Output embeddings and fused cross-entropy loss


# References
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

# Footnotes
[^1]: Nowadays, FFNs with a slightly different structure are often used (see [Llama3](https://arxiv.org/abs/2407.21783) models as an example).

[^2]: In the Megatron paper, the dropout computation is duplicated on each device (i.e., not sharded). The authors don't say much about why this is, I assume it's because these layers are cheap computationally and it was not clear (at the time) that attempting to shard these layers would have a favorable "memory reduction vs communication overhead" trade-off. However, this sets the stage for a future paper, [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), which observed that layers in the non-TP regions of the transformer (namely dropout and layer normalization) do not require much computation but *do* require a lot of activation memory, making them a potentially juicy target for optimization. They also observed the computation for these layers can be performed *independently along the sequence dimension* without violating the mathematics - meaning theoretically, they can shard along the sequence dimension and potentially reduce activation memory per device, thus avoiding the need to recompute activations in the backward pass to train larger models. If you're interested in this, I presented this paper at the Eleuther AI ML Scalability & Performance reading group, which you can check out the recording for [here](https://danielvegamyhre.github.io/ml/performance/2025/03/23/eleutherai-reading-group-session-9.html).

[^3]: In more modern transformer models, the hidden dimension can be as high as $$2^14 = 16,384$$ which would require ~1.67GB to store in bfloat16 with the same vocabulary size of 51,200.