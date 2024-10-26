---
title: 'Review: A Mathematical Framework for Transformer Circuits'
date: 2024-10-25
permalink: /posts/transformer-circuits/
tags:
  - Mechanistic Interpretability
  - Circuits
  - Deep Learning
  - Language Models
  - Transformers
---

A paper review of [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), to build strong mental models that can be used to reason about the internal functioning of deep neural networks.

### General Notes
1. The paper gives a nice mental model to think about transformers and attention heads. This is helpful in reasoning about big models.

2. At a lot of points, computations in the transformer are re-strcutred for efficiency, but that makes the operations less human understandable. So it is possible to find multiple expressions of the same operation. In most cases, varying expressions are mathematically equivalent.



### Residual Stream

The residual stream is a critical concept when we talk about transformers. The model can choose which layers it wants to go through, it doesnt need to go through all the layers. The stream allows the model to do this. Because all heads and MLP layers just add their information back to the residual stream. 


![Transformer Architecture](/images/blogs/architecture.png)
_Architecture: The center line (from embed to unembed) is the residual stream_

Each head will take the entire residual stream (original tokens + information added by the previous heads) and output a vector that will be added to the residual stream. Every head works on a subspace of the residual stream. Different components of the model can communicate with each other by using the dimensions of the residual stream. For example: An attention head can write some information in a certain subspace of the residual stream. Another attention head may be reading from that subspace. In this way the residual stream can be used by attention heads and MLPs to communicate with each other.

Often, heads will be responsible for some function (that is hypothesized to be human understandable), which can be isolated. 

### Residual Stream: Super-position and Interference
Packing more features than there are dimensions. Clearly, those features cant be orthogonal. So there are close to orthogonal (dot product close to 0). The residual stream would typically encode more features than its dimensions. When projecting this residual stream directions to a small space, there would be interference because the residual stream features are not orthogonal. But since all the features encoded are sparse in training, the interference doesnt create a problem for the model. If the features occured very often, the model would face a lot of inteference during projections.

### Attention Heads (WIP)


1. An attention layer is a sum of individual attention heads working independently and parallel to each other. Each attention head focuses on a subspace of the residual stream for all the tokens in the given sequence.
2. Inside a single head: For each token, that head learns a probability distribution over that and all previous tokens. This probability distribution (per token) serves as the weights (importance of the context of each of the previous tokens for the current token). The weights are normalized using softmax. Each row adds up to 1.
3. Each of the weights are multiplied by the respective value vector of that token, and finally summed, to arrive at the final weighted-average value vector (per token).

![Attention Visual](/images/blogs/attention_visual.png)

[Download File here.](https://pratik-doshi-99.github.io/files/attention_visual.xlsx)
4. We now have a matrix _(d_head * sequence)_ for each head.
   * **From first principles**, a separate output matrix _(d_model * d_head)_ for each head is used to project these matrices _(d_head * sequence)_ back to the residual stream dimensions _(d_model * sequence)_. Finally all of them are added to the residual stream individually. These output matrices (1 per head) are learnable.
   * **In practice**, you concatenate these matrices _(d_head * sequence)_ one below the other to arrive at the final matrix _(d_model * sequence)_ [Note that: _d_head = d_model / num_heads_]. This concatenated matrix is multiplied by a learnable output matrix _(d_model * d_model)_ and the result is added back to the residual stream. This variant is mathematically equivalent and computationally better.












## References
1. [Neel's Walkthrough of the Paper](https://youtu.be/KV5gbOmHbjU?si=AybyWlRCTxAFhuqO)
2. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

