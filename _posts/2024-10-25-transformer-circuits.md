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


# Review: A Mathematical Framework for Transformer Circuits

### General Notes
1. The paper gives a nice mental model to think about transformers and attention heads. This is helpful in reasoning about big models.

2. At a lot of points, computations in the transformer are re-strcutred for efficiency, but that makes the operations less human understandable. So it is possible to find multiple expressions of the same operation. In most cases, varying expressions are mathematically equivalent.



### Residual Stream

The residual stream is a critical concept when we talk about transformers. The model can choose which layers it wants to go through, it doesnt need to go through all the layers. The stream allows the model to do this. Because all heads and MLP layers just add their information back to the residual stream. 


![Transformer Architecture](/images/blogs/architecture.png)
_Architecture: The center line is the residual stream_

Each head will take the entire residual stream (original tokens + information added by the previous heads) and output a vector that will be added to the residual stream. Every head works on a subspace of the residual stream. Different components of the model can communicate with each other by using the dimensions of the residual stream. For example: An attention head can write some information in a certain subspace of the residual stream. Another attention head may be reading from that subspace. In this way the residual stream can by attention heads and MLPs to communicate with the each other.

Often, heads will be responsible for some function (that is hypothesized to be human understandable), which can be isolated. 

### Residual Stream: Super-position and Interference
Packing more features than there are dimensions. Clearly, those features cant be orthogonal. So there are close to orthogonal (dot product close to 0). The residual stream would typically encode more features than its dimensions. When projecting this residual stream directions to a small space, there would be interference because the residual stream features are not orthogonal. But since all the features encoded are sparse in training, the interference doesnt create a problem for the model. If the features occured very often, the model would face a lot of inteference during projections.

### Attention Heads

An attention head has two components.
1. Attention Pattern














## References
1. [Neel's Walkthrough of the Paper](https://youtu.be/KV5gbOmHbjU?si=AybyWlRCTxAFhuqO)
2. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

