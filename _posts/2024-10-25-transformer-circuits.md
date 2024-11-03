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



This paper provides a mental model for reasoning about the internal workings of transformers and attention heads in deep neural networks. The insights here help understand and analyze the behaviors of large models.

<!--
Original Content
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



Concat and multiply is a problematic reordering of the computation. The right way to think is all attention head add their results to the residual stream

 

makes sense to look at diff attn heads as individually reading from stream's subspace and writing back to another (or same) subspace.
Ateention heads move info (only this component can )

Attention heads move information between tokens.
If an attention heads seems to be attending to a particular token (other activations are sparse in the attention pattern) of interest , we cant infer a lot from it because sometimes models can consolidate contextual information of the sentence in one token
and then use that information in later layers.

Since attention heads are simply moving information already in the residual stream, the circuit experiments focusing only on attention heads must
be done for tasks where we dont expect the model to do any factual recall.
Is factual recall done in MLP layers?

1 attention head can be seen as

Circuit diagram in notebook

Analogy between attention and 1-d convolution <> so attention is a great tool for a model to assess long range dependencies between tokens, which a 1-d convolution can
So attention can support longer dependencies as compared to 1-d convolution, but essentially both play the same role.

The independence of the QK and OV circuit
QK deals with the source (keys) and destination (query) of information by evaluating which token has important information for any given token
OV deals with the information that must be moved
It is useful to reason about attention behaviour using QK and OV circuits as our unit of analysis, not the Q,K,V vectors by themselves
So,
QK generates a probablity distribution
OV captures the information that is useful from the residual stream. aka source and destination
Multiplying both allows the flow of information from one token to the other in the context window (aka sequence)
Wq, Wk can be seen as low rank adaptations of QK circuit. Together the QK circuit given by Wqk is _(d_model, d_model)_ dimensions
Wv and Wo can be seen as low ran adaptations of OV circuit. Together the OV circtui is given by Wov is _(d_model, d_model)_ dimension
Although both the circuits are of the same dimensions, they do fundamentally different tasks.
QK circuit takes the embeddings from the residual stream and output a scalar representing the pairwise dotproduct of the different vectors
OV circuit takes the embeddings for the residual stream and outputs another set of vectors (corresponding to the embeddings it took).
QK is 






## References
1. [Neel's Walkthrough of the Paper](https://youtu.be/KV5gbOmHbjU?si=AybyWlRCTxAFhuqO)
2. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)




-->

















<!--AI Modified-->

---

### General Notes
1. This paper offers a clear mental framework for thinking about transformers and attention heads, which is helpful for reasoning about large models.
2. Frequently, computations within transformers are restructured for efficiency, making the operations less interpretable. Therefore, it is possible to find multiple expressions for the same operation, which are typically mathematically equivalent but may differ in clarity.

---

### Residual Stream

The residual stream is essential to understanding transformer architecture. It allows the model to bypass layers as needed, as it doesn't require going through all layers. This is possible because every attention head and MLP layer simply adds its information back to the residual stream.

![Transformer Architecture](/images/blogs/architecture.png)
_Architecture: The center line (from embed to unembed) is the residual stream._

Each head reads from the entire residual stream (including both original tokens and the information added by previous heads) and outputs a vector, which is then added back to the residual stream. Every head operates on a subspace of the residual stream, allowing components to communicate by encoding information within specific subspaces. For instance, one attention head can write information to a particular subspace, which another head can then read. This functionality enables attention heads and MLPs to use the residual stream to share information effectively.

Often, individual heads appear to perform distinct, hypothesized functions that can be isolated and analyzed.

#### Superposition and Interference in the Residual Stream

The residual stream often encodes more features than there are dimensions, so these features are approximately orthogonal (dot products close to zero). When projecting the residual stream to a smaller space, some interference can occur since the features are not fully orthogonal. However, since features are typically sparsely activated during training, this interference does not usually hinder model performance. If features occurred frequently together, interference would become more problematic.

---

### Attention Heads

1. An attention layer consists of multiple attention heads working independently and in parallel, with each head focusing on a subspace of the residual stream across all tokens in a sequence.
2. Each token in the sequence plays two roles: (i) A source of information for other tokens and (ii) A destination of information from other tokens. This can be summarized as below.

    ![QK Circuit](/images/blogs/attention_diagram.png)

3. **Within a single head**: for each token, the head learns a probability distribution over that token and all previous tokens. This probability distribution represents the relative importance (context) of each previous token for the current token. The weights are normalized using softmax, ensuring each row sums to 1.
4. We now have an attention pattern *A* which decides the source and destination of information. The next step can be understood in two different ways: from the point of view of first principles (easy to interpret, discussed in step 5) and from the point of view of computation (used while implementing the computation, discussed in steps 6 and 7).
5.  **From first principles**, you take a weight matrix _W_v_ and use that to project the vectors in the original sequence to the same space as _d_model_. This project is called _V_ matrix. This modified version of the original sequence of tokens represents the information for each token that will be written to the residual stream if the attention head choses to attend to that token. The next step is to multiply the attention pattern to the _V_ matrix to determine the net change to the embeddings for all tokens in the sequence.
6.  **In practical implementation**, you compute a value vector for each token by projecting the original token to a smaller space. Each weight (from step 3) is then multiplied by the corresponding value vector of that token, and the results are summed to produce a final weighted-average value vector for each token.

    ![Attention Visual](/images/blogs/attention_visual.png)

    [Download File here.](https://pratik-doshi-99.github.io/files/attention_visual.xlsx)

7. We now obtain a matrix _(d_head * sequence)_ for each head. Here *d_head* is the lower dimensional subspace. The matrices _(d_head * sequence)_ are concatenated to form a final matrix _(d_model * sequence)_ [where _d_head = d_model / num_heads_]. This concatenated matrix is multiplied by a learnable output matrix _(d_model * d_model)_, and the result is added to the residual stream. This variation is mathematically equivalent but computationally more efficient.

---

Although concatenation and multiplication provide computational efficiency, conceptually, it’s best to view each attention head as individually adding its results to the residual stream.

It can be helpful to consider each attention head as operating on its own subspace of the residual stream, reading from and writing back to specific dimensions within it.

Since attention heads facilitate information transfer across tokens, if an attention head appears to attend to a particular token (while other activations are sparse), we cannot necessarily infer much, as models sometimes consolidate contextual information into a single token to be used in later layers.

Because attention heads primarily move information that is already in the residual stream, experiments analyzing circuits through attention heads alone are best suited for tasks where factual recall by the model is not expected. It remains to be explored whether factual recall happens within MLP layers.

---

### Insights on Attention Mechanisms and Circuits

- Attention heads can be conceptually compared to a 1-D convolution: both enable a model to assess long-range dependencies between tokens, though attention supports longer dependencies. Essentially, both serve to transfer context between tokens.

#### QK and OV Circuits in Attention Mechanisms

1. **QK Circuit**: Determines the source (keys) and destination (query) of information by identifying the tokens with relevant information for each target token.
2. **OV Circuit**: Manages the content to be moved between tokens.

Analyzing attention behavior is more intuitive when viewing QK and OV circuits as separate units rather than focusing on individual Q, K, V vectors. 

- **QK Circuit**: This generates a probability distribution.
- **OV Circuit**: Retrieves useful information from the residual stream, representing the source and destination of information.

The multiplication of these two circuits facilitates information flow across tokens within the context window (sequence).

**Circuit Weights**:
   - **QK Circuit**: Defined by matrices \( W_q \) and \( W_k \), which represent low-rank approximations. The combined QK circuit (given by \( W_{qk} \)) has dimensions _(d_model, d_model)_.
   - **OV Circuit**: Defined by matrices \( W_v \) and \( W_o \), also as low-rank approximations. The combined OV circuit (given by \( W_{ov} \)) also has dimensions _(d_model, d_model)_.

Although the QK and OV circuits share dimensions, they perform fundamentally different tasks:
   - **QK Circuit**: Takes embeddings from the residual stream and outputs a scalar representing the pairwise dot product of different vectors.
   - **OV Circuit**: Processes the embeddings from the residual stream and outputs modified vectors corresponding to the inputs it received.



## References
1. [Neel's Walkthrough of the Paper](https://youtu.be/KV5gbOmHbjU?si=AybyWlRCTxAFhuqO)
2. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)