---
title: 'Implementing GPT from Scratch'
date: 2024-10-30
permalink: /posts/implementing-gpt.md/
tags:
  - Language Models
  - Autoregressive Training
  - PyTorch
  - Transformers
---

My implementation of Andrej's Karpathys [GPT from scratch](https://youtu.be/kCc8FmEb1nY?si=Rdc_bMOWquUdAUJ5).


<!--
This is AI modified
Original Notes: https://docs.google.com/document/d/1pO4NgBASUFp5qJaIMCmLURGAzvipNB4XmBXtpL3PfhM/edit?usp=sharing
-->

## Tokenization and Vocabulary
- Google utilizes SentencePiece for tokenization, while OpenAI employs tiktoken.
- Tokenization operates on a vocabulary: the longer the vocabulary, the shorter the individual encodings, and vice versa.
  
## Tokenization and Autoregressive Training
In the initial steps, tokenization involves assigning a serially incrementing integer to each character in the dataset. The entire dataset is then converted into a list of integers, which is subsequently transformed into a PyTorch 1-D tensor. 

While training the transformer, random chunks of the dataset are taken. The maximum length of such a chunk is referred to as the block size, which represents the maximum context length for predicting the next token. For instance, given a chunk \([18, 17, 47, 56, 1, 15]\), multiple examples can be derived:

- \(X = [18], Y = [17]\)
- \(X = [18, 17], Y = [47]\)
- \(X = [18, 17, 47], Y = [56]\)
- \(X = [18, 17, 47, 56], Y = [1]\)
- \(X = [18, 17, 47, 56, 1], Y = [15]\)

This approach is not only for increasing the sampling but also to train the transformer to process a single token, preparing it to accept any input size less than the block size. This can be seen as managing the time dimension.

## Mini-Batching
Mini-batches are formed by stacking multiple random chunks into a single tensor for efficiency, ensuring that the GPU remains busy. 

## Embeddings
The `nn.Embedding` layer functions as a lookup table (or hashmap) that maps word indices to a corresponding dimensional vector (the word’s embedding). For example, passing a tensor of shape \((4, 8)\) while wanting to maintain the word embeddings in 10 dimensions will yield a tensor of dimensions \((4, 8, 10)\), where each element has 10 dimensions. 

If no specifications are made during object creation, the vectors are initialized randomly and updated during training. Alternatively, pre-trained weights can be used, necessitating that the indices of the words remain unchanged. This means each word or token is assigned an incrementing index, typically starting from 0.

### Purpose of Embeddings
In time series analysis, inputs are generally numerical, making embeddings less of an issue. However, in Natural Language Processing (NLP), dealing with non-numerical inputs such as words necessitates encoding sentences, paragraphs, and documents for the model to interpret them effectively.

A naive approach for this is One-Hot encoding, where each word is represented as a vector the size of the vocabulary with a single 1 at the corresponding word's index. For example, if "hello" has an index of 42, its One-Hot encoded vector would appear as follows:

```
[0, 0, 0, 0, 0, ..., 0, 1, 0, ..., 0]
```

The resulting vector is as long as the vocabulary size. However, with vocabularies exceeding 100,000 words, such as a 100-word input document, this leads to a matrix of shape \((100, 100000)\)—unnecessarily large.

Typically, the first layer to receive this input is a linear layer, \((100000, 300)\), intended to reduce dimensionality. The operation \(M.dot(E)\) effectively selects the rows in \(E\) corresponding to the indices of the 1’s in \(M\), eliminating the need for the One-Hot encoded matrix \(M\). Thus, `nn.Embedding` serves as a linear layer that facilitates this operation without requiring the large matrix \(M\). This is why explicit embedding layers are ubiquitous in NLP, barring character-based models with smaller vocabularies (e.g., < 100 characters).

## Understanding Cross Entropy
The `cross_entropy` function accepts two parameters: logits and targets. The target is a simple 1-D tensor or a list of class indices corresponding to the ground truth for each sample in the batch. If the first sample has a ground truth class index of 1, the first element in the target list will be 1. The length of this list corresponds to the number of samples in the batch.

Logits represent the class-wise bare activations per sample. For example, if there are 10 classes and a batch size of 100, the logits would be a matrix with 100 rows (batches) and 10 columns (one column per class). The function automatically performs One-Hot encoding for the targets, resulting in a matrix of 100 rows and 10 columns (one hot vector for each sample).

The function computes the softmax activation of the raw logits to create a probability distribution of the logit vector for each sample. The cross-entropy loss is calculated as \(-\log(\text{softmax activated vector} \times \text{ground truth vector})\). Given that the ground truth vector is One-Hot encoded, this effectively results in taking the \(-\log\) of the softmax action of the ground truth class. The formula for loss \(l(x, y)\) simplifies as follows:

\[
l(x, y) = -\log\left(\frac{\exp(x_n y_n)}{Z}\right)
\]

Where \(Z\) is the softmax denominator. The weight \(w_y\) outside the log allows for adjusting loss weights for specific classes. For example, passing a weight vector with appropriate weights can increase the loss weight for class 10. The overall cross-entropy loss can be computed by aggregating the costs for each sample in the batch, either by summing or averaging.

You can read more about this in the [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## The Bigram Model
The Bigram model predicts the next token based on the current token. A lookup into the embeddings table retrieves the embedded vector. As only a single token is used for prediction, the result from the embeddings is reshaped, and cross-entropy is computed.

The dimensions of your batch are structured as follows: 
- (Batch size / number of random training chunks, Time/context length / block size / size of random training chunk).

From the embeddings table, an embedding vector is obtained for each token in the current batch, resulting in a tensor with dimensions \((\text{batch size}, \text{block size}, \text{embedding dimensions})\). In this case, the embedding dimensions match the vocabulary size, yielding dimensions of \((\text{batch size}, \text{block size}, \text{vocab size})\).

This tensor is reshaped to \((\text{batch size} \times \text{block size}, \text{vocab size})\). Each row represents a token, and the columns denote the embeddings. Since the embedding dimensions match the vocabulary size, each row can be viewed as an activation distribution over the vocabulary, allowing for direct computation of the cross-entropy loss with the ground truth labels (the token immediately following the current token).

If the embedding dimension differed from the vocabulary size, this interpretation as a probability distribution over the vocabulary would not hold.

## Model Evaluation and Training
The methods `model.eval()` and `model.train()` are employed to switch the neural network’s mode. Some architectures exhibit different behaviors during training compared to evaluation.

The `@torch.no_grad()` context is defined for a function, indicating that the function will not call `loss.backward()`, meaning PyTorch won’t perform intermediary calculations during the forward pass.

## Attention Mechanism
Batch multiplication using the “@” operator enables batch matrix multiplication with triangular matrices for weighted sums.

### Computing the Attention Pattern
The attention pattern is derived by computing the dot product of the query and key vectors. To maintain the autoregressive property of the model—ensuring that when predicting the \(t+1\) token, the \(t+2\) token is not utilized—we can use all tokens up to \(t\).

In matrix terms, the query matrix has dimensions \((\text{time}, \text{embedding dimensions})\), while the key matrix also has dimensions \((\text{time}, \text{embedding dimensions})\). The attention pattern is computed by multiplying the query vectors with the key vectors as follows:

\[
\text{attention pattern} = q \times k^T
\]

For instance, if the context length is 8 and embedding dimensions are 16, both \(q\) and \(k\) would have dimensions of \((8, 16)\), and \(k^T\) would be \((16, 8)\). Each column of \(k^T\) represents an embedding, and the number of columns determines the context.

The operation \(q @ k^T\) results in a matrix of dimensions \((B, T, T)\), where each row represents the dot product of a single token query with all other token keys. To preserve the autoregressive property, all elements above the principal diagonal are masked to \(-\infty\). This masking ensures that when the softmax is applied across the rows, these elements become zero.

For efficiency on the GPU, the operation is executed in PyTorch as follows:

```python
q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) → (B, T, T)
```

In this notation, each row of the resulting matrix represents the dot product of

 a token query with all token keys. Let's denote this matrix as \(wei\).

Afterward, the row-wise softmax is applied to the \(wei\) matrix, resulting in the attention pattern. The output matrix for the attention head is then computed as:

```python
output = wei @ v  # (B, T, T) @ (B, T, 16) → (B, T, C)
```

Here, \(v\) represents the value matrix corresponding to the key. The resulting output matrix retains dimensions \((B, T, C)\).
