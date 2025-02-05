---
title: 'Paged Attention and vLLM'
date: 2024-12-15
permalink: /posts/paged-attention-vllm/
tags:
  - Attention
  - LLM Inference
  - vLLM
  - KV Cache
---


# Paged Attention and the Design of vLLM

Paged attention is an innovative approach inspired by the concepts of virtual memory and page tables in operating systems. vLLM builds on this idea to efficiently manage memory during inference. The true magic of paged attention lies in how it stores the KV cache in memory, ensuring optimal memory usage and high performance.

## Introduction

Large language models (LLMs) operate in two phases: **prefill** and **decoding**. In the prefill phase, the model computes key (K) and value (V) vectors and caches them for later use during the decoding phase. This cached data, known as the **KV cache**, plays a crucial role in generating responses.

## What Is the KV Cache?

The KV cache is essential for efficient LLM inference. Here’s how it works:

- **Prefill Phase:**  
  When a request is received, the model processes the input tokens and computes the corresponding key and value vectors. These vectors are stored sequentially in the cache.

- **Decoding Phase:**  
  During response generation, the model reuses the stored KV vectors to predict subsequent tokens without recomputing them.

### Structure of the KV Cache

For each request, the KV cache holds:
- **Key and Value vectors:** Two sets per token.
- **Dimensions:** Each vector is of size *hidden dimension*.
- **Sequence:** Vectors for each token in the sequence.
- **Layers:** The KV cache spans all layers of the model.

For instance, consider the 13B OPT model:
- **Vectors:** 2 (one for keys and one for values)
- **Hidden state dimensions:** 5120
- **Layers:** 40
- **Memory per element:** 2 bytes (using FP16)

This results in roughly 800 KB of memory per token. With a maximum sequence length of 2048 tokens, a single request could require up to 1.6 GB of memory for the KV cache.

## Memory Challenges in LLM Inference

Inference for LLMs is often memory bound. While GPUs such as the H100 deliver twice the FLOPs of the A100, they share similar high-bandwidth memory (HBM) capacities—typically around 80 GB. This makes memory and memory bandwidth significant bottlenecks in serving multiple requests.

### The Problem with Contiguous Memory Allocation

Earlier designs allocated a contiguous block of memory for each request based on the model’s maximum response length. For example, if a model has a maximum sequence length of 2048 tokens, that full length is reserved for every request—even if the actual response is shorter.  
Consider a model that, on average, generates 1000 tokens per request. In such cases, approximately 1048 tokens worth of memory remain unused. This unused memory cannot be reallocated for other requests, leading to severe fragmentation. In practice, research indicates that only 20-40% of the allocated memory is used for tokens, with the rest wasted due to fragmentation.

![Memory Wastage Graph](/images/blogs/vllm_memory_waste.png)

## The Concept of Paged Attention

Paged attention addresses the memory fragmentation issue by rethinking how memory is allocated for the KV cache. Instead of reserving one large contiguous block per request, the memory is divided into **logical blocks**. Each logical block is designed to store the key and value vectors for a predefined number of tokens.

### How It Works

- **Logical Blocks:**  
  Each logical block (similar to a virtual address) can store a fixed number of tokens’ KV vectors. When a block is full, a new block is allocated.

- **Block Table Mapping:**  
  These logical blocks are mapped to physical blocks on the GPU’s DRAM through a block table, which functions similarly to a page table in operating systems.

- **During the Prefill Phase:**  
  The first forward pass stores the KV vectors for the input tokens in a logical block (e.g., Block 0). Once this block is exhausted, a new block (e.g., Block 1) is allocated.

- **During the Decoding Phase:**  
  As new tokens are generated, their corresponding KV vectors are stored in the same logical block as the last chunk of the input prompt. If the current block fills up, another block (e.g., Block 2) is created.

This strategy ensures that each request only reserves memory for the actual tokens generated rather than for the entire potential sequence length.

![Block Table Diagram](/images/blogs/block_table.png)

## Analogy with Operating Systems

The design of paged attention closely mirrors the memory management in operating systems. In traditional systems, a process is given virtual pages, each mapped to physical memory frames through a page table. Although the process sees a contiguous block of memory, it is physically scattered across different areas of RAM.

Similarly, in vLLM:
- Each request is allocated logical blocks on demand.
- These blocks are mapped to non-contiguous physical memory on the GPU.
- This approach minimizes wastage, as memory is reserved only for the actual positions that will be used.

![Multiple Requests Block Table](/images/blogs/multiple_requests_block_table.png)

## Batching Techniques in Inference

Efficient batching is another key aspect of large language model inference. Two main challenges arise:

1. **Asynchronous Request Arrival:**  
   Requests may arrive at different times. A naive solution would be to delay processing until multiple requests are ready, or to wait until the current batch is complete.

2. **Varying Lengths of Input and Output:**  
   When requests have different sequence lengths, using padding tokens to equalize their lengths can lead to inefficient computations.

### A Better Approach

A more effective method is to operate at the iteration level:
- After each iteration, completed requests are removed from the batch.
- New requests are added dynamically.
- This method ensures that the GPU processes only actual data rather than dummy tokens from padding.
- As a result, an incoming request only waits for the current iteration to finish, thereby enhancing overall efficiency.

## Conclusion

Efficient memory management is crucial for large language model inference. By leveraging the concept of paged attention, vLLM significantly reduces memory fragmentation and optimizes GPU memory usage. This innovative design, inspired by operating system principles, not only makes better use of available resources but also improves the performance and scalability of LLM inference.

## References

1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)



<!--
-------------------------



Paged attention is inspired by the concept of virtualk memory and page table in operating systems. vLLM is built on top of the this implementation. The magic of paged attention lies in the manner in which it stores the KV Cache in memory.

What is kv cache?
llm has two phases: prefill and decoding.
kv is computed in prefill and cached to use in decoding phase.
the structure of the kv cache:
For 1 request: k and v vectors of a hidden dimension * the sequence of tokens * the number of layers
paper gives example of the 13B OPT model: 2 (key and value vectors) * 5120 (hidden state dimensions) * 40 (layers) * 2 bytes (per element, FP16) ~ 800 KB of space.
Max sequence length of 2048, so KV cache can be as much as 2048 * 800KB ~ 1.6GB for single request.

Doing inference for LLMs is memory bound (you can only serve as many requests in one batch as your have memory to do so).
GPU computation is very powerful: h100 does 2x FLOPs of A100, but has the same HBM at 80GB. Memory and memory bandwidth is an increasing bottleneck
Conclusion: memory management is key in llm inference

Earlier, during reference contiguous memory is allocated for each request. so if a model has max response length of 2048 tokens, memory corresponding to that is allocated in contiguous manner. For other requests, the next chunk of memory is allocated. In most cases, the model doesnt generate response length tokens, this causes memory fragmentation.
Say a model generates 1000 tokens on average and has max sequence length of 2048. a total of 1048 units of memory goes unused, which cant be used by any other smaller request also because it is reserved for a given request. Such chuncks of unused memory will exist for most requests, making the entire allocation highly fragmented.

Their research claims that in practice only 20-40% of allocated memory is actually used for tokens. Remaining is the different forms of fragmentation.
![graph showing wastage of memory](/images/blogs/vllm_memory_waste.png)


The magic of Paged Attention is in its analogy with paged memory implementation in operating systems. Instead of allocating contiguous memory proportional to the max response length of the model, it divides the memory into logical blocks. Each logical block can store the key and value vectors for a preconfigured number of tokens. Each logical block (synonymous to virtual address) is mapped to a physical block (physical address) on the GPU DRAM via a block table (synonymous to page table)

![diagram showing how block table works](/images/blogs/block_table.png)

When the request is received, the first forward pass (prefill phase) generates the KV vectors at each layer. For input token, the KV vectors are stored one after the other on a logical block (Block 0 in the image). Once the logical block is exhausted, a new block is allocated to next set of tokens (Block 1). After the prefill phase, when the next token is generated, its is on the same block as the last chunk of the input prompt. Each successive predicted token is placed one after the other on the logical block. If the logical block is exhausted, a new one is created (Block 2). This way logical blocks are generated, and mapped to physical blocks on GPU DRAM.
In operating systems, a process is allocated memory in virtual pages. Each process owns a certain number of virtual pages, which are mapped to physical frames/pgaes on the RAM. A process can have multiple pages, but no two processes will be allocated the same page (assuming no shared memory). From that process's point of view, it is saving data in a contiguous space, but that is not the case (physically). 

Similarly, each request is allocated logical blocks on demand. From the point of view of that request, it is saving KV vectors in a continuous space (one after the other on the logical block). And each block is mapped to a phyiscal block, but different physical blocks may not be contigous. In this fashion, at any given point a request is reserving memory corresponding the the reamining positions on the block assigned to it, not to the extent of its theoretical response length.

![block table for multiple requests](/images/blogs/multiple_requests_block_table.png)

Other inference stuff:

batching techniques:
problem 1: requests arrive at different times
naive approach: make a request wait till other requests arrive (to batch them), or make it wait till the currently executing batch is complete
problem 2: different lengths of input and output
naive approach: use padding tokens to equalize their length.

better approach? work at the iteration level. after each iteration, completed requests are removed and new ones are added.
Better GPU use because now actual forward pass on the new requests instead of dummy forward pass on padded tokens
An incoming request must wait only for the current iteration to complete.



References:
1. The Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention

-->