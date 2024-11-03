---
title: 'Review: Interpretability in the Wild: A Circuit for Indirect Object Detection in GPT2-Small'
date: 2024-10-30
permalink: /posts/ioi-circuit/
tags:
  - Mechanistic Interpretability
  - Circuits
  - Deep Learning
  - Language Models
  - Transformers
---

A paper review highlighting the key discoveries with respect to attention heads and the algorithms used.

<!--
Original Content
A paper review highlighting the key discoveries with respect to attention heads and the algorithm used for the discovery.

## The Discovery

1. For each head, patch it to see the imapct on logit difference and visualize which heads lead to a strong increase in logit difference (meaning those heads were writing information against the IO token), and which ones reduce logit difference (they are working in the direction of the IO token). This can be understood via a heatmap
3. Name Mover Heads: The name mover head attends to names, and copies whatever they attend to (separation of the QK and OV circuits). To check this, the next two things are done. 
4. Not very sure about the part where they map attention probabilitiy vs logit score:
    - Attention probability for the name implies that in the Attention Matrix (Q * K), the Key for the name token is given a high probability against the query for the end token? Or query for which token?
5. Copy Score: While point 4 tries to identify whether the correct tokens are attended to by the name mover heads, the copy score determines whether there is actual copying taking place. In this, you take a name token from the residual stream after the first MLP layer (why? not quite sure), project it using the OV matrix of a name mover head (remember that the OV circuit determines the info written by the head into a residual stream). This is used to simulate the siutation when that head completely attends to the name (giving wieght of 100% to that name token). then multiplied with the unembedding matrix. Then you check if the original name is in the top 5 logits. For name mover heads that is the cae 95% of the times. For other heads, its below 20% times. This is the copy score.
6. For the negative name mover heads, you do the same, just do the negative of the OV matrix and compute the copy score (this is called the negative copy score). For negative name movers, the negative copy score is 98%, compared to 12% for an average head.
7. Moving backwards from the name mover heads to the S-inhibitor heads. Patching all direct paths from all heads (individually) occuring before the name mover heads till the name mover heads and assessing the logit difference. Also assessing how that changes the attention pattern of the name mover heads. Turns out that S-inhibitors reduce the extent to which the name mover attends to the S1 and S2 tokens (which are the wrong prediction). They are called S-inhibitors because they inhibit attention to the Subject token. In specific, after identifying the name movers, they only look for heads that affect the query of the name mover and give an explaination for discarding the values and the keys.
8. After arriving at the S-inhibitors, they go back further to assess different heads before the S-inhibitors. They find a few minor heads. This time they check the impact of previous heads on the S-inhibitors values. They dont find anything tangible for the queries and keys.
9. They find the duplicate token heads that are active at the S2 token and attend primarily to the S1 token. They signal that token duplication has occured to the S-inhibitor heads
-->


<!--AI Modified-->
A paper review highlighting the key discoveries with respect to attention heads and the algorithms used.

## The Discovery

1. For each head, patch it to observe the impact on logit difference and visualize which heads lead to a strong increase in logit difference (indicating that these heads are writing information against the IO token) and which reduce logit difference (indicating alignment with the IO token). This can be visualized via a heatmap.

2. **Name Mover Heads**: The name mover head attends to names and copies whatever they attend to, due to the separation of the QK and OV circuits. To verify this, the next two steps are performed.

3. **Attention Probability Mapping**: The attention probability for a name token implies that in the Attention Matrix (Q * K), the Key for the name token is given a high probability against the query for the end token. However, it’s unclear which query is used here—whether it’s the end token’s query or another.

4. **Copy Score**: While the previous step assesses whether the correct tokens are attended to by the name mover heads, the copy score determines whether actual copying is taking place. Here’s the process:
   - Take a name token from the residual stream after the first MLP layer (reason unclear), then project it using the OV matrix of a name mover head (since the OV circuit determines what information is written by the head into the residual stream).
   - This projection simulates a scenario in which the head fully attends to the name (giving 100% weight to that name token), and is then multiplied with the unembedding matrix.
   - The original name is then checked to see if it appears in the top 5 logits. For name mover heads, this is the case 95% of the time; for other heads, it occurs below 20% of the time. This measure is referred to as the copy score.

5. For the **negative name mover heads**, the same procedure is applied, except using the negative of the OV matrix to compute the copy score, called the **negative copy score**. For negative name movers, the negative copy score is 98%, compared to 12% for an average head.

6. Moving backwards from the name mover heads to the S-inhibitor heads, all direct paths from heads (individually) occurring before the name mover heads are patched and assessed for logit difference. Additionally, changes in the attention pattern of the name mover heads are analyzed. It turns out that **S-inhibitors** reduce the extent to which the name mover attends to the S1 and S2 tokens (which are the incorrect predictions). They are called S-inhibitors because they inhibit attention to the subject token. Specifically, after identifying the name movers, only heads affecting the query of the name mover are analyzed, and an explanation is given for discarding the values and the keys.

7. After identifying the S-inhibitors, the analysis continues further back to assess different heads preceding the S-inhibitors. A few **minor heads** are identified. This time, the impact of previous heads on the values of S-inhibitors is examined, but no tangible results are found for the queries and keys.

8. **Duplicate Token Heads** are identified as being active at the S2 token, primarily attending to the S1 token. They signal to the S-inhibitor heads that token duplication has occurred.