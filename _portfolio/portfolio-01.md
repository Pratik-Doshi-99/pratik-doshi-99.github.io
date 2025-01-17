---
title: "Automated Image Captioning"
excerpt: "Using Attention to predict image captions with greater accuracy.<br/><br/><img src='/images/img_caption.png'>"
collection: portfolio
---

This project explores the use of Vision Language Models for Automated Image Captioning. We stack a ResNet image encoder with Attention and LSTMs to predict token sequences from image inputs. We successfully implement the architecture from [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), train the model on the Flickr8k dataset with a custom tokenizer and observe a 25% improvement on the BLEU metric.

**Tech Stack:** Python, PyTorch, Kubernetes, NLTK

Detailed report and code [here.](https://github.com/Pratik-Doshi-99/image-captioning)