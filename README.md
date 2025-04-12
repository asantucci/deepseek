# 🧠 DeepSeek Fine-Tuning Framework

This repository provides a lightweight pipeline for fine-tuning [DeepSeek](https://huggingface.co/deepseek-ai) and other instruction-tuned LLMs using curated, token-efficient datasets. At present, it's intended for learning.

## 🚀 Features

- ✅ Shard-based streaming dataloaders (e.g. 10B Token Sample of `FineWeb-EDU`)
- ✅ Efficient tokenization using `tiktoken` (OpenAI-compatible)
- ✅ Multiprocessing-backed data preprocessing
- ✅ Support for 8-bit optimizer checkpoints
- ✅ Ready for long-context causal LM training
