# 🧠 DeepSeek Fine-Tuning Framework

This repository provides a lightweight pipeline for fine-tuning [DeepSeek](https://huggingface.co/deepseek-ai) and other instruction-tuned LLMs using curated, token-efficient datasets. An example model training plot can be seen [here](https://wandb.ai/asantucci-stanford-university/deepseek%20training/reports/Example-loss-curve--VmlldzoxMjI2OTk0OQ).

## 🚀 Features

- ✅ Shard-based streaming dataloaders (10B Token Sample of `FineWeb-EDU`)
- ✅ LoRa, KV-caching, RoPE, MoE...
- ✅ Interactive chat UI available for fine-tuned model.

## Limitations
- 🚫 Currently does not start with a pre-trained model.
- 🚫 ...