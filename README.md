# ld_qwen2.c

## 目录说明
single_deploy: 单文件部署
src: 模型部署文件
tools: 工具文件，包括模型export, python run wrapper, demo等
scripts: 脚本文件

# 已完成
## CPU
支持x86 avx512, aarch64 neon
## kv cache
## 混合精度 

accumulate float

a,b half

## 算子融合

group attention

flash attention

GHA

MHA

## 量化
### Vector-wise Quantization
W8A16

W4A16

## batch
naive batch / static batch

# TODO
## kv cache

paged attention

## 量化

activation 量化

## batch
continuous batch / in-flight batch

# 参考资料

## 大模型qwen2
[qwen2 github code](https://github.com/QwenLM/Qwen2)

[qwen2 transformers code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2)

[qwen2 blog](https://qwenlm.github.io/zh/blog/qwen2/)

## 量化
[llm.int8](https://arxiv.org/pdf/2208.07339)

[SmoothQuant](https://arxiv.org/pdf/2211.10438)

[AWQ](https://arxiv.org/pdf/2306.00978)

## cuda
[PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/)

[Dissecting Tensor Cores via Microbenchmarks:
Latency, Throughput and Numeric Behaviors](https://arxiv.org/pdf/2206.02874)

[Benchmarking and Dissecting the Nvidia Hopper
GPU Architecture](https://arxiv.org/pdf/2402.13499)

[GPUs Go Brrr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)

## 模型并行
[Megatron-LM: Training Multi-Billion Parameter Language Models Using
Model Parallelism](https://arxiv.org/pdf/1909.08053)

## continuous batching
[How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)

## flash-attention
[From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

[Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
