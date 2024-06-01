# ld_qwen2.c

# 已完成
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
weight int8

weight int4

# TODO
## kv cache

paged attention

## 量化

activation 量化

# openai api server

# lm_eval

OPENAI_API_KEY=YOUR_KEY_HERE lm_eval --model local-chat-completions --tasks gsm8k --model_args model="Qwen/Qwen1.5-0.5B-Chat",base_url=http://127.0.0.1:8000/v1

curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }'
