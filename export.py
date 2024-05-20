import argparse
import struct
from typing import Union, Optional
from transformers import AutoModelForCausalLM
import torch
import numpy as np

def write_fp32(tensor, file):
    file.write(tensor.detach().numpy().astype("float32").tobytes())

def write_fp16(tensor, file):
    file.write(tensor.detach().numpy().astype("float16").tobytes())

# def write_fp16(tensor, file):
#     file.write(tensor.detach().numpy().astype("bfloat16").tobytes())


def fp32_write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        head_len = 0
        file.write(struct.pack("i", 20240516))
        head_len += 1
        file.write(struct.pack("i", config.hidden_size))
        head_len += 1
        file.write(struct.pack("i", config.intermediate_size))
        head_len += 1
        file.write(struct.pack("i", config.max_position_embeddings))
        head_len += 1
        file.write(struct.pack("i", config.max_window_layers))
        head_len += 1
        file.write(struct.pack("i", config.num_attention_heads))
        head_len += 1
        file.write(struct.pack("i", config.num_hidden_layers))
        head_len += 1
        file.write(struct.pack("i", config.num_key_value_heads))
        head_len += 1
        file.write(struct.pack("f", config.rms_norm_eps))
        head_len += 1
        file.write(struct.pack("f", config.rope_theta))
        head_len += 1
        file.write(struct.pack("i", config.sliding_window))
        head_len += 1
        file.write(struct.pack("i", config.vocab_size))
        head_len += 1
        print(f"header length: {head_len}")

        sd = model.state_dict()

        keys_len = 0
        # embedder 4
        ll = config.hidden_size * config.vocab_size
        file.write(struct.pack("Q", ll))
        write_fp32(sd["model.embed_tokens.weight"], file)
        # print(sd["model.embed_tokens.weight"].shape) # [hidden_size, vocab_size] [151936, 1024]
        keys_len += 1

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        ll = config.num_hidden_layers * config.hidden_size * (num_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.q_proj.weight"], file) # [hidden_size, num_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.q_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (num_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.q_proj.bias"], file) # [num_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.q_proj.bias"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.k_proj.weight"], file) # [hidden_size, num_key_value_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.k_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.k_proj.bias"], file) # [num_key_value_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.k_proj.bias"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.v_proj.weight"], file) # [hidden_size, num_key_value_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.v_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.v_proj.bias"], file) # [num_key_value_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.v_proj.bias"].shape)
        keys_len += config.num_hidden_layers


        ll = config.num_hidden_layers * (num_heads * head_dim) * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.o_proj.weight"], file) # [num_heads * head_dim, hidden_size] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.o_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.mlp.gate_proj.weight"], file) # [intermediate_size, hidden_size]  [2816, 1024]
            # print(sd[f"model.layers.{i}.mlp.gate_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.mlp.up_proj.weight"], file) # [intermediate_size, hidden_size] [2816, 1024]
            # print(sd[f"model.layers.{i}.mlp.up_proj.weight"].shape)
        keys_len += config.num_hidden_layers
        
        ll = config.num_hidden_layers * config.intermediate_size * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.mlp.down_proj.weight"], file) # [hidden_size, intermediate_size] [1024, 2816]
            # print(sd[f"model.layers.{i}.mlp.down_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.input_layernorm.weight"], file) # [hidden_size] [1024]
            # print(sd[f"model.layers.{i}.input_layernorm.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.post_attention_layernorm.weight"], file) # [hidden_size] [1024]
            # print(sd[f"model.layers.{i}.post_attention_layernorm.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.hidden_size
        file.write(struct.pack("Q", ll))
        write_fp32(sd[f"model.norm.weight"], file)
        # print(sd[f"model.norm.weight"].shape)
        keys_len += 1

        ll = config.vocab_size * config.hidden_size
        file.write(struct.pack("Q", ll))
        write_fp32(sd[f"lm_head.weight"], file) # [vocab_size, hidden_size] [151936, 1024]
        # print(sd[f"lm_head.weight"].shape)
        keys_len += 1

        print(f"keys length: {keys_len}")
        assert keys_len == len(model.state_dict())

def fp16_write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        head_len = 0
        file.write(struct.pack("i", 20240516))
        head_len += 1
        file.write(struct.pack("i", config.hidden_size))
        head_len += 1
        file.write(struct.pack("i", config.intermediate_size))
        head_len += 1
        file.write(struct.pack("i", config.max_position_embeddings))
        head_len += 1
        file.write(struct.pack("i", config.max_window_layers))
        head_len += 1
        file.write(struct.pack("i", config.num_attention_heads))
        head_len += 1
        file.write(struct.pack("i", config.num_hidden_layers))
        head_len += 1
        file.write(struct.pack("i", config.num_key_value_heads))
        head_len += 1
        file.write(struct.pack("f", config.rms_norm_eps))
        head_len += 1
        file.write(struct.pack("f", config.rope_theta))
        head_len += 1
        file.write(struct.pack("i", config.sliding_window))
        head_len += 1
        file.write(struct.pack("i", config.vocab_size))
        head_len += 1
        print(f"header length: {head_len}")

        sd = model.state_dict()

        keys_len = 0
        # embedder 4
        ll = config.hidden_size * config.vocab_size
        file.write(struct.pack("Q", ll))
        write_fp16(sd["model.embed_tokens.weight"], file)
        # print(sd["model.embed_tokens.weight"])
        # print(sd["model.embed_tokens.weight"].shape) # [hidden_size, vocab_size] [151936, 1024]
        keys_len += 1

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        ll = config.num_hidden_layers * config.hidden_size * (num_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.self_attn.q_proj.weight"], file) # [hidden_size, num_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.q_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (num_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.self_attn.q_proj.bias"], file) # [num_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.q_proj.bias"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.self_attn.k_proj.weight"], file) # [hidden_size, num_key_value_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.k_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.self_attn.k_proj.bias"], file) # [num_key_value_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.k_proj.bias"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.self_attn.v_proj.weight"], file) # [hidden_size, num_key_value_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.v_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.self_attn.v_proj.bias"], file) # [num_key_value_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.v_proj.bias"].shape)
        keys_len += config.num_hidden_layers


        ll = config.num_hidden_layers * (num_heads * head_dim) * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.self_attn.o_proj.weight"], file) # [num_heads * head_dim, hidden_size] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.o_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.mlp.gate_proj.weight"], file) # [intermediate_size, hidden_size]  [2816, 1024]
            # print(sd[f"model.layers.{i}.mlp.gate_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.mlp.up_proj.weight"], file) # [intermediate_size, hidden_size] [2816, 1024]
            # print(sd[f"model.layers.{i}.mlp.up_proj.weight"].shape)
        keys_len += config.num_hidden_layers
        
        ll = config.num_hidden_layers * config.intermediate_size * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.mlp.down_proj.weight"], file) # [hidden_size, intermediate_size] [1024, 2816]
            # print(sd[f"model.layers.{i}.mlp.down_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.input_layernorm.weight"], file) # [hidden_size] [1024]
            # print(sd[f"model.layers.{i}.input_layernorm.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp16(sd[f"model.layers.{i}.post_attention_layernorm.weight"], file) # [hidden_size] [1024]
            # print(sd[f"model.layers.{i}.post_attention_layernorm.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.hidden_size
        file.write(struct.pack("Q", ll))
        write_fp16(sd[f"model.norm.weight"], file)
        # print(sd[f"model.norm.weight"].shape)
        keys_len += 1

        ll = config.vocab_size * config.hidden_size
        file.write(struct.pack("Q", ll))
        write_fp16(sd[f"lm_head.weight"], file) # [vocab_size, hidden_size] [151936, 1024]
        # print(sd[f"lm_head.weight"].shape)
        keys_len += 1

        print(f"keys length: {keys_len}")
        assert keys_len == len(model.state_dict())

def bf16_write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        head_len = 0
        file.write(struct.pack("i", 20240516))
        head_len += 1
        file.write(struct.pack("i", config.hidden_size))
        head_len += 1
        file.write(struct.pack("i", config.intermediate_size))
        head_len += 1
        file.write(struct.pack("i", config.max_position_embeddings))
        head_len += 1
        file.write(struct.pack("i", config.max_window_layers))
        head_len += 1
        file.write(struct.pack("i", config.num_attention_heads))
        head_len += 1
        file.write(struct.pack("i", config.num_hidden_layers))
        head_len += 1
        file.write(struct.pack("i", config.num_key_value_heads))
        head_len += 1
        file.write(struct.pack("f", config.rms_norm_eps))
        head_len += 1
        file.write(struct.pack("f", config.rope_theta))
        head_len += 1
        file.write(struct.pack("i", config.sliding_window))
        head_len += 1
        file.write(struct.pack("i", config.vocab_size))
        head_len += 1
        print(f"header length: {head_len}")

        sd = model.state_dict()

        keys_len = 0
        # embedder 4
        ll = config.hidden_size * config.vocab_size
        file.write(struct.pack("Q", ll))
        write_fp32(sd["model.embed_tokens.weight"], file)
        print(sd["model.embed_tokens.weight"].dtype) # [hidden_size, vocab_size] [151936, 1024]
        print(sd["model.embed_tokens.weight"])
        keys_len += 1

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        ll = config.num_hidden_layers * config.hidden_size * (num_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.q_proj.weight"], file) # [hidden_size, num_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.q_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (num_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.q_proj.bias"], file) # [num_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.q_proj.bias"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.k_proj.weight"], file) # [hidden_size, num_key_value_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.k_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.k_proj.bias"], file) # [num_key_value_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.k_proj.bias"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.v_proj.weight"], file) # [hidden_size, num_key_value_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.v_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.v_proj.bias"], file) # [num_key_value_heads * head_dim] [1024]
            # print(sd[f"model.layers.{i}.self_attn.v_proj.bias"].shape)
        keys_len += config.num_hidden_layers


        ll = config.num_hidden_layers * (num_heads * head_dim) * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.self_attn.o_proj.weight"], file) # [num_heads * head_dim, hidden_size] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.o_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.mlp.gate_proj.weight"], file) # [intermediate_size, hidden_size]  [2816, 1024]
            # print(sd[f"model.layers.{i}.mlp.gate_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.mlp.up_proj.weight"], file) # [intermediate_size, hidden_size] [2816, 1024]
            # print(sd[f"model.layers.{i}.mlp.up_proj.weight"].shape)
        keys_len += config.num_hidden_layers
        
        ll = config.num_hidden_layers * config.intermediate_size * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.mlp.down_proj.weight"], file) # [hidden_size, intermediate_size] [1024, 2816]
            # print(sd[f"model.layers.{i}.mlp.down_proj.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.input_layernorm.weight"], file) # [hidden_size] [1024]
            # print(sd[f"model.layers.{i}.input_layernorm.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        file.write(struct.pack("Q", ll))
        for i in range(config.num_hidden_layers): 
            write_fp32(sd[f"model.layers.{i}.post_attention_layernorm.weight"], file) # [hidden_size] [1024]
            # print(sd[f"model.layers.{i}.post_attention_layernorm.weight"].shape)
        keys_len += config.num_hidden_layers

        ll = config.hidden_size
        file.write(struct.pack("Q", ll))
        write_fp32(sd[f"model.norm.weight"], file)
        # print(sd[f"model.norm.weight"].shape)
        keys_len += 1

        ll = config.vocab_size * config.hidden_size
        file.write(struct.pack("Q", ll))
        write_fp32(sd[f"lm_head.weight"], file) # [vocab_size, hidden_size] [151936, 1024]
        # print(sd[f"lm_head.weight"].shape)
        keys_len += 1

        print(f"keys length: {keys_len}")
        assert keys_len == len(model.state_dict())


if __name__ == "__main__":
    '''
    python export.py --filepath="qwen1.5-0.5B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-0.5B-Chat
    python export.py --filepath="qwen1.5-1.8B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-1.8B-Chat
    python export.py --filepath="qwen1.5-4B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-4B-Chat
    python export.py --filepath="qwen1.5-14B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-14B-Chat
    python export.py --filepath="qwen1.5-32B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-32B-Chat
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="qwen1.5-0.5B.bin")
    parser.add_argument("--model_type", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "q80", "q40"), type=str, default="fp16")
    args = parser.parse_args()
    model_type = args.model_type
    print("loading weights from pretrained qwen1.5: %s" % model_type)

    model = AutoModelForCausalLM.from_pretrained(model_type, trust_remote_code=True)
    # print(model)
    sd = model.state_dict()
    # print(sd.keys())
    config = model.config
    # print(config)
    dtype = args.dtype
    filepath = args.filepath
    print(f"dtype:{dtype} filepath:{filepath}")
    if (dtype == "fp32"):
        fp32_write_model(model, filepath)
    if (dtype == "fp16"):
        fp16_write_model(model, filepath)
    if (dtype == "bf16"):
        bf16_write_model(model, filepath)
    # if (dtype == "q80"):
    #     q80_write_model(model, filepath)
    # if (dtype == "q40"):
    #     q40_write_model(model, filepath)