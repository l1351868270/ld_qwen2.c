import argparse
import struct
from typing import Union, Optional
from transformers import AutoModelForCausalLM
import torch
import numpy as np

MODEL_LIANMENT = 16

def write_fp32(tensor, file):
    file.write(tensor.detach().numpy().astype("float32").tobytes())

def write_fp16(tensor, file):
    file.write(tensor.detach().numpy().astype("float16").tobytes())

def write_int8(tensor, file):
    file.write(tensor.detach().numpy().astype(np.int8).tobytes())

# refer to https://github.com/karpathy/llama2.c/blob/master/export.py

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

def write_header(config, file):
        head_len = 0
        head_bytes = 0
        file.write(struct.pack("i", 20240516))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.hidden_size))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.intermediate_size))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.max_position_embeddings))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.max_window_layers))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.num_attention_heads))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.num_hidden_layers))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.num_key_value_heads))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("f", config.rms_norm_eps))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("f", config.rope_theta))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.sliding_window))
        head_len += 1
        head_bytes += 4
        file.write(struct.pack("i", config.vocab_size))
        head_len += 1
        head_bytes += 4
        print(f"header length:{head_len}, header bytes:{head_bytes}")

        if (head_bytes % MODEL_LIANMENT != 0):
            head_bytes += MODEL_LIANMENT - head_bytes % MODEL_LIANMENT
            file.write(struct.pack("x" * (MODEL_LIANMENT - head_bytes % MODEL_LIANMENT)))

def fp16_write_single_key(model_sd, key, ll, file):
    file.write(struct.pack("Q", ll))
    ll_bytes = ll * 2
    if (ll_bytes % MODEL_LIANMENT != 0):
        ll_bytes += MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT
    file.write(struct.pack("Q", ll_bytes))
    write_fp16(model_sd[key], file)
    # print(model_sd[key])
    # print(model_sd["model.embed_tokens.weight"].shape)
    if (ll_bytes % MODEL_LIANMENT != 0):
        file.write(struct.pack("x" * (MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT)))

def fp16_write_layer_key(model_sd, num_hidden_layers, format_str, ll, file):
    file.write(struct.pack("Q", ll))
    ll_bytes = ll * 2
    if (ll_bytes % MODEL_LIANMENT != 0):
        ll_bytes += MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT
    file.write(struct.pack("Q", ll_bytes))
    for i in range(num_hidden_layers): 
        write_fp16(model_sd[format_str.format(i)], file) # [hidden_size, num_heads * head_dim] [1024, 1024]
            # print(sd[f"model.layers.{i}.self_attn.q_proj.weight"].shape)
    if (ll_bytes % MODEL_LIANMENT != 0):
        file.write(struct.pack("x" * (MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT)))

def q80_write_layer_key_2d(model_sd, format_str, num_hidden_layers, width, height, file):
    ll = num_hidden_layers * width * height
    file.write(struct.pack("Q", ll))
    ll_bytes = ll * 1
    if (ll_bytes % MODEL_LIANMENT != 0):
        ll_bytes += MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT
    file.write(struct.pack("Q", ll_bytes))
    ew = []
    for i in range(num_hidden_layers): 
        # write_fp16(model_sd[format_str.format(i)], file) # [hidden_size, num_heads * head_dim] [1024, 1024]
        #     # print(sd[f"model.layers.{i}.self_attn.q_proj.weight"].shape)
        w = model_sd[format_str.format(i)]
        q, _, err = quantize_q80(w, width)
        write_int8(q, file)
        ew.append((err, w.shape))
        # print(f"{format_str.format(i)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")
    if (ll_bytes % MODEL_LIANMENT != 0):
        file.write(struct.pack("x" * (MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT)))
    ew.sort(reverse=True)
    # print(f"{format_str.format(num_hidden_layers)} max quantization group error across all weights: {ew[0][0]}")

    ll = num_hidden_layers * height
    file.write(struct.pack("Q", ll))
    ll_bytes = ll * 2
    if (ll_bytes % MODEL_LIANMENT != 0):
        ll_bytes += MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT
    file.write(struct.pack("Q", ll_bytes))
    for i in range(num_hidden_layers): 
        w = model_sd[format_str.format(i)]
        _, s, err = quantize_q80(w, width)
        write_fp16(s, file)
        print(f"{format_str.format(i)} quantized {tuple(s.shape)} to Q8_0 with max error {err}")

def fp16_write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        write_header(config, file)

        sd = model.state_dict()

        keys_len = 0
        # embedder 4
        ll = config.hidden_size * config.vocab_size
        fp16_write_single_key(sd, "model.embed_tokens.weight", ll, file) # [hidden_size, vocab_size] [151936, 1024]
        keys_len += 1

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        ll = config.num_hidden_layers * config.hidden_size * (num_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.q_proj.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (num_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.q_proj.bias", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.k_proj.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.k_proj.bias", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.v_proj.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.v_proj.bias", ll, file)
        keys_len += config.num_hidden_layers


        ll = config.num_hidden_layers * (num_heads * head_dim) * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.o_proj.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.mlp.gate_proj.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.mlp.up_proj.weight", ll, file)
        keys_len += config.num_hidden_layers
        
        ll = config.num_hidden_layers * config.intermediate_size * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.mlp.down_proj.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.input_layernorm.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.post_attention_layernorm.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.hidden_size
        fp16_write_single_key(sd, "model.norm.weight", ll, file)
        keys_len += 1

        ll = config.vocab_size * config.hidden_size
        fp16_write_single_key(sd, "lm_head.weight", ll, file)
        keys_len += 1

        print(f"keys length: {keys_len}")
        assert keys_len == len(model.state_dict())

# Vector-wise Quantization
def q80_write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        write_header(config, file)

        sd = model.state_dict()

        keys_len = 0
        # embedder 4
        ll = config.hidden_size * config.vocab_size
        fp16_write_single_key(sd, "model.embed_tokens.weight", ll, file) # [hidden_size, vocab_size] [151936, 1024]
        keys_len += 1

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        ll = config.num_hidden_layers * config.hidden_size * (num_heads * head_dim)
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.self_attn.q_proj.weight", config.num_hidden_layers, config.hidden_size, num_heads * head_dim, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (num_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.q_proj.bias", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.self_attn.k_proj.weight", config.num_hidden_layers, config.hidden_size, config.num_key_value_heads * head_dim, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.k_proj.bias", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.self_attn.v_proj.weight", config.num_hidden_layers, config.hidden_size, config.num_key_value_heads * head_dim, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.v_proj.bias", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * (num_heads * head_dim) * config.hidden_size
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.self_attn.o_proj.weight", config.num_hidden_layers, num_heads * head_dim, config.hidden_size, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.mlp.gate_proj.weight", config.num_hidden_layers, config.hidden_size, config.intermediate_size, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.mlp.up_proj.weight", config.num_hidden_layers, config.hidden_size, config.intermediate_size, file)
        keys_len += config.num_hidden_layers
        
        ll = config.num_hidden_layers * config.intermediate_size * config.hidden_size
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.mlp.down_proj.weight", config.num_hidden_layers, config.intermediate_size, config.hidden_size, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.input_layernorm.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.post_attention_layernorm.weight", ll, file)
        keys_len += config.num_hidden_layers

        ll = config.hidden_size
        fp16_write_single_key(sd, "model.norm.weight", ll, file)
        keys_len += 1

        ll = config.vocab_size * config.hidden_size
        fp16_write_single_key(sd, "lm_head.weight", ll, file)
        keys_len += 1

        print(f"keys length: {keys_len}")
        assert keys_len == len(model.state_dict())


if __name__ == "__main__":
    '''
    python export.py --filepath="qwen1.5-0.5B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-0.5B-Chat
    python export.py --filepath="qwen1.5-1.8B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-1.8B-Chat
    python export.py --filepath="qwen1.5-4B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-4B-Chat
    python export.py --filepath="qwen1.5-14B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-14B-Chat
    python export.py --filepath="qwen1.5-14B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-14B-Chat
    python export.py --filepath="qwen1.5-32B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-32B-Chat

    python export.py --filepath="qwen1.5-0.5B-q80.bin" --dtype="q80" --model_type=Qwen/Qwen1.5-0.5B-Chat
    python export.py --filepath="qwen1.5-14B-q80.bin" --dtype="q80" --model_type=Qwen/Qwen1.5-14B-Chat
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
    if (dtype == "fp16"):
        fp16_write_model(model, filepath)
    if (dtype == "q80"):
        q80_write_model(model, filepath)

    # if (dtype == "q80"):
    #     q80_write_model(model, filepath)
    # if (dtype == "q40"):
    #     q40_write_model(model, filepath)