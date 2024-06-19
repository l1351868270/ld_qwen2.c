import argparse
import struct
from typing import Union, Optional
from transformers import AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM, Qwen2Config
from accelerate import init_empty_weights, infer_auto_device_map, disk_offload
import torch
import numpy as np
import gc
import os
import torch.nn as nn

MODEL_LIANMENT = 16

def write_fp32(tensor, file):
    file.write(tensor.detach().cpu().numpy().astype("float32").tobytes())

def write_fp16(tensor, file):
    file.write(tensor.detach().cpu().numpy().astype("float16").tobytes())

def write_int8(tensor, file):
    d = tensor.detach()
    d = d.cpu()
    d = d.view(-1)
    d = d.numpy()
    d = d.astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

# def write_int4(tensor, file):
#     data = tensor.detach().numpy().astype(np.uint8)
#     data_binary = np.unpackbits(data)
#     data_packed = np.packbits(data_binary, bitorder='little')

#     file.write(.tobytes())

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

def quantize_q40(w, w_bit, group_size):
    """
    takes a tensor and returns the w_bit quantized version
    i.e. symmetric quantization into uint128, range [0, 2^w_bit-1]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 7.0
    # scale into range [-max_int, max_int]
    quant = w / scale[:,None] 
    # scale into range [0, 2 * max_int]
    quant = quant + 7.0
    # round to nearest integer
    # int4val_low = torch.round(quant).to(torch.uint8).clamp_(0, 15) # low 4 bit
    int4val_low = torch.round(quant).to(torch.uint8) # low 4 bit
    # assert torch.all(int4val_low >= 0)
    # assert torch.all(int4val_low <= 15)
    int4val_high = torch.round(quant).to(torch.uint8) << 4 # high 4 bit
    int8val = int4val_low[:, ::2] + int4val_high[:, 1::2]
    # dequantize by rescaling
    fp32val = ((int4val_low.float() - 7.0) * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    # print(int4val_low)
    # print(int4val_high)
    # print(int8val)
    # print(fp32valr)
    # print(w)
    maxerr = err.max()
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


def q40_write_layer_key_2d(model_sd, format_str, num_hidden_layers, width, height, file):
    w_bit = 4
    ll = num_hidden_layers * width * height
    file.write(struct.pack("Q", ll))
    ll_bytes = ll // 2
    if (ll_bytes % MODEL_LIANMENT != 0):
        ll_bytes += MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT
    file.write(struct.pack("Q", ll_bytes))
    ew = []
    for i in range(num_hidden_layers): 
        w = model_sd[format_str.format(i)]
        q, _, err = quantize_q40(w, w_bit, width)
        write_int8(q, file)
        ew.append((err, w.shape))
        print(f"{format_str.format(i)} quantized {tuple(w.shape)} to Q4_0 with max error {err}")
    if (ll_bytes % MODEL_LIANMENT != 0):
        file.write(struct.pack("x" * (MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT)))
    ew.sort(reverse=True)
    print(f"{format_str.format(num_hidden_layers)} max quantization group error across all weights: {ew[0][0]}")
    
    ll = num_hidden_layers * height
    file.write(struct.pack("Q", ll))
    ll_bytes = ll * 2
    if (ll_bytes % MODEL_LIANMENT != 0):
        ll_bytes += MODEL_LIANMENT - ll_bytes % MODEL_LIANMENT
    file.write(struct.pack("Q", ll_bytes))
    for i in range(num_hidden_layers): 
        w = model_sd[format_str.format(i)]
        _, s, err = quantize_q40(w, w_bit, width)
        write_fp16(s, file)
        del(w)
        del(model_sd[format_str.format(i)])
        gc.collect()
        # print(f"{format_str.format(i)} quantized {tuple(s.shape)} to Q4_0 with max error {err}")


def fp16_write_model(model, config, file, partial_id):
    sd = model.state_dict()

    if partial_id == 0:
        # embedder 4
        ll = config.hidden_size * config.vocab_size
        fp16_write_single_key(sd, "model.embed_tokens.weight", ll, file) # [hidden_size, vocab_size] [151936, 1024]

    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    if partial_id == 1:
        ll = config.num_hidden_layers * config.hidden_size * (num_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.q_proj.weight", ll, file)

        ll = config.num_hidden_layers * (num_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.q_proj.bias", ll, file)

    if partial_id == 2:
        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.k_proj.weight", ll, file)

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.k_proj.bias", ll, file)

    if partial_id == 3:
        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.v_proj.weight", ll, file)

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.v_proj.bias", ll, file)

    if partial_id == 4:
        ll = config.num_hidden_layers * (num_heads * head_dim) * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.o_proj.weight", ll, file)

    if partial_id == 5:
        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.mlp.gate_proj.weight", ll, file)

    if partial_id == 6:
        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.mlp.up_proj.weight", ll, file)
        
    if partial_id == 7:
        ll = config.num_hidden_layers * config.intermediate_size * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.mlp.down_proj.weight", ll, file)

    if partial_id == 8:
        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.input_layernorm.weight", ll, file)

    if partial_id == 9:
        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.post_attention_layernorm.weight", ll, file)

    if partial_id == 10:
        ll = config.hidden_size
        fp16_write_single_key(sd, "model.norm.weight", ll, file)

    if partial_id == 11:
        ll = config.vocab_size * config.hidden_size
        fp16_write_single_key(sd, "lm_head.weight", ll, file)


# Vector-wise Quantization
def q80_write_model(model, config, file, partial_id):
    sd = model.state_dict()
    if partial_id == 0:
        # embedder 4
        ll = config.hidden_size * config.vocab_size
        fp16_write_single_key(sd, "model.embed_tokens.weight", ll, file) # [hidden_size, vocab_size] [151936, 1024]

    
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    if partial_id == 1:
        ll = config.num_hidden_layers * config.hidden_size * (num_heads * head_dim)
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.self_attn.q_proj.weight", config.num_hidden_layers, config.hidden_size, num_heads * head_dim, file)

        ll = config.num_hidden_layers * (num_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.q_proj.bias", ll, file)

    if partial_id == 2:
        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.self_attn.k_proj.weight", config.num_hidden_layers, config.hidden_size, config.num_key_value_heads * head_dim, file)

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.k_proj.bias", ll, file)

    if partial_id == 3:
        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.self_attn.v_proj.weight", config.num_hidden_layers, config.hidden_size, config.num_key_value_heads * head_dim, file)

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.v_proj.bias", ll, file)

    if partial_id == 4:
        ll = config.num_hidden_layers * (num_heads * head_dim) * config.hidden_size
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.self_attn.o_proj.weight", config.num_hidden_layers, num_heads * head_dim, config.hidden_size, file)

    if partial_id == 5:
        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.mlp.gate_proj.weight", config.num_hidden_layers, config.hidden_size, config.intermediate_size, file)

    if partial_id == 6:
        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.mlp.up_proj.weight", config.num_hidden_layers, config.hidden_size, config.intermediate_size, file)
        
    if partial_id == 7:
        ll = config.num_hidden_layers * config.intermediate_size * config.hidden_size
        # print(f"ll={ll}")
        q80_write_layer_key_2d(sd, "model.layers.{}.mlp.down_proj.weight", config.num_hidden_layers, config.intermediate_size, config.hidden_size, file)

    if partial_id == 8:
        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.input_layernorm.weight", ll, file)

    if partial_id == 9:
        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.post_attention_layernorm.weight", ll, file)

    if partial_id == 10:
        ll = config.hidden_size
        fp16_write_single_key(sd, "model.norm.weight", ll, file)

    if partial_id == 11:
        ll = config.vocab_size * config.hidden_size
        fp16_write_single_key(sd, "lm_head.weight", ll, file)


# Vector-wise Quantization
def q40_write_model(model, config, file, partial_id):
    sd = model.state_dict()
    
    if partial_id == 0:
        ll = config.hidden_size * config.vocab_size
        fp16_write_single_key(sd, "model.embed_tokens.weight", ll, file) # [hidden_size, vocab_size] [151936, 1024]

    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    if partial_id == 1:
        ll = config.num_hidden_layers * config.hidden_size * (num_heads * head_dim)
        q40_write_layer_key_2d(sd, "model.layers.{}.self_attn.q_proj.weight", config.num_hidden_layers, config.hidden_size, num_heads * head_dim, file)
        
        ll = config.num_hidden_layers * (num_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.q_proj.bias", ll, file)

    if partial_id == 2:
        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        q40_write_layer_key_2d(sd, "model.layers.{}.self_attn.k_proj.weight", config.num_hidden_layers, config.hidden_size, config.num_key_value_heads * head_dim, file)

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.k_proj.bias", ll, file)

    if partial_id == 3:
        ll = config.num_hidden_layers * config.hidden_size * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        q40_write_layer_key_2d(sd, "model.layers.{}.self_attn.v_proj.weight", config.num_hidden_layers, config.hidden_size, config.num_key_value_heads * head_dim, file)

        ll = config.num_hidden_layers * (config.num_key_value_heads * head_dim)
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.self_attn.v_proj.bias", ll, file)

    if partial_id == 4:
        ll = config.num_hidden_layers * (num_heads * head_dim) * config.hidden_size
        # print(f"ll={ll}")
        q40_write_layer_key_2d(sd, "model.layers.{}.self_attn.o_proj.weight", config.num_hidden_layers, num_heads * head_dim, config.hidden_size, file)

    if partial_id == 5:
        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        q40_write_layer_key_2d(sd, "model.layers.{}.mlp.gate_proj.weight", config.num_hidden_layers, config.hidden_size, config.intermediate_size, file)

    if partial_id == 6:
        ll = config.num_hidden_layers * config.hidden_size * config.intermediate_size
        # print(f"ll={ll}")
        q40_write_layer_key_2d(sd, "model.layers.{}.mlp.up_proj.weight", config.num_hidden_layers, config.hidden_size, config.intermediate_size, file)

    if partial_id == 7:        
        ll = config.num_hidden_layers * config.intermediate_size * config.hidden_size
        # print(f"ll={ll}")
        q40_write_layer_key_2d(sd, "model.layers.{}.mlp.down_proj.weight", config.num_hidden_layers, config.intermediate_size, config.hidden_size, file)

    if partial_id == 8:
        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.input_layernorm.weight", ll, file)

    if partial_id == 9:
        ll = config.num_hidden_layers * config.hidden_size
        # print(f"ll={ll}")
        fp16_write_layer_key(sd, config.num_hidden_layers, "model.layers.{}.post_attention_layernorm.weight", ll, file)

    if partial_id == 10:
        ll = config.hidden_size
        fp16_write_single_key(sd, "model.norm.weight", ll, file)

    if partial_id == 11:
        ll = config.vocab_size * config.hidden_size
        fp16_write_single_key(sd, "lm_head.weight", ll, file)

class PartialQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, partial_id):
        super(PartialQwen2ForCausalLM, self).__init__(config)

        if (partial_id == 0): # self.model.embed_tokens
            self.model.layers = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 1: # self.model.layers[i].self_attn.q_proj
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn.k_proj = None
                decoder_layer.self_attn.v_proj = None
                decoder_layer.self_attn.o_proj = None
                decoder_layer.self_attn.rotary_emb = None
                decoder_layer.mlp = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 2: # self.model.layers[i].self_attn.k_proj
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn.q_proj = None
                decoder_layer.self_attn.v_proj = None
                decoder_layer.self_attn.o_proj = None
                decoder_layer.self_attn.rotary_emb = None
                decoder_layer.mlp = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 3: # self.model.layers[i].self_attn.v_proj
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn.q_proj = None
                decoder_layer.self_attn.k_proj = None
                decoder_layer.self_attn.o_proj = None
                decoder_layer.self_attn.rotary_emb = None
                decoder_layer.mlp = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 4: # self.model.layers[i].self_attn.o_proj
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn.q_proj = None
                decoder_layer.self_attn.k_proj = None
                decoder_layer.self_attn.v_proj = None
                decoder_layer.self_attn.rotary_emb = None
                decoder_layer.mlp = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 5: # self.model.layers[i].mlp.gate_proj
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn = None
                decoder_layer.mlp.up_proj = None
                decoder_layer.mlp.down_proj = None
                decoder_layer.mlp.act_fn = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 6: # self.model.layers[i].mlp.up_proj
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn = None
                decoder_layer.mlp.gate_proj = None
                decoder_layer.mlp.down_proj = None
                decoder_layer.mlp.act_fn = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 7: # self.model.layers[i].mlp.down_proj
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn = None
                decoder_layer.mlp.gate_proj = None
                decoder_layer.mlp.up_proj = None
                decoder_layer.mlp.act_fn = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 8: # self.model.layers[i].input_layernorm
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn = None
                decoder_layer.mlp = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 9: # self.model.layers[i].post_attention_layernorm
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn = None
                decoder_layer.mlp = None
                decoder_layer.input_layernorm = None
            self.model.norm = None
            self.lm_head = None

        if partial_id == 10: # self.model.norm
            self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn = None
                decoder_layer.mlp = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.lm_head = None

        if partial_id == 11: # self.model.lm_head
            # self.model.embed_tokens = None
            for decoder_layer in self.model.layers:
                decoder_layer.self_attn = None
                decoder_layer.mlp = None
                decoder_layer.input_layernorm = None
                decoder_layer.post_attention_layernorm = None
            self.model.norm = None
        

if __name__ == "__main__":
    '''
    python tools/export_72B.py --filepath="qwen1.5-72B.bin" --dtype="fp16" --model_type=Qwen/Qwen1.5-72B-Chat
    python tools/export_72B.py --filepath="qwen1.5-72B-q80.bin" --dtype="q80" --model_type=Qwen/Qwen1.5-72B-Chat
    python tools/export_72B.py --filepath="qwen1.5-72B-q40.bin" --dtype="q40" --model_type=Qwen/Qwen1.5-72B-Chat

    python tools/export_72B.py --filepath="qwen1.5-32B-q40.bin" --dtype="q40" --model_type=Qwen/Qwen1.5-32B-Chat
    python tools/export_72B.py --filepath="qwen1.5-14B-q40.bin" --dtype="q40" --model_type=Qwen/Qwen1.5-14B-Chat

    python tools/export_72B.py --filepath="qwen1.5-0.5B-q40.bin" --dtype="q40" --model_type=Qwen/Qwen1.5-0.5B-Chat
    python tools/export_72B.py --filepath="qwen1.5-0.5B-q40.bin" --dtype="q40" --model_type=Qwen/Qwen1.5-0.5B-Chat
    python tools/export_72B.py --filepath="qwen1.5-0.5B-q80.bin" --dtype="q80" --model_type=Qwen/Qwen1.5-0.5B-Chat

    python tools/export_72B.py --filepath="qwen2-72B-q80.bin" --dtype="q80" --model_type="Qwen/Qwen2-72B-Instruct"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="qwen1.5-0.5B.bin")
    parser.add_argument("--model_type", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "q80", "q40"), type=str, default="fp16")
    args = parser.parse_args()
    model_type = args.model_type
    print("loading weights from pretrained qwen1.5: %s" % model_type)

    config = AutoConfig.from_pretrained(model_type)

    # sd = model.state_dict()
    # # print(sd.keys())
    # config = model.config
    # # print(config)
    dtype = args.dtype
    filepath = args.filepath
    ld_qwen2_home = os.environ.get("LD_QWEN2_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ld_qwen2_cache", "qwen2"))

    if not os.path.exists(os.path.join(ld_qwen2_home, "checkpoints")):
        os.makedirs(os.path.join(ld_qwen2_home, "checkpoints"))

    filepath = os.path.join(ld_qwen2_home, "checkpoints", filepath)

    print(f"dtype:{dtype} filepath:{filepath}")

    if (dtype == "fp16"):
        with open(filepath, "wb") as file:
            write_header(config, file)
            for i in range(0, 12):
                with init_empty_weights():
                    model = PartialQwen2ForCausalLM(config, i)
                    device_map = infer_auto_device_map(model, max_memory={0: "0GB", "cpu": "100GB", "disk": "0GB"}, verbose=True)
                    print(device_map)
                model = model.from_pretrained(model_type, i, torch_dtype=torch.float16, device_map=device_map)
                fp16_write_model(model, config, file, i)

    if (dtype == "q80"):
        with open(filepath, "wb") as file:
            write_header(config, file)
            for i in range(0, 12):
                with init_empty_weights():
                    model = PartialQwen2ForCausalLM(config, i)
                    device_map = infer_auto_device_map(model, max_memory={0: "0GB", "cpu": "100GB", "disk": "0GB"}, verbose=True)
                    print(device_map)
                model = model.from_pretrained(model_type, i, torch_dtype=torch.float16, device_map=device_map)
                q80_write_model(model, config, file, i)

    if (dtype == "q40"):
        with open(filepath, "wb") as file:
            write_header(config, file)
            for i in range(0, 12):
                gc.collect()
                with init_empty_weights():
                    model = PartialQwen2ForCausalLM(config, i)
                    device_map = infer_auto_device_map(model, max_memory={0: "0GB", "cpu": "100GB", "disk": "0GB"}, verbose=True)
                    print(device_map)
                model = model.from_pretrained(model_type, i, torch_dtype=torch.float16, device_map=device_map)
                q40_write_model(model, config, file, i)
