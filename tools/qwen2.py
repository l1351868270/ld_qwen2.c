import torch
from torch import nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cpu" # the device to load the model onto

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def generate(model, tokenizer, tokenized_prompt, steps):
    # tokenized_prompt = tokenizer.encode(prompt)
    tokenized_prompt = torch.tensor([tokenized_prompt])
    # print(tokenized_prompt.shape)
    config = model.config
    pos = 0
    past_key_values_length = 0
    seq_length = tokenized_prompt.shape[-1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    print(position_ids)
    
    while(pos < 1):
        inputs_embeds  = model.model.embed_tokens(tokenized_prompt[:,:1])
        # print(inputs_embeds.dtype)
        # print(inputs_embeds)
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(model.model.layers[:1]):
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            # print(hidden_states.shape)
            # print(hidden_states)
           
            query_states = decoder_layer.self_attn.q_proj(hidden_states)
            print(query_states.shape)
            print(query_states)

            print(decoder_layer.self_attn.q_proj.weight)

            key_states = decoder_layer.self_attn.k_proj(hidden_states)
            # print(key_states.shape)
            # print(key_states)
            value_states = decoder_layer.self_attn.v_proj(hidden_states)
            # print(value_states.shape)
            # print(value_states)

            bsz, q_len, _ = hidden_states.size()
            query_states = query_states.view(bsz, q_len, decoder_layer.self_attn.num_heads, decoder_layer.self_attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, decoder_layer.self_attn.num_key_value_heads, decoder_layer.self_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, decoder_layer.self_attn.num_key_value_heads, decoder_layer.self_attn.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            cos, sin = decoder_layer.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)
            # print(query_states.shape)
            # print(query_states)
            # print(key_states.shape)
            # print(key_states)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(decoder_layer.self_attn.head_dim)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=decoder_layer.self_attn.attention_dropout, training=decoder_layer.self_attn.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, decoder_layer.self_attn.hidden_size)
            # print(attn_output.shape)
            # print(attn_output)
            
            attn_output = decoder_layer.self_attn.o_proj(attn_output)
            # print(attn_output.shape)
            # print(attn_output)
            hidden_states = attn_output
            hidden_states = residual + hidden_states
            # print(hidden_states.shape)
            # print(hidden_states)

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            # print(hidden_states.shape)
            # print(hidden_states)
            # hidden_states = decoder_layer.mlp(hidden_states)
            gate_ = decoder_layer.mlp.gate_proj(hidden_states)
            # print(gate_.shape)
            # print(gate_)
            up_ = decoder_layer.mlp.up_proj(hidden_states)
            # print(up_.shape)
            # print(up_)
            # hidden_states1 = decoder_layer.mlp(hidden_states)
            # print(hidden_states1.shape)
            # print(hidden_states1)
            hidden_states = decoder_layer.mlp.act_fn(gate_) * up_
            # print(hidden_states.shape)
            # print(hidden_states)
            hidden_states = decoder_layer.mlp.down_proj(hidden_states)
            # print(hidden_states.shape)
            # print(hidden_states)
            hidden_states = residual + hidden_states
            # print(hidden_states.shape)
            # print(hidden_states)

        hidden_states = model.model.norm(hidden_states)
        # print(hidden_states.shape)
        # print(hidden_states)
        logits = model.lm_head(hidden_states)
        # print(logits)
        pos += 1

def generate1(model, tokenizer, tokenized_prompt, steps):
    # tokenized_prompt = tokenizer.encode(prompt)
    tokenized_prompt = torch.tensor([tokenized_prompt])
    # print(tokenized_prompt.shape)
    config = model.config
    # model.generate
    pos = 0
    past_key_values_length = 0
    seq_length = tokenized_prompt.shape[-1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    print(position_ids)
    
    while(pos < 1):
        hidden_states  = model.model(tokenized_prompt[:,:1])
        # print(hidden_states.shape)
        # print(hidden_states)
        logits = model.lm_head(hidden_states[0])
        print(logits)

        pos += 1
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat",
        torch_dtype=torch.float32,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", dtype=torch.float32)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    model_inputs = model_inputs["input_ids"]
    tokenized_prompt = model_inputs.flatten().tolist()
    seq_len = len(tokenized_prompt)
    steps = seq_len + 1
    generate(model, tokenizer, tokenized_prompt, steps)
    # generate1(model, tokenizer, tokenized_prompt, steps)
