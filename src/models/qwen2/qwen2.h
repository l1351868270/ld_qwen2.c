#pragma once

namespace ld_infer {
namespace ld_qwen2 {
constexpr int MODEL_LIANMENT{16};

struct Qwen2Config {
    int hidden_size;
    int intermediate_size;
    int max_position_embeddings;
    int max_window_layers;
    int num_attention_heads;
    int num_hidden_layers;
    int num_key_value_heads;
    float rms_norm_eps;
    float rope_theta;
    int sliding_window;
    int vocab_size;
};

template <typename T>
struct Qwen2Weights {
    T *embed_tokens;    // model.embed_tokens.weight
    T *q_proj_w;        // model.layers.{i}.self_attn.q_proj.weight
    T *q_proj_b;        // model.layers.{i}.self_attn.q_proj.bias
    T *k_proj_w;        // model.layers.{i}.self_attn.k_proj.weight
    T *k_proj_b;        // model.layers.{i}.self_attn.k_proj.bias
    T *v_proj_w;        // model.layers.{i}.self_attn.v_proj.weight
    T *v_proj_b;        // model.layers.{i}.self_attn.v_proj.bias
    T *o_proj;          // model.layers.{i}.self_attn.o_proj.weight
    T *gate_proj;       // model.layers.{i}.mlp.gate_proj.weight
    T *up_proj;                    // model.layers.{i}.mlp.up_proj.weight
    T *down_proj;                  // model.layers.{i}.mlp.down_proj.weight
    T *input_layernorm;            // model.layers.{i}.input_layernorm.weight
    T *post_attention_layernorm;   // model.layers.{i}.post_attention_layernorm.weight
    T *norm;            // model.norm.weight
    T *lm_head;         // lm_head.weight
};

template <typename T>
struct RunState {
    T *x;
    T *xb;
    T *xb2;
    T *hb;
    T *hb2;
    T *q;
    T *k;
    T *v;
    T *att;
    T *key_cache;
    T *value_cache;
    T *logits;
    int *next;
    int *token;

    int batch;
    int max_seq_len;

    int flops;
    int flops_sfu;

    int num_parameters;
#ifdef ENABLE_MUTI
    int mpiRank;
    int mpiSize;
#endif
} ;

template <typename TS, typename TW>
struct Qwen2 {
    Qwen2Config config;
    RunState<TS> state;
    Qwen2Weights<TW> weights;
} ;
} // namespace ld_qwen2
} // namespace ld_infer