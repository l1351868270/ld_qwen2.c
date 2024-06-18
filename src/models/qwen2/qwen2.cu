/*
make qwen2
python run.py --model_type=Qwen/Qwen1.5-0.5B-Chat --prompt="天空为什么是蓝色的,答案大于1000字"
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <fcntl.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "./src/kernels/cuda/embedding.cuh"
#include "./src/kernels/cuda/rope_half.cuh"
#include "./src/kernels/cuda/rmsnorm_half.cuh"
#include "./src/kernels/cuda/linear_half.cuh"
#include "./src/kernels/cuda/flash_attention_half.cuh"
#include "./src/kernels/cuda/residual_half.cuh"
#include "./src/kernels/cuda/silu_half.cuh"
#include "./src/kernels/cuda/argmax_half.cuh"

extern "C" {
    void c_init(int batch, int max_seq_len, const char *checkpoint_path);
    int* c_qwen2_forward(int batch, int seq_len, int *data, int pos);
    // void c_generate(int batch, int seq_len, int *data, int steps);
    // void c_chat ();
}

constexpr int MODEL_LIANMENT{16};

typedef struct {
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
} Qwen2Config;

typedef struct {
    half *embed_tokens;    // model.embed_tokens.weight
    half *q_proj_w;        // model.layers.{i}.self_attn.q_proj.weight
    half *q_proj_b;        // model.layers.{i}.self_attn.q_proj.bias
    half *k_proj_w;        // model.layers.{i}.self_attn.k_proj.weight
    half *k_proj_b;        // model.layers.{i}.self_attn.k_proj.bias
    half *v_proj_w;        // model.layers.{i}.self_attn.v_proj.weight
    half *v_proj_b;        // model.layers.{i}.self_attn.v_proj.bias
    half *o_proj;          // model.layers.{i}.self_attn.o_proj.weight
    half *gate_proj;       // model.layers.{i}.mlp.gate_proj.weight
    half *up_proj;                    // model.layers.{i}.mlp.up_proj.weight
    half *down_proj;                  // model.layers.{i}.mlp.down_proj.weight
    half *input_layernorm;            // model.layers.{i}.input_layernorm.weight
    half *post_attention_layernorm;   // model.layers.{i}.post_attention_layernorm.weight
    half *norm;            // model.norm.weight
    half *lm_head;         // lm_head.weight
} Qwen2Weights;

typedef struct {
    half *half_x;
    half *half_xb;
    half *half_xb2;
    half *half_hb;
    half *half_hb2;
    half *half_q;
    half *half_k;
    half *half_v;
    half *half_key_cache;
    half *half_value_cache;
    half *half_logits;
    int *next;
    int *token;
    int *next_cpu;
    int1 a;
    int batch;
    int max_seq_len;

    int flops;
    int flops_sfu;

    int num_parameters;
    cublasHandle_t *handle;
} RunState;

typedef struct {
    Qwen2Config config;
    RunState state;
    Qwen2Weights weights;
} Qwen2;

void malloc_run_state(RunState* s, Qwen2Config* p) {
    int seq_len = s->max_seq_len;
    int batch = s->batch;
    int hidden_size = p->hidden_size;
    int intermediate_size = p->intermediate_size;
    
    int num_heads = p->num_attention_heads;
    int head_dim = p->hidden_size / num_heads;
    int num_key_value_heads = p->num_key_value_heads;

    int num_hidden_layers = p->num_hidden_layers;

    unsigned long long run_cache = 0;

    cudaMalloc((void**)&s->half_x, batch * hidden_size * sizeof(half));
    run_cache += batch * hidden_size * sizeof(half);
    cudaMalloc((void**)&s->half_xb, batch * hidden_size * sizeof(half));
    run_cache += batch * hidden_size * sizeof(half);
    cudaMalloc((void**)&s->half_xb2, batch * hidden_size * sizeof(half));
    run_cache += batch * hidden_size * sizeof(half);
    cudaMalloc((void**)&s->half_hb, batch * intermediate_size * sizeof(half));
    run_cache += batch * intermediate_size * sizeof(half);
    cudaMalloc((void**)&s->half_hb2, batch * intermediate_size * sizeof(half));
    run_cache += batch * intermediate_size * sizeof(half);
    cudaMalloc((void**)&s->half_q, batch * hidden_size * sizeof(half));
    run_cache += batch * hidden_size * sizeof(half);
    unsigned long long kv_cache_size = batch * num_hidden_layers * seq_len * num_key_value_heads * head_dim * sizeof(half);
    cudaMalloc((void**)&s->half_key_cache, kv_cache_size);
    run_cache += kv_cache_size;
    cudaMalloc((void**)&s->half_value_cache, kv_cache_size);
    run_cache += kv_cache_size;
    printf("total kv cache size: %llu bytes, via %fKB, via %fMB, via %fGB\n", 2 * kv_cache_size, 
            (float)kv_cache_size  * 2.0 / 1024, (float)kv_cache_size  * 2.0 / 1024 / 1024, (float)kv_cache_size  * 2.0 / 1024 / 1024 / 1024);

    cudaMalloc((void**)&s->half_logits, batch * p->vocab_size * sizeof(half));
    run_cache += batch * p->vocab_size * sizeof(half);

    cudaMalloc((void**)&s->next, batch * sizeof(int));
    run_cache += batch * sizeof(int);
    cudaMalloc((void**)&s->token, batch * sizeof(int));
    run_cache += batch * sizeof(int);
    printf("total run cache size: %llu bytes, via %fKB, via %fMB, via %fGB\n", run_cache, 
            (float)run_cache / 1024, (float)run_cache / 1024 / 1024, (float)run_cache / 1024 / 1024 / 1024);

    s->next_cpu = (int*)malloc(batch * sizeof(int));
}

void free_run_state(RunState* s) {
    cudaFree(s->half_x);
    cudaFree(s->half_xb);
    cudaFree(s->half_xb2);
    cudaFree(s->half_hb);
    cudaFree(s->half_hb2);
    cudaFree(s->half_q);
    cudaFree(s->half_k);
    cudaFree(s->half_v);
    cudaFree(s->half_key_cache);
    cudaFree(s->half_value_cache);
    cudaFree(s->half_logits);
    free(s->next);
    free(s->token);
    free(s->next_cpu);
    cudaFree(s->half_logits);
    cudaFree(s->next);
    cudaFree(s->token);
    free(s->next_cpu);
}

void parse_ll(char** ptr, unsigned long long *ll, unsigned long long *ll_bytes, char *weight_name = nullptr) {
    cudaMemcpy(ll, *ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    *ptr += sizeof(unsigned long long);
    cudaMemcpy(ll_bytes, *ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    *ptr += sizeof(unsigned long long);
#ifdef WEIGHTS_DEBUG
    printf("weights length is:       %llu\n", *ll);
    printf("weights bytes length is: %llu\n", *ll_bytes);
#endif
}

void memory_map_weights(Qwen2Weights *w, Qwen2Config* p, char* ptr) {
    unsigned long long ll;
    unsigned long long ll_bytes;;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->embed_tokens = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->q_proj_w = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->q_proj_b = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->k_proj_w = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->k_proj_b = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->v_proj_w = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->v_proj_b = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->o_proj = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->gate_proj = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->up_proj = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->down_proj = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->input_layernorm = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->post_attention_layernorm = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->norm = (half*)ptr;
    ptr += ll_bytes;
    parse_ll(&ptr, &ll, &ll_bytes);
    w->lm_head = (half*)ptr;
}

void qwen2_build_from_checkpoint(Qwen2 *model, const char* checkpoint_path) {
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) {
        printf("Error opening model file %s\n", checkpoint_path);
    }
    char model_alignment[MODEL_LIANMENT];
    size_t file_size = 0;
    fseek(model_file, 0, SEEK_END);
    file_size = ftell(model_file);
    fseek(model_file, 0, SEEK_SET);

#ifdef WEIGHTS_DEBUG
    printf("file_size is: %ld\n", file_size);
#endif

    int rcount = 0;
    int model_magic;
    rcount = fread(&model_magic, sizeof(int), 1, model_file);
    if (rcount != 1) {
        fprintf(stderr, "Bad read magic from model file %s\n", checkpoint_path);
        exit(1);
    }
    
    if (model_magic != 20240516) {
        fprintf(stderr, "Bad magic model file %s\n", checkpoint_path);
        exit(1);
    }

#ifdef WEIGHTS_DEBUG
    printf("model magic is: %d\n", model_magic);
#endif
    rcount = fread(&model->config, sizeof(int), sizeof(model->config) / sizeof(int), model_file);
    if (rcount != sizeof(model->config) / sizeof(int)) {
        fprintf(stderr, "Bad read config from model file %s\n", checkpoint_path);
        exit(1);
    }

#ifdef WEIGHTS_DEBUG
    printf("config hidden_size is: %d\n", model->config.hidden_size);
    printf("config intermediate_size is: %d\n", model->config.intermediate_size);
    printf("config max_position_embeddings is: %d\n", model->config.max_position_embeddings);
    printf("config max_window_layers is: %d\n", model->config.max_window_layers);
    printf("config num_attention_heads is: %d\n", model->config.num_attention_heads);
    printf("config num_hidden_layers is: %d\n", model->config.num_hidden_layers);
    printf("config num_key_value_heads is: %d\n", model->config.num_key_value_heads);
    printf("config rms_norm_eps is: %f\n", model->config.rms_norm_eps);
    printf("config rope_theta is: %f\n", model->config.rope_theta);
    printf("config sliding_window is: %d\n", model->config.sliding_window);
    printf("config vocab_size is: %d\n", model->config.vocab_size);
#endif

    size_t head_bytes = sizeof(model->config) + sizeof(int);
    if (head_bytes % MODEL_LIANMENT != 0) {
        head_bytes += MODEL_LIANMENT - head_bytes % MODEL_LIANMENT;
        rcount = fread(model_alignment, sizeof(char), MODEL_LIANMENT - head_bytes % MODEL_LIANMENT, model_file);
    }
    size_t model_size = file_size - head_bytes;

    printf("model_size: %ld bytes, via %f KB, via %f MB, via %f GB\n", 
            model_size, (float)model_size / 1024, (float)model_size / 1024 / 1024, (float)model_size / 1024 / 1024 / 1024);

    // fclose(model_file);
    // int fd = open(checkpoint_path, O_RDONLY);
    // if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    // void *data;
    // data = mmap(NULL, file_size, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
    // if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    // char *host_memory = (char*)data + sizeof(int) + sizeof(Qwen2Config);

    // cudaMallocHost((void **)&data, model_size);


    char *host_memory = (char*)malloc(model_size);
    size_t chunck_size = 1024 * 1024 * 512; 
    size_t n_chuncks = model_size / chunck_size;
    size_t tail_size = model_size % chunck_size;

    printf("loading model from disk to host memory chuncks: %ld ......\n", n_chuncks);
    for (size_t i = 0; i < n_chuncks; i++) {
        rcount = fread(host_memory + i * chunck_size, sizeof(char), chunck_size, model_file);
        if (rcount != chunck_size) {
            fprintf(stderr, "Bad read model from model file %s\n", checkpoint_path);
            exit(1);
        }
    #ifdef WEIGHTS_DEBUG
        printf("n_chuncks:%lu the %lu chuncks\n", n_chuncks, i);
    #endif
    }

    if (tail_size > 0) {
        rcount = fread(host_memory + n_chuncks * chunck_size, sizeof(char), tail_size, model_file);
        if (rcount != tail_size) {
            fprintf(stderr, "Bad read model from model file %s\n", checkpoint_path);
            exit(1);
        }
    }


    // // https://people.csail.mit.edu/xchen/gpu-programming/Lecture14-stream.pdf
    // char *host_memory;
    // cudaHostAlloc((void **)&host_memory, model_size, cudaHostAllocDefault);
    // rcount = fread(host_memory, sizeof(char), model_size, model_file);
    // if (rcount != model_size) {
    //     fprintf(stderr, "Bad read model from model file %s\n", checkpoint_path);
    //     exit(1);
    // }

    void *device_memory;
    cudaError_t err;
    printf("loading model from host memory to device memory......\n");
    cudaMalloc((void**)&device_memory, model_size);
    ld_infer::cuda::CudaCheckError();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // printf("%s\n", cudaGetErrorName(err));
    cudaMemcpy(device_memory, host_memory, model_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("milliseconds: %.3f ms \n", milliseconds);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed %ld\n", model_size);
        exit(-1);
    }
    memory_map_weights(&model->weights, &model->config, (char*)device_memory);
    
    // free(host_memory);
}

typedef struct {

} Context;

__device__ bool thread0() {
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
}


void* qwen2_forward(Context *ctx, cublasHandle_t *handle, Qwen2* qwen2, int *token, int batch, int pos) {
    Qwen2Config *p = &qwen2->config;
    Qwen2Weights *w = &qwen2->weights;
    RunState* s = &qwen2->state;

    s->flops = 0;
    s->flops_sfu = 0;
    int max_seq_len = s->max_seq_len;
    // float *x = s->x;

    int hidden_size = p->hidden_size;
    int intermediate_size = p->intermediate_size;
    // int max_position_embeddings = p->max_position_embeddings;
    // int max_window_layers = p->max_window_layers;
    int num_attention_heads = p->num_attention_heads;
    int num_hidden_layers = p->num_hidden_layers;
    int num_key_value_heads = p->num_key_value_heads;
    float rms_norm_eps = p->rms_norm_eps;
    float rope_theta = p->rope_theta;
    // int sliding_window = p->sliding_window;
    int vocab_size = p->vocab_size;

    int num_heads = num_attention_heads;
    int head_dim = hidden_size / num_heads;
    
    ld_infer::cuda::embedding::embedding_fwd_launch<half>(s->half_x, w->embed_tokens, token, batch, hidden_size);
    ld_infer::cuda::CudaCheckError();

    // for(int l = 0; l < 1; l++) {
    for(int l = 0; l < p->num_hidden_layers; l++) {
        // attn_norm
        ld_infer::cuda::rmsnorm_half::rmsnorm_fwd_launch(s->half_xb, s->half_x, w->input_layernorm + l*hidden_size, rms_norm_eps, batch, hidden_size);

        int offset_k = l * max_seq_len * batch * num_key_value_heads * head_dim 
                         + pos * batch * num_key_value_heads * head_dim;
        int offset_v = l * max_seq_len * batch * num_key_value_heads * head_dim 
                         + pos * batch * num_key_value_heads * head_dim;

        s->half_k = s->half_key_cache + offset_k;
        s->half_v = s->half_value_cache + offset_v;

        ld_infer::cuda::linear_half::linear_fwd_launch(s->handle, s->half_q, s->half_xb, w->q_proj_w + l * hidden_size * (num_heads * head_dim), w->q_proj_b + l * (num_heads * head_dim), batch, hidden_size, num_heads * head_dim);        
        ld_infer::cuda::linear_half::linear_fwd_launch(s->handle, s->half_k, s->half_xb, w->k_proj_w + l * hidden_size * (num_key_value_heads * head_dim), w->k_proj_b + l * (num_key_value_heads * head_dim), batch, hidden_size, num_key_value_heads * head_dim);
        ld_infer::cuda::linear_half::linear_fwd_launch(s->handle, s->half_v, s->half_xb, w->v_proj_w + l * hidden_size * (num_key_value_heads * head_dim), w->v_proj_b + l * (num_key_value_heads * head_dim), batch, hidden_size, num_key_value_heads * head_dim);

        ld_infer::cuda::rope_half::rope_launch(s->half_q, rope_theta, batch, num_heads, head_dim, pos);
        ld_infer::cuda::rope_half::rope_launch(s->half_k, rope_theta, batch, num_key_value_heads, head_dim, pos);

        ld_infer::cuda::flash_attention_half::flash_attention_half_fwd_launch(s->half_xb, s->half_q, s->half_key_cache, s->half_value_cache, 
                             batch, num_heads, num_key_value_heads, head_dim, num_heads, num_key_value_heads, max_seq_len, 
                             num_hidden_layers, l, pos);



        ld_infer::cuda::linear_half::linear_fwd_launch(s->handle, s->half_xb2, s->half_xb, w->o_proj + l * (num_heads * head_dim) * hidden_size, NULL, batch, num_heads * head_dim, hidden_size);

        ld_infer::cuda::residual_half::residual_fwd_launch(s->half_x, s->half_xb2, batch, hidden_size);

        // ffn_norm
        ld_infer::cuda::rmsnorm_half::rmsnorm_fwd_launch(s->half_xb, s->half_x, w->post_attention_layernorm + l*hidden_size, rms_norm_eps, batch, hidden_size);
        ld_infer::cuda::linear_half::linear_fwd_launch(s->handle, s->half_hb, s->half_xb, w->gate_proj + l*intermediate_size*hidden_size, NULL, batch, hidden_size, intermediate_size);
        ld_infer::cuda::linear_half::linear_fwd_launch(s->handle, s->half_hb2, s->half_xb, w->up_proj + l*intermediate_size*hidden_size, NULL, batch, hidden_size, intermediate_size);

        ld_infer::cuda::silu_half::silu_fwd_launch(s->half_hb, s->half_hb2, batch, intermediate_size);

        ld_infer::cuda::linear_half::linear_fwd_launch(s->handle, s->half_xb, s->half_hb, w->down_proj + l* hidden_size * intermediate_size, NULL, batch, intermediate_size, hidden_size);
                
        ld_infer::cuda::residual_half::residual_fwd_launch(s->half_x, s->half_xb, batch, hidden_size);

        // cudaDeviceSynchronize();
    }
    
    ld_infer::cuda::rmsnorm_half::rmsnorm_fwd_launch(s->half_x, s->half_x, w->norm, rms_norm_eps, batch, hidden_size);
    ld_infer::cuda::linear_half::linear_fwd_launch1(s->handle, s->half_logits, s->half_x, w->lm_head, NULL, batch, hidden_size, vocab_size);

    return s->half_logits;
}


Qwen2 py_model;
cublasHandle_t handle;

void c_init(int batch, int max_seq_len, const char *checkpoint_path) {
    printf("checkpoint_path: %s\n", checkpoint_path);
    if (checkpoint_path == NULL) {
        checkpoint_path = "qwen1.5-0.5B.bin";
    }
    // const char *checkpoint_path = "qwen1.5-0.5B.bin";
    qwen2_build_from_checkpoint(&py_model, checkpoint_path);
    py_model.state.batch = batch;
    py_model.state.max_seq_len = max_seq_len;

    cublasCreate(&handle);
    py_model.state.handle = &handle;
    malloc_run_state(&py_model.state, &py_model.config);
    // cublasDestroy(handle);
}

// void get_mod
int* c_qwen2_forward(int batch, int seq_len, int *data, int pos) {
    // printf("c_openelm_forward batch:%d, seq_len:%d, pos:%d\n", batch, seq_len, pos);
    RunState *s = &py_model.state;
    
    // int* prompt_tokens = data;
    // int start = 0;
    for (int i = 0; i < batch; i++) {
        // s->token[i] = data[i];
        cudaMemcpy(s->token + i, data + i, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    Context ctx;
    qwen2_forward(&ctx, s->handle, &py_model, s->token, batch, pos);
    // cudaDeviceSynchronize();
    ld_infer::cuda::argmax_half::argmax_fwd_launch(s->next, s->half_logits, s->batch, py_model.config.vocab_size);

    // if (pos == 10) {
    //     cudaDeviceSynchronize();
    //     exit(1);
    // }

    for (int i = 0; i < s->batch; i++) {
        cudaMemcpy(s->next_cpu + i, s->next + i, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // printf("pos:%d ", pos+1);
    // for (int i = 0; i < s->batch; i++) {
    //     printf("%d ", s->next_cpu[i]);
    // }
    // printf("\n");
    return s->next_cpu;
}
