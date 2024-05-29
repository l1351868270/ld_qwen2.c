/*
nvcc --shared -Xcompiler -fPIC -o qwen2.so -O3 qwen2.cu -lm -gencode arch=compute_86,code=sm_86
python run.py
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

extern "C" {
    void c_init(int batch, int max_seq_len, const char *checkpoint_path);
    int* c_qwen2_forward(int batch, int seq_len, int *data, int pos);
    // void c_generate(int batch, int seq_len, int *data, int steps);
    // void c_chat ();
}

constexpr int MODEL_LIANMENT{16};
constexpr int WARP_THREADS{32};
constexpr int WARPGROUP_THREADS{128};
constexpr int WARPGROUP_WARPS{4};

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
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *key_cache;
    float *value_cache;
    float *att;
    float *logits;
    int *next;
    int *token;
    int *next_cpu;

    int batch;
    int max_seq_len;

    int flops;
    int flops_sfu;

    int num_parameters;
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

    cudaMalloc((void**)&s->x, batch * hidden_size * sizeof(float));
    run_cache += batch * hidden_size * sizeof(float);
    cudaMalloc((void**)&s->xb, batch * hidden_size * sizeof(float));
    run_cache += batch * hidden_size * sizeof(float);
    cudaMalloc((void**)&s->xb2, batch * hidden_size * sizeof(float));
    run_cache += batch * hidden_size * sizeof(float);
    cudaMalloc((void**)&s->hb, batch * intermediate_size * sizeof(float));
    run_cache += batch * intermediate_size * sizeof(float);
    cudaMalloc((void**)&s->hb2, batch * intermediate_size * sizeof(float));
    run_cache += batch * intermediate_size * sizeof(float);
    cudaMalloc((void**)&s->q, batch * hidden_size * sizeof(float));
    run_cache += batch * hidden_size * sizeof(float);
    cudaMalloc((void**)&s->att, s->batch * num_heads * seq_len * sizeof(float));
    run_cache += s->batch * num_heads * seq_len * sizeof(float);
    unsigned long long kv_cache_size = batch * num_hidden_layers * seq_len * num_key_value_heads * head_dim * sizeof(float);
    cudaMalloc((void**)&s->key_cache, kv_cache_size);
    run_cache += kv_cache_size;
    cudaMalloc((void**)&s->value_cache, kv_cache_size);
    run_cache += kv_cache_size;
    printf("total kv cache size: %llu bytes, via %fKB, via %fMB, via %fGB\n", 2 * kv_cache_size, 
            (float)kv_cache_size  * 2.0 / 1024, (float)kv_cache_size  * 2.0 / 1024 / 1024, (float)kv_cache_size  * 2.0 / 1024 / 1024 / 1024);
    cudaMalloc((void**)&s->logits, batch * p->vocab_size * sizeof(float));
    run_cache += batch * p->vocab_size * sizeof(float);
    cudaMalloc((void**)&s->next, batch * sizeof(int));
    run_cache += batch * sizeof(int);
    cudaMalloc((void**)&s->token, batch * sizeof(int));
    run_cache += batch * sizeof(int);
    printf("total run cache size: %llu bytes, via %fKB, via %fMB, via %fGB\n", run_cache, 
            (float)run_cache / 1024, (float)run_cache / 1024 / 1024, (float)run_cache / 1024 / 1024 / 1024);

    s->next_cpu = (int*)malloc(batch * sizeof(int));
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->att);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
    cudaFree(s->logits);
    cudaFree(s->next);
    cudaFree(s->token);
    free(s->next_cpu);
}

void memory_map_weights(Qwen2Weights *w, Qwen2Config* p, char* ptr) {
    unsigned long long ll;
    unsigned long long ll_bytes;;
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->embed_tokens = (half*)ptr;
    ptr += ll_bytes;
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->q_proj_w = (half*)ptr;
    ptr += ll_bytes;
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->q_proj_b = (half*)ptr;
    ptr += ll_bytes;
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->k_proj_w = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->k_proj_b = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->v_proj_w = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->v_proj_b = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->o_proj = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->gate_proj = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->up_proj = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->down_proj = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->input_layernorm = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->post_attention_layernorm = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
    w->norm = (half*)ptr;
    ptr += ll * sizeof(half);
    cudaMemcpy(&ll, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    cudaMemcpy(&ll_bytes, ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    ptr += sizeof(unsigned long long);
    // printf("++++++++++++--------%llu\n", ll);
    // printf("++++++++++++--------%llu\n", ll_bytes);
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
    printf("file_size is: %ld\n", file_size);
    
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
    printf("model magic is: %d\n", model_magic);

    rcount = fread(&model->config, sizeof(int), sizeof(model->config) / sizeof(int), model_file);
    if (rcount != sizeof(model->config) / sizeof(int)) {
        fprintf(stderr, "Bad read config from model file %s\n", checkpoint_path);
        exit(1);
    }
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

    printf("loading model from disk to host memory......\n");
    for (size_t i = 0; i < n_chuncks; i++) {
        rcount = fread(host_memory + i * chunck_size, sizeof(char), chunck_size, model_file);
        if (rcount != chunck_size) {
            fprintf(stderr, "Bad read model from model file %s\n", checkpoint_path);
            exit(1);
        }
        printf("n_chuncks:%lu the %lu chuncks\n", n_chuncks, i);
    }

    if (tail_size > 0) {
        printf("tail_size:%lu \n", tail_size);
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
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed %ld %s\n", model_size, cudaGetErrorName(err));
    }

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

__global__ 
void get_content_row(float *x, half* embed_tokens, int *token, int batch, int dim) {
    int bidx = blockIdx.x; // batch
    int bidy = blockIdx.y; // dim = 
    int tidx = threadIdx.x;
    int offset_x = bidx * dim + bidy * blockDim.x + tidx;
    int offset_t = bidy * blockDim.x + tidx;
    x[offset_x] = *(embed_tokens + token[bidx] * dim + offset_t);

    // if (thread0()) {
    //     printf("[");
    //     for (int b = 0; b < batch; b++) {
    //         int offset_x = b * dim;
    //         printf("[");
    //         for (int i = 0; i<dim; i++) {
    //             printf("%f, ", x[offset_x + i]);
    //         }
    //         printf("],\n");
    //     }
    //     printf("]\n");
    // }
}

// https://arxiv.org/pdf/1910.07467
__global__
void rmsnorm_forward(float* o, float* x, half *weight, float rms_norm_eps, int batch, int dim) {
    int bidx = blockIdx.x; // batch
    int bidy = blockIdx.y;
    int tid = threadIdx.x; // thread id
    int lid = tid % 32; // lane id
    
    float ss = 0.0f;
    int offset = bidx * dim;
    #pragma unroll
    for (int i = lid; i < dim; i += WARP_THREADS) {
        ss += x[offset + i] * x[offset + i];
    }
    __syncwarp();

    #pragma unroll
    for (int mask = 32 / 2; mask > 0; mask /= 2) {
        ss += __shfl_xor_sync(uint32_t(-1), ss, mask);
        __syncwarp();
    }

    ss /= dim;
    ss += rms_norm_eps;
    ss = rsqrtf(ss);

    int offset_x = bidx * dim + bidy * blockDim.x + tid;
    int offset_w = bidy * blockDim.x + tid;
    int offset_o = bidx * dim + bidy * blockDim.x + tid;
    o[offset_o] = x[offset_x] * ss * __half2float(weight[offset_w]);

    // if (thread0()) {
    //     printf("rmsnorm:\n");
    //     for (int b = 0; b < batch; b++) {
    //         int offset = b * dim;
    //         printf("[");
    //         for (int d = 0; d < dim; d++) {
    //              printf("%f, ", o[offset + d]);
    //         }
    //         printf("],\n");
    //     }
    // }
}

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
__global__
void linear_forward(float* output, float* input, half *weight, half* bias, int batch, int in_features, int out_features) {
    int bidy = blockIdx.y;
    int tid = threadIdx.x;
    int laneid = tid % WARP_THREADS;
    int num_per_thread = in_features / WARP_THREADS;
    // int col = laneid * num_per_thread;
    int row = bidy * WARPGROUP_WARPS + threadIdx.y;

    float4* input4 = reinterpret_cast<float4*>(input);
    float4* weight4 = reinterpret_cast<float4*>(weight + row * in_features);

    float ss = 0.0f;

    for (int i = 0; i < (num_per_thread >> 3); i++) {
        int col_i = laneid + i * WARP_THREADS;
        if (col_i < in_features >> 3) {
        float4 a = input4[2 * col_i];
        float4 b = weight4[col_i];
        half2* b_0 = reinterpret_cast<half2*>(&b.x);
        half2* b_1 = reinterpret_cast<half2*>(&b.y);
        half2* b_2 = reinterpret_cast<half2*>(&b.z);
        half2* b_3 = reinterpret_cast<half2*>(&b.w);

        ss += a.x * __half2float(b_0->x);
        ss += a.y * __half2float(b_0->y);
        ss += a.z * __half2float(b_1->x);
        ss += a.w * __half2float(b_1->y);
        float4 c = input4[2 * col_i + 1];
        ss += c.x * __half2float(b_2->x);
        ss += c.y * __half2float(b_2->y);
        ss += c.z * __half2float(b_3->x);
        ss += c.w * __half2float(b_3->y);
        }
    }

    ss += __shfl_down_sync(0xffffffff, ss, 16);
    ss += __shfl_down_sync(0xffffffff, ss, 8);
    ss += __shfl_down_sync(0xffffffff, ss, 4);
    ss += __shfl_down_sync(0xffffffff, ss, 2);
    ss += __shfl_down_sync(0xffffffff, ss, 1);

    if (laneid == 0) {
        output[row] = ss;
    }
                
    if (bias != NULL) {
        output[row] += __half2float(bias[row]);
    } 

    // if (thread0()) {
    //     printf("linear:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int i = 0; i < out_features; i++) {
    //             printf("%f, ", output[b * out_features + i]);
    //         }
    //         printf("]\n");
    //     }
    //     printf("]\n");
    // }
}

__global__ 
void rope_forward(float *q, float rope_freq_constant, int batch, int q_heads, int head_dim, int pos) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    // int lid = tid % WARP_THREADS;

    int offset = b * q_heads * head_dim + h * head_dim;

    for (int hd = tid; hd < head_dim / 2; hd += WARPGROUP_THREADS) {
        float v0 = q[offset + hd];
        float v1 = q[offset + hd + head_dim / 2];

        float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
        // printf("sl=%d %d=%f ", sl, hd, sl * freq);
        float cos_val = cosf(pos * freq);
        float sin_val = sinf(pos * freq);
        // printf("sl=%d %d=%f ", sl, hd, sin_val);
        q[offset + hd] = v0 * cos_val - v1 * sin_val;
        q[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
    }

    // if (thread0()) {
    //     printf("rope: \n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int h = 0; h < q_heads; h++) {
    //             printf("[");    
    //             int offset = b * q_heads * head_dim + h * head_dim;
    //             for (int hd = 0; hd < head_dim; hd++) {     
    //                 printf("%f,", q[offset + hd]);
    //             }
    //             printf("],\n");
    //         }
    //         printf("],\n");
    //     }
    // }
}

// https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
__global__
void group_flash_attention_forward(float* output, float *q, float *key_cache, float *value_cache, float *att,
                             int batch, int q_heads, int k_heads, int head_dim, int max_q_heads, int max_kv_heads, int max_seq_len, 
                             int num_transformer_layers, int layer_idx, int pos) {
    int num_groups = q_heads / k_heads;
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int lid = tid % WARP_THREADS;
    int kNThreads = blockDim.x;
    extern __shared__ float smem_[];

    float *d = smem_;
    float *m = smem_ + 1;
    float *o = smem_ + 2;
    
    *d = 1.0f;

    int offset_q = b * q_heads * head_dim + h * head_dim;
    int offset_k = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                 + layer_idx * max_seq_len * max_kv_heads * head_dim 
                 + 0 * max_kv_heads * head_dim
                 + (h / num_groups)  * head_dim;
        
    float score = 0.0f;
    for (int i = lid; i < head_dim; i += WARP_THREADS){
        score += q[offset_q + i] * key_cache[offset_k + i];
    }

    __syncwarp();

    #pragma unroll
    for (int mask = 32 / 2; mask > 0; mask /= 2) {
        score += __shfl_down_sync(uint32_t(-1), score, mask);
        // __syncwarp();
    }
    __syncwarp();
    if (lid == 0) {
        score /= sqrtf((float)head_dim);
        *m = score;
    }
    

    int offset_o = b * q_heads * head_dim + h * head_dim;
    for (int lv = tid; lv < head_dim; lv += WARPGROUP_THREADS){
        int offset_v = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                     + layer_idx * max_seq_len * max_kv_heads * head_dim 
                     + 0 * max_kv_heads * head_dim
                     + (h / num_groups) * head_dim;
        o[lv] = value_cache[offset_v + lv];
        output[offset_o + lv] = o[lv];
    }

    
    
    // flash attention
    float m_i1 = 0.0f;
    float m_i = 0.0f;
    float d_i1 = 0.0f;
    float d_i = 0.0f;
    float o_i1 = 0.0f;
    float o_i = 0.0f;

    __syncthreads();
    for (int lk = 1; lk < pos + 1; lk++) {
        int offset_k = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim 
                         + lk * max_kv_heads * head_dim
                         + (h / num_groups)  * head_dim;
        
        // score = 0.0f;
        // for (int i = 0; i < head_dim; i++) {
        //     score += q[offset_q + i] * key_cache[offset_k + i];
        // }
        score = 0.0f;
        for (int i = lid; i < head_dim; i += WARP_THREADS){
            score += q[offset_q + i] * key_cache[offset_k + i];
        }

        __syncwarp();

        #pragma unroll
        for (int mask = 32 / 2; mask > 0; mask /= 2) {
            score += __shfl_xor_sync(uint32_t(-1), score, mask);
            // __syncwarp();
        }

        score /= sqrtf((float)head_dim);

        // att[offset_att + lk] = score;
        m_i1 = *m;
        m_i = m_i1;
        if (score > m_i1) {
            m_i = score;
        }

        d_i1 = *d;

        d_i = d_i1 * __expf(m_i1 - m_i) + __expf(score - m_i);

        __syncthreads();
        for (int lv = tid; lv < head_dim; lv += kNThreads){
            o_i1 = o[lv];
            int offset_v = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim 
                         + lk * max_kv_heads * head_dim
                         + (h / num_groups) * head_dim;
            o_i = o_i1 * (d_i1 * __expf(m_i1 - m_i) / d_i) + __expf(score - m_i) / d_i * value_cache[offset_v + lv];
            o[lv] = o_i;
            output[offset_o + lv] = o_i;
        }

        *d = d_i;
        *m = m_i;
    }

    // if (thread0()) {
    //     printf("group_attention:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int d = 0; d < q_heads * head_dim; d++) {
    //             int offset = b * q_heads * head_dim;
    //                 printf("%f, ",output[offset + d]);
    //         }
    //         printf("],\n");
    //     }
    // }
}

__global__
void residual_forward(float *x, float *xb, int batch, int dim) {
    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;
    int kNThreads = blockDim.x;
    int offset = b * dim + bidy * kNThreads + tid;

    x[offset] += xb[offset];

    // if (thread0()) {
    //     printf("residual:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int i = 0; i < dim; i++) {
    //             int offset_x = b * dim + i;
    //             printf("%f, ", x[offset_x]);
    //         }
    //         printf("]\n");
    //     }
    // }
}

// https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
__global__
void silu_forward(float *hb, float* hb2, int batch, int intermediate_dim) {

    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;

    int offset = b * intermediate_dim + bidy * WARPGROUP_THREADS + tid;

    float val = hb[offset];
    val *= 1.0f / (1.0f + __expf(-val));
    val *= hb2[offset];
    hb[offset] = val;

    // if (thread0()) {
    //     printf("silu:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int i = 0; i < intermediate_dim; i++) {
    //             printf("%f, ", hb[b * intermediate_dim + i]);
    //         }
    //         printf("]\n");
    //     }
    // }
}

__global__
void logits_forward(float* output, float* input, half *weight, half* bias, int batch, int in_features, int out_features) {
    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;
    int kNThreads = blockDim.x;

    int out = bidy * kNThreads + tid;
    int offset_out = b * out_features + out;
    int offset_bias = out;
    float value = 0.0f;
    for (int in = 0; in < in_features; in++) {
        int offset_in = b * in_features + in;
        int offset_weight = out * in_features + in;
        value += input[offset_in] * __half2float(weight[offset_weight]);
    }
    output[offset_out] = value;
    if (bias != NULL) {
        output[offset_out] += __half2float(bias[offset_bias]);
    } 

    // if (thread0()) {
    //     printf("logits: \n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int i = 0; i < out_features; i++) {
    //             printf("%f, ", output[b * out_features + i]);
    //         }
    //         printf("]\n");
    //     }
    // }
}

__global__
void argmax_forward(int* output, float* input, int batch, int dim) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lid = tid % 32; // lane id

    int offset = b * dim;

    int max_i = lid;
    float max_val = input[offset + max_i];
    
    for (int i = lid; i < dim; i += WARP_THREADS) { 
        if (input[offset + i] > max_val) {
            max_val = input[offset + i];
            max_i = i;
        }
    }

    __syncwarp();

    #pragma unroll
    for (int mask = 32 / 2; mask > 0; mask /= 2) {
        int shfl_i = __shfl_xor_sync(uint32_t(-1), max_i, mask);
        if (input[offset + shfl_i] > max_val) {
            max_val = input[offset + shfl_i];
            max_i = shfl_i;
        }
        __syncwarp();
    }
    
    output[b] = max_i;

    // if (thread0()) {
    //     printf("argmax:\n");
    //     printf("[");
    //     for (int b = 0; b < batch; b++) {
    //         printf("%d, ", output[b]);
    //     }
    //     printf("]\n");
    // }
}

void* qwen2_forward(Context *ctx, Qwen2* qwen2, int *token, int batch, int pos) {
    Qwen2Config *p = &qwen2->config;
    Qwen2Weights *w = &qwen2->weights;
    RunState* s = &qwen2->state;

    s->flops = 0;
    s->flops_sfu = 0;
    int max_seq_len = s->max_seq_len;
    float *x = s->x;

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
    
    // printf("qwen2_forward pos:%d, batch:%d, hidden_size:%d \n", pos, batch, hidden_size);
    cudaError_t err;
    
    get_content_row<<<dim3(batch, hidden_size/WARPGROUP_THREADS), WARPGROUP_THREADS>>>(x, w->embed_tokens, token, batch, hidden_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorName(err));
    }

    cudaDeviceSynchronize();
    dim3 linear_grid_dim;
    dim3 linear_block_dim(WARP_THREADS, WARPGROUP_WARPS);
    // for(int l = 0; l < 1; l++) {
    for(int l = 0; l < p->num_hidden_layers; l++) {
        // attn_norm
        rmsnorm_forward<<<dim3(batch, hidden_size / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(s->xb, s->x, w->input_layernorm + l*hidden_size, rms_norm_eps, batch, hidden_size);

        int offset_k = l * max_seq_len * num_key_value_heads * head_dim 
                         + pos * num_key_value_heads * head_dim;
        int offset_v = l * max_seq_len * num_key_value_heads * head_dim 
                         + pos * num_key_value_heads * head_dim;
        s->k = s->key_cache + offset_k;
        s->v = s->value_cache + offset_v;

        // batch * p->num_hidden_layers * seq_len * num_heads * head_dim
        linear_grid_dim = dim3(batch, num_heads * head_dim / WARPGROUP_WARPS);
        linear_forward<<<linear_grid_dim, linear_block_dim>>>(s->q, s->xb, w->q_proj_w + l * hidden_size * (num_heads * head_dim), w->q_proj_b + l * (num_heads * head_dim), batch, hidden_size, num_heads * head_dim);
        // cudaDeviceSynchronize();
        // exit(-1);
        linear_grid_dim = dim3(batch, num_key_value_heads * head_dim / WARPGROUP_WARPS);
        linear_forward<<<linear_grid_dim, linear_block_dim>>>(s->k, s->xb, w->k_proj_w + l * hidden_size * (num_key_value_heads * head_dim), w->k_proj_b + l * (num_key_value_heads * head_dim), batch, hidden_size, num_key_value_heads * head_dim);
        linear_grid_dim = dim3(batch, num_key_value_heads * head_dim / WARPGROUP_WARPS);
        linear_forward<<<linear_grid_dim, linear_block_dim>>>(s->v, s->xb, w->v_proj_w + l * hidden_size * (num_key_value_heads * head_dim), w->v_proj_b + l * (num_key_value_heads * head_dim), batch, hidden_size, num_key_value_heads * head_dim);
        
        rope_forward<<<dim3(batch, num_heads), WARPGROUP_THREADS>>>(s->q, rope_theta, batch, num_heads, head_dim, pos);

        rope_forward<<<dim3(batch, num_key_value_heads), WARPGROUP_THREADS>>>(s->k, rope_theta, batch, num_heads, head_dim, pos);

        // group attention
        group_flash_attention_forward<<<dim3(batch, num_heads), WARPGROUP_THREADS>>>(s->xb, s->q, s->key_cache, s->value_cache, s->att,
                             batch, num_heads, num_key_value_heads, head_dim, num_heads, num_key_value_heads, max_seq_len, 
                             num_hidden_layers, l, pos);

        linear_grid_dim = dim3(batch, hidden_size / WARPGROUP_WARPS);
        linear_forward<<<linear_grid_dim, linear_block_dim>>>(s->xb2, s->xb, w->o_proj + l * (num_heads * head_dim) * hidden_size, NULL, batch, num_heads * head_dim, hidden_size);

        residual_forward<<<dim3(batch, hidden_size / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(s->x, s->xb2, batch, hidden_size);

        // ffn_norm
        rmsnorm_forward<<<dim3(batch, hidden_size / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(s->xb, s->x, w->post_attention_layernorm + l*hidden_size, rms_norm_eps, batch, hidden_size);

        linear_grid_dim = dim3(batch, intermediate_size / WARPGROUP_WARPS);
        linear_forward<<<linear_grid_dim, linear_block_dim>>>(s->hb, s->xb, w->gate_proj + l*intermediate_size*hidden_size, NULL, batch, hidden_size, intermediate_size);
        linear_grid_dim = dim3(batch, intermediate_size / WARPGROUP_WARPS);
        linear_forward<<<linear_grid_dim, linear_block_dim>>>(s->hb2, s->xb, w->up_proj + l*intermediate_size*hidden_size, NULL, batch, hidden_size, intermediate_size);
  
        silu_forward<<<dim3(batch, intermediate_size / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(s->hb, s->hb2, batch, intermediate_size);

        linear_grid_dim = dim3(batch, hidden_size / WARPGROUP_WARPS);
        linear_forward<<<linear_grid_dim, linear_block_dim>>>(s->xb, s->hb, w->down_proj + l* hidden_size * intermediate_size, NULL, batch, intermediate_size, hidden_size);

        residual_forward<<<dim3(batch, hidden_size / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(s->x, s->xb, batch, hidden_size);

        // cudaDeviceSynchronize();
    }

    rmsnorm_forward<<<dim3(batch, hidden_size / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(s->x, s->x, w->norm, rms_norm_eps, batch, hidden_size);
    
    logits_forward<<<dim3(batch, vocab_size / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(s->logits, s->x, w->lm_head, NULL, batch, hidden_size, vocab_size);

    return s->logits;
}


Qwen2 py_model;

void c_init(int batch, int max_seq_len, const char *checkpoint_path) {
    printf("checkpoint_path: %s\n", checkpoint_path);
    if (checkpoint_path == NULL) {
        checkpoint_path = "qwen1.5-0.5B.bin";
    }
    // const char *checkpoint_path = "qwen1.5-0.5B.bin";
    qwen2_build_from_checkpoint(&py_model, checkpoint_path);
    py_model.state.batch = batch;
    py_model.state.max_seq_len = max_seq_len;
    malloc_run_state(&py_model.state, &py_model.config);
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
    qwen2_forward(&ctx, &py_model, s->token, batch, pos);
    // cudaDeviceSynchronize();
    argmax_forward<<<s->batch, WARPGROUP_THREADS>>>(s->next, s->logits, s->batch, py_model.config.vocab_size);
    cudaDeviceSynchronize();

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
