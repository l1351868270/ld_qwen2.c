#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include "utils.cuh"

namespace ld_infer {
namespace cuda {
namespace flash_attention_half {

// https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
__global__
void group_flash_attention_half_fwd(half* output, half *half_q, half *key_cache, half *value_cache,
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
    int offset_k = layer_idx * max_seq_len * batch * max_kv_heads * head_dim 
                 + 0 * batch * max_kv_heads * head_dim
                 + b * max_kv_heads * head_dim
                 + (h / num_groups)  * head_dim;
        
    float score = 0.0f;
    for (int i = lid; i < head_dim; i += WARP_THREADS){
        score += __half2float(half_q[offset_q + i]) * __half2float(key_cache[offset_k + i]);
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
        int offset_v = layer_idx * max_seq_len * batch * max_kv_heads * head_dim 
                    + 0 * batch * max_kv_heads * head_dim
                    + b * max_kv_heads * head_dim
                    + (h / num_groups) * head_dim;
        o[lv] = __half2float(value_cache[offset_v + lv]);
        output[offset_o + lv] = __float2half(o[lv]);
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
        int offset_k = layer_idx * max_seq_len * batch * max_kv_heads * head_dim 
                    + lk * batch * max_kv_heads * head_dim
                    + b * max_kv_heads * head_dim
                    + (h / num_groups)  * head_dim;
        
        // score = 0.0f;
        // for (int i = 0; i < head_dim; i++) {
        //     score += q[offset_q + i] * key_cache[offset_k + i];
        // }
        score = 0.0f;
        for (int i = lid; i < head_dim; i += WARP_THREADS){
            score += __half2float(half_q[offset_q + i]) * __half2float(key_cache[offset_k + i]);
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
            int offset_v = layer_idx * max_seq_len * batch * max_kv_heads * head_dim 
                         + lk * batch * max_kv_heads * head_dim
                         + b * max_kv_heads * head_dim
                         + (h / num_groups) * head_dim;
            o_i = o_i1 * (d_i1 * __expf(m_i1 - m_i) / d_i) + __expf(score - m_i) / d_i * __half2float(value_cache[offset_v + lv]);
            o[lv] = o_i;
            output[offset_o + lv] = __float2half(o_i);
        }

        *d = d_i;
        *m = m_i;
    }

#ifdef FLASH_ATTENTION_DEBUG
    if (thread0()) {
        int batch = gridDim.x;
        printf("group_attention:\n");
        for (int b = 0; b < batch; b++) {
            printf("[");
            for (int d = 0; d < q_heads * head_dim; d++) {
                int offset = b * q_heads * head_dim;
                    printf("%f, ", __half2float(output[offset + d]));
            }
            printf("],\n");
        }
    }
#endif // FLASH_ATTENTION_DEBUG

}

void flash_attention_half_fwd_launch(half* output, half *q, half *key_cache, half *value_cache,
                             int batch, int q_heads, int k_heads, int head_dim, int max_q_heads, int max_kv_heads, int max_seq_len, 
                             int num_transformer_layers, int layer_idx, int pos) {

    group_flash_attention_half_fwd<<<dim3(batch, q_heads), WARPGROUP_THREADS>>>(output, q, key_cache, value_cache, 
                             batch, q_heads, k_heads, head_dim, max_q_heads, max_kv_heads, max_seq_len, 
                             num_transformer_layers, layer_idx, pos);
}

} // namespace flash_attention_half
} // namespace cuda
} // namespace ld_infer