#pragma once

#include <stdint.h>
#include <math.h>
#include "utils.h"

#ifdef AVX512_FWD
#include <immintrin.h>
#endif

#ifdef NEON_FWD
#include <arm_neon.h>
#endif

namespace ld_infer {
namespace cpu {
namespace group_attention {
    
void group_attention_v1_fwd(float* output, float *q, float *key_cache, float *value_cache, float *att,
                             int batch, int q_heads, int k_heads, int head_dim, int max_q_heads, int max_kv_heads, int max_seq_len, 
                             int num_transformer_layers, int layer_idx, int pos) {
    int num_groups = q_heads / k_heads;

    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < q_heads; h++) {
            int offset_att = b * max_q_heads * max_seq_len + h * max_seq_len;
            int offset_q = b * q_heads * head_dim + h * head_dim;

            for (int lk = 0; lk < pos + 1; lk++) {

                int offset_k = layer_idx * max_seq_len * batch * max_kv_heads * head_dim 
                    + lk * batch * max_kv_heads * head_dim
                    + b * max_kv_heads * head_dim
                    + (h / num_groups)  * head_dim;
                         

                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q[offset_q + i] * key_cache[offset_k + i];
                }

                score /= sqrtf((float)head_dim);
                att[offset_att + lk] = score;
            }
            // printf("\n");
            float max_val = att[offset_att];
            for (int lk = 1; lk < pos + 1; lk++) { 
                if (att[offset_att + lk] > max_val) {
                    max_val = att[offset_att + lk];
                }
            }
            float ss = 0.0f;
            for (int lk = 0; lk < pos + 1; lk++) { 
                ss += expf(att[offset_att + lk] - max_val);
            }

            for (int lk = 0; lk < pos + 1; lk++) { 
                att[offset_att + lk] = expf(att[offset_att + lk] - max_val) / ss;
            }
                
            int offset_o = b * q_heads * head_dim + h * head_dim;
            for (int lv = 0; lv < head_dim; lv++){
                float sv = 0.0f;
                for (int k = 0; k < pos + 1; k++) { 
                    int offset_v = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim 
                         + k * max_kv_heads * head_dim
                         + (h / num_groups) * head_dim;
                    sv += att[offset_att + k] * (value_cache[offset_v + lv]);
                }
                output[offset_o + lv] = sv;
            }
        }
    }

#ifdef FLASH_ATTENTION_DEBUG
    printf("group_attention:\n");
    for (int b = 0; b < batch; b++) {
         printf("[");
        for (int d = 0; d < q_heads * head_dim; d++) {
            int offset = b * q_heads * head_dim;
                printf("%f, ", output[offset + d]);
        }
        printf("],\n");
    }
#endif // FLASH_ATTENTION_DEBUG
}

#ifdef AVX512_FWD
void group_attention_avx512_fwd(float* output, float *q, float *key_cache, float *value_cache, float *att,
                             int batch, int q_heads, int k_heads, int head_dim, int max_q_heads, int max_kv_heads, int max_seq_len, 
                             int num_transformer_layers, int layer_idx, int pos) {
    int num_groups = q_heads / k_heads;

    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < q_heads; h++) {
            int offset_att = b * max_q_heads * max_seq_len + h * max_seq_len;
            int offset_q = b * q_heads * head_dim + h * head_dim;

            for (int lk = 0; lk < pos + 1; lk++) {

                int offset_k = layer_idx * max_seq_len * batch * max_kv_heads * head_dim 
                    + lk * batch * max_kv_heads * head_dim
                    + b * max_kv_heads * head_dim
                    + (h / num_groups)  * head_dim;
                         

                float score = 0.0f;
                for (int i = 0; i < head_dim; i += 16) {
                    score += q[offset_q + i] * key_cache[offset_k + i];
                    score += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_loadu_ps(q + offset_q + i), _mm512_loadu_ps(key_cache + offset_k + i)));

                }

                score /= sqrtf((float)head_dim);
                att[offset_att + lk] = score;
            }

            float max_val = att[offset_att];
            for (int lk = 1; lk < pos + 1; lk += 16) { 
                __m512 max_avx = _mm512_loadu_ps(att + offset_att + lk);
                for (int i = 0; i < 16 && lk + i < pos + 1; i++) {
                    if (max_avx[i] > max_val) {
                        max_val = max_avx[i];
                    }
                }  
            }

            float ss = 0.0f;
            for (int lk = 0; lk < pos + 1; lk += 16) { 
                __m512 ss_avx = _mm512_loadu_ps(att + offset_att + lk);
                for (int i = 0; i < 16 && lk + i < pos + 1; i++) {
                    ss += expf(ss_avx[i] - max_val);
                } 
            }

            for (int lk = 0; lk < pos + 1; lk += 16) { 
                __m512 a_avx = _mm512_loadu_ps(att + offset_att + lk);
                for (int i = 0; i < 16 && lk + i < pos + 1; i++) {
                    a_avx[i] = expf(a_avx[i] - max_val) / ss;
                    att[offset_att + lk + i] = a_avx[i];
                } 
                // _mm512_storeu_ps(att + offset_att + lk, a_avx);
            }
                
            int offset_o = b * q_heads * head_dim + h * head_dim;
            for (int lv = 0; lv < head_dim; lv++){
                float sv = 0.0f;
                for (int k = 0; k < pos + 1; k++) { 
                    int offset_v = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim 
                         + k * max_kv_heads * head_dim
                         + (h / num_groups) * head_dim;
                    sv += att[offset_att + k] * (value_cache[offset_v + lv]);
                }
                output[offset_o + lv] = sv;
            }
        }
    }

#ifdef FLASH_ATTENTION_DEBUG
    printf("group_attention:\n");
    for (int b = 0; b < batch; b++) {
         printf("[");
        for (int d = 0; d < q_heads * head_dim; d++) {
            int offset = b * q_heads * head_dim;
                printf("%f, ", output[offset + d]);
        }
        printf("],\n");
    }
#endif // FLASH_ATTENTION_DEBUG
}
#endif

#ifdef NEON_FWD
void group_attention_neon_fwd(float* output, float *q, float *key_cache, float *value_cache, float *att,
                             int batch, int q_heads, int k_heads, int head_dim, int max_q_heads, int max_kv_heads, int max_seq_len, 
                             int num_transformer_layers, int layer_idx, int pos) {
    int num_groups = q_heads / k_heads;

    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < q_heads; h++) {
            int offset_att = b * max_q_heads * max_seq_len + h * max_seq_len;
            int offset_q = b * q_heads * head_dim + h * head_dim;

            for (int lk = 0; lk < pos + 1; lk++) {

                int offset_k = layer_idx * max_seq_len * batch * max_kv_heads * head_dim 
                    + lk * batch * max_kv_heads * head_dim
                    + b * max_kv_heads * head_dim
                    + (h / num_groups)  * head_dim;
                         

                float score = 0.0f;
                float32x4_t d_neon;
                float32x2_t sum_neon;
                for (int i = 0; i < head_dim; i += 4) {
                    d_neon = vmulq_f32(vld1q_f32(q + offset_q + i), vld1q_f32(key_cache + offset_k + i));
                    sum_neon = vadd_f32(vget_low_f32(d_neon), vget_high_f32(d_neon)); 
                    sum_neon = vpadd_f32(sum_neon, sum_neon); 
                    score += vget_lane_f32(sum_neon, 0);
                }

                score /= sqrtf((float)head_dim);
                att[offset_att + lk] = score;
            }

            float max_val = att[offset_att];
            for (int lk = 1; lk < pos + 1; lk += 4) { 
                float32x4_t max_neon = vld1q_f32(att + offset_att + lk);
                for (int i = 0; i < 4 && lk + i < pos + 1; i++) {
                    if (max_neon[i] > max_val) {
                        max_val = max_neon[i];
                    }
                }  
            }

            float ss = 0.0f;
            for (int lk = 0; lk < pos + 1; lk += 4) { 
                float32x4_t ss_neon = vld1q_f32(att + offset_att + lk);
                for (int i = 0; i < 4 && lk + i < pos + 1; i++) {
                    ss += expf(ss_neon[i] - max_val);
                } 
            }
                
            for (int lk = 0; lk < pos + 1; lk += 4) { 
                float32x4_t a_neon = vld1q_f32(att + offset_att + lk);
                for (int i = 0; i < 4 && lk + i < pos + 1; i++) {
                    a_neon[i] = expf(a_neon[i] - max_val) / ss;
                    att[offset_att + lk + i] = a_neon[i];
                } 
                // vst1q_f32(att + offset_att + lk, a_avx);
            }

            int offset_o = b * q_heads * head_dim + h * head_dim;
            for (int lv = 0; lv < head_dim; lv++){
                float sv = 0.0f;
                for (int k = 0; k < pos + 1; k++) { 
                    int offset_v = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim 
                         + k * max_kv_heads * head_dim
                         + (h / num_groups) * head_dim;
                    sv += att[offset_att + k] * (value_cache[offset_v + lv]);
                }
                output[offset_o + lv] = sv;
            }
        }
    }

#ifdef FLASH_ATTENTION_DEBUG
    printf("group_attention:\n");
    for (int b = 0; b < batch; b++) {
         printf("[");
        for (int d = 0; d < q_heads * head_dim; d++) {
            int offset = b * q_heads * head_dim;
                printf("%f, ", output[offset + d]);
        }
        printf("],\n");
    }
#endif // FLASH_ATTENTION_DEBUG
}
#endif

void group_attention_fwd(float* output, float *q, float *key_cache, float *value_cache, float *att,
                             int batch, int q_heads, int k_heads, int head_dim, int max_q_heads, int max_kv_heads, int max_seq_len, 
                             int num_transformer_layers, int layer_idx, int pos) {
#ifdef AVX512_FWD
    group_attention_avx512_fwd(output, q, key_cache, value_cache, att,
                        batch, q_heads, k_heads, head_dim, max_q_heads, max_kv_heads, max_seq_len, 
                        num_transformer_layers, layer_idx, pos);
#elif NEON_FWD
    group_attention_neon_fwd(output, q, key_cache, value_cache, att,
                        batch, q_heads, k_heads, head_dim, max_q_heads, max_kv_heads, max_seq_len, 
                        num_transformer_layers, layer_idx, pos);
#else
    group_attention_v1_fwd(output, q, key_cache, value_cache, att,
                        batch, q_heads, k_heads, head_dim, max_q_heads, max_kv_heads, max_seq_len, 
                        num_transformer_layers, layer_idx, pos);
#endif                            
}

} // namespace group_attention
} // namespace cpu
} // namespace ld_infer