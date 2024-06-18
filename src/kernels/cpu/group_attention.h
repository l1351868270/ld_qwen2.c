#pragma once

#include <stdint.h>
#include <math.h>
#include "utils.h"

namespace ld_infer {
namespace cpu {
namespace group_attention {
    
void group_attention_fwd(float* output, float *q, float *key_cache, float *value_cache, float *att,
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

} // namespace group_attention
} // namespace cpu
} // namespace ld_infer