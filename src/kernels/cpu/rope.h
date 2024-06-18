#pragma once

#include <math.h>
#include "utils.h"

#ifdef AVX512_FWD
#include <immintrin.h>
#endif

namespace ld_infer {
namespace cpu {
namespace rope {

void ropeV1_fwd(float *q, float rope_freq_constant, int batch, int num_heads, int head_dim, int pos) {

    int elem_per_cpu = head_dim / 2 / NUM_CPUS;
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < num_heads; h++) {
            int offset = b * num_heads * head_dim + h * head_dim;
            int t;
            #pragma omp parallel for private(t)
            for (t = 0; t < NUM_CPUS; t++) {
                for (int d = 0; d < elem_per_cpu; d++) {
                    // https://arxiv.org/pdf/2104.09864
                    int hd = t * elem_per_cpu + d;
                    float v0 = q[offset + hd];
                    float v1 = q[offset + hd + head_dim / 2];

                    float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
                    float cos_val = cosf(pos * freq);
                    float sin_val = sinf(pos * freq);
                    // printf("sl=%d %d=%f ", sl, hd, sin_val);
                    q[offset + hd] = v0 * cos_val - v1 * sin_val;
                    q[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
                }
            }
        }
    }

#ifdef ROPE_DEBUG
    printf("rope: \n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int h = 0; h < num_heads; h++) {
            printf("[");    
            int offset = b * num_heads * head_dim + h * head_dim;
            for (int hd = 0; hd < head_dim; hd++) {     
                printf("%f,", q[offset + hd]);
            }
            printf("],\n");
        }
        printf("],\n");
    }
#endif // ROPE_DEBUG

}

void ropeV2_fwd(float *q, float rope_freq_constant, int batch, int num_heads, int head_dim, int pos) {
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < num_heads; h++) {
            int offset = b * num_heads * head_dim + h * head_dim;
            int hd;
            #pragma omp parallel for private(hd)
            for (hd = 0; hd < head_dim / 2; hd++) {
                    // https://arxiv.org/pdf/2104.09864
                    float v0 = q[offset + hd];
                    float v1 = q[offset + hd + head_dim / 2];

                    float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
                    float cos_val = cosf(pos * freq);
                    float sin_val = sinf(pos * freq);
                    // printf("sl=%d %d=%f ", sl, hd, sin_val);
                    q[offset + hd] = v0 * cos_val - v1 * sin_val;
                    q[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
            }
        }
    }

#ifdef ROPE_DEBUG
    printf("rope: \n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int h = 0; h < num_heads; h++) {
            printf("[");    
            int offset = b * num_heads * head_dim + h * head_dim;
            for (int hd = 0; hd < head_dim; hd++) {     
                printf("%f,", q[offset + hd]);
            }
            printf("],\n");
        }
        printf("],\n");
    }
#endif // ROPE_DEBUG

}

void rope_fwd(float *q, float rope_freq_constant, int batch, int num_heads, int head_dim, int pos) {
    ropeV2_fwd(q, rope_freq_constant, batch, num_heads, head_dim, pos);
}

} // namespace rope
} // namespace cpu
} // namespace ld_infer