#pragma once

#include <math.h>
#include "utils.h"
#include <omp.h>

#ifdef AVX512_FWD
#include <immintrin.h>
#endif

#ifdef NEON_FWD
#include <arm_neon.h>
#endif

namespace ld_infer {
namespace cpu {
namespace rope {

void ropeV1_fwd(float *q, float rope_freq_constant, int batch, int num_heads, int head_dim, int pos) {

    int elem_per_cpu = head_dim / 2 / utils::NUM_CPUS;
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < num_heads; h++) {
            int offset = b * num_heads * head_dim + h * head_dim;
            int t;
            #pragma omp parallel for private(t)
            for (t = 0; t < utils::NUM_CPUS; t++) {
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
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int b = 0; b < batch; b++) {
        // #pragma omp parallel for
        for (int h = 0; h < num_heads; h++) {
            int offset = b * num_heads * head_dim + h * head_dim;
            // #pragma omp parallel for
            for (int hd = 0; hd < head_dim / 2; hd++) {
                // https://arxiv.org/pdf/2104.09864
                float v0 = q[offset + hd];
                float v1 = q[offset + hd + head_dim / 2];

                float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
                float cos_val = cosf(pos * freq);
                float sin_val = sinf(pos * freq);
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


#ifdef AVX512_FWD
void rope_avx512_fwd(float *q, float rope_freq_constant, int batch, int num_heads, int head_dim, int pos) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < num_heads; h++) {
            int offset = b * num_heads * head_dim + h * head_dim;
            int hd;
            #pragma omp parallel for private(hd)
            for (hd = 0; hd < head_dim / 2; hd += 16) {
                // https://arxiv.org/pdf/2104.09864
                __m512 v0_avx = _mm512_loadu_ps(q + offset + hd);  
                __m512 v1_avx = _mm512_loadu_ps(q + offset + hd + head_dim / 2);
                for (int i = 0; i < 16; i++) {
                    float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * (hd + i)) / head_dim));
                    float cos_val = cosf(pos * freq);
                    float sin_val = sinf(pos * freq);
                    q[offset + hd + i] = v0_avx[i] * cos_val - v1_avx[i] * sin_val;
                    q[offset + head_dim / 2 + hd + i] = v1_avx[i] * cos_val + v0_avx[i] * sin_val;
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
#endif

#ifdef NEON_FWD
void rope_neon_fwd(float *q, float rope_freq_constant, int batch, int num_heads, int head_dim, int pos) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < num_heads; h++) {
            int offset = b * num_heads * head_dim + h * head_dim;
            int hd;
            #pragma omp parallel for private(hd)
            for (hd = 0; hd < head_dim / 2; hd += 4) {
                // https://arxiv.org/pdf/2104.09864
                float32x4_t v0_neon = vld1q_f32(q + offset + hd);  
                float32x4_t v1_neon = vld1q_f32(q + offset + hd + head_dim / 2);
                for (int i = 0; i < 4; i++) {
                    float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * (hd + i)) / head_dim));
                    float cos_val = cosf(pos * freq);
                    float sin_val = sinf(pos * freq);
                    q[offset + hd + i] = v0_neon[i] * cos_val - v1_neon[i] * sin_val;
                    q[offset + head_dim / 2 + hd + i] = v1_neon[i] * cos_val + v0_neon[i] * sin_val;
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
#endif


void rope_fwd(float *q, float rope_freq_constant, int batch, int num_heads, int head_dim, int pos) {
#ifdef AVX512_FWD
    rope_avx512_fwd(q, rope_freq_constant, batch, num_heads, head_dim, pos);
#elif NEON_FWD
    rope_neon_fwd(q, rope_freq_constant, batch, num_heads, head_dim, pos);
#elif OPENMP_V1
    ropeV1_fwd(q, rope_freq_constant, batch, num_heads, head_dim, pos);
#else
    // double tdata = omp_get_wtime();
    ropeV2_fwd(q, rope_freq_constant, batch, num_heads, head_dim, pos);
    // tdata = omp_get_wtime() - tdata;
    // printf("batch=%d, num_heads=%d, head_dim=%d,  in %f secs\n", batch, num_heads, head_dim, tdata);
#endif
}

} // namespace rope
} // namespace cpu
} // namespace ld_infer