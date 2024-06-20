#pragma once


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
namespace linear {

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
void linearV1_fwd(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
    int elem_per_cpu = out_features / utils::NUM_CPUS;
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < utils::NUM_CPUS; t++) {
            for (int d = 0; d < elem_per_cpu; d++) {
                int offset_out = b * out_features + t * elem_per_cpu + d;
                int offset_bias = t * elem_per_cpu + d;
                float value = 0.0f;
                for (int in = 0; in < in_features; in++) {
                    int offset_in = b * in_features + in;
                    int offset_weight = (t * elem_per_cpu + d) * in_features + in;                 
                    value += input[offset_in] * weight[offset_weight];
                }
                output[offset_out] = value;

                if (bias != NULL) {
                    output[offset_out] += bias[offset_bias];
                } 
            }
        }
    }

#ifdef LINEAR_DEBUG
    printf("linear_forward:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < out_features; i++) {
            printf("%f, ", output[b * out_features + i]);
        }
        printf("]\n");
    }
#endif
}

void linearV2_fwd(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
    #pragma omp parallel  for collapse(2) schedule(dynamic)
    for (int b = 0; b < batch; b++) {
        for (int d = 0; d < out_features; d++) {
            int offset_out = b * out_features + d;
            int offset_bias = d;
            float value = 0.0f;

            for (int in = 0; in < in_features; in++) {
                int offset_in = b * in_features + in;
                int offset_weight = d * in_features + in;                 
                value += input[offset_in] * weight[offset_weight];
            }
            output[offset_out] = value;

            if (bias != NULL) {
                output[offset_out] += bias[offset_bias];
            } 
        }
    }

#ifdef LINEAR_DEBUG
    printf("linear_forward:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < out_features; i++) {
            printf("%f, ", output[b * out_features + i]);
        }
        printf("]\n");
    }
#endif
}

#ifdef AVX512_FWD
void linear_avx512_fwd(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int d = 0; d < out_features; d++) {
            int offset_out = b * out_features + d;
            int offset_bias = d;
            float value = 0.0f;
            __m512 a_avx;
            __m512 b_avx;
            __m512 d_avx;

            for (int in = 0; in < in_features; in += 16) {
                int offset_in = b * in_features + in;
                int offset_weight = d * in_features + in;  
                a_avx = _mm512_loadu_ps(input + offset_in);  
                b_avx = _mm512_loadu_ps(weight + offset_weight); 
                d_avx = _mm512_mul_ps(a_avx, b_avx);
                value += _mm512_reduce_add_ps(d_avx); 
            }
            output[offset_out] = value;

            if (bias != NULL) {
                output[offset_out] += bias[offset_bias];
            } 
        }
    }

#ifdef LINEAR_DEBUG
    printf("linear_forward:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < out_features; i++) {
            printf("%f, ", output[b * out_features + i]);
        }
        printf("]\n");
    }
#endif
}
#endif

#ifdef NEON_FWD
void linear_neon_fwd(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int d = 0; d < out_features; d++) {
                int offset_out = b * out_features + d;
                int offset_bias = d;
                float value = 0.0f;
                float32x4_t a_neon;
                float32x4_t b_neon;
                float32x4_t d_neon;
                float32x2_t sum_neon;
                for (int in = 0; in < in_features; in += 4) {
                    int offset_in = b * in_features + in;
                    int offset_weight = d * in_features + in;
                    a_neon = vld1q_f32(input + offset_in); 
                    b_neon = vld1q_f32(weight + offset_weight);    
                    d_neon = vmulq_f32(a_neon, b_neon); 
                    sum_neon = vadd_f32(vget_low_f32(d_neon), vget_high_f32(d_neon)); 
                    sum_neon = vpadd_f32(sum_neon, sum_neon); 
                    value += vget_lane_f32(sum_neon, 0);
                }
                output[offset_out] = value;

                if (bias != NULL) {
                    output[offset_out] += bias[offset_bias];
                } 
            }
    }

#ifdef LINEAR_DEBUG
    printf("linear_forward:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < out_features; i++) {
            printf("%f, ", output[b * out_features + i]);
        }
        printf("]\n");
    }
#endif
}

#endif

void linear_fwd(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
#ifdef AVX512_FWD
    linear_avx512_fwd(output, input, weight, bias, batch, in_features, out_features);
#elif NEON_FWD
    linear_neon_fwd(output, input, weight, bias, batch, in_features, out_features);
#elif OPENMP_V1
    linearV1_fwd(output, input, weight, bias, batch, in_features, out_features);
#else
    // double tdata = omp_get_wtime();
    linearV2_fwd(output, input, weight, bias, batch, in_features, out_features);
    // tdata = omp_get_wtime() - tdata;
    // if (out_features==151936) {
    //     printf("batch=%d, in_features=%d, out_features=%d,  in %f secs\n", batch, in_features, out_features, tdata);
    // }
#endif
}

} // namespace linear
} // namespace cpu
} // namespace ld_infer
