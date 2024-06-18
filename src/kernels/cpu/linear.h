#pragma once


#include "utils.h"

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
    int elem_per_cpu = out_features / NUM_CPUS;
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int t;
        #pragma omp parallel for private(t)
        for (t = 0; t < NUM_CPUS; t++) {
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
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < out_features; d++) {
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
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < out_features; d++) {
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

void linear_fwd(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
#ifdef AVX512_FWD
    linear_avx512_fwd(output, input, weight, bias, batch, in_features, out_features);
#elif OPENMP_V1
    linearV1_fwd(output, input, weight, bias, batch, in_features, out_features);
#else
    linearV2_fwd(output, input, weight, bias, batch, in_features, out_features);
#endif
}

} // namespace linear
} // namespace cpu
} // namespace ld_infer
