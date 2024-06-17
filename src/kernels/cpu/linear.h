#pragma once


#include "utils.h"

namespace ld_infer {
namespace cpu {
namespace linear {

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
void linear_fwd(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
    int elem_per_cpu = out_features / NUM_CPUS;
    int b;
    #pragma omp parallel for private(b)
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

void linear_fwd1(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
    int elem_per_cpu = out_features / NUM_CPUS;
    int b;
    #pragma omp parallel for private(b)
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

#ifdef LINEAR_DEBUG1
    printf("linear_forward:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < out_features; i++) {
            printf("%d=%f, ", i, output[b * out_features + i]);
        }
        printf("]\n");
    }
#endif
}

} // namespace linear
} // namespace cpu
} // namespace ld_infer
