
#pragma once

#include <cuda_fp16.h>
#include "utils.cuh"
#include <stdio.h>

namespace ld_infer {
namespace rope_half {
__global__ 
void rope_fwd(half *q, float rope_freq_constant, int num_heads, int head_dim, int pos) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    int offset = b * num_heads * head_dim + h * head_dim;

    for (int hd = tid; hd < head_dim / 2; hd += WARPGROUP_THREADS) {
        float v0 = __half2float(q[offset + hd]);
        float v1 = __half2float(q[offset + hd + head_dim / 2]);

        float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
        float cos_val = cosf(pos * freq);
        float sin_val = sinf(pos * freq);
        q[offset + hd] = __float2half(v0 * cos_val - v1 * sin_val);
        q[offset + head_dim / 2 + hd] = __float2half(v1 * cos_val + v0 * sin_val);
    }
    #ifdef ROPE_DEBUG
    if (thread0()) {
        printf("rope: \n");
        int batch = gridDim.x;
        for (int b = 0; b < batch; b++) {
            printf("[");
            for (int h = 0; h < num_heads; h++) {
                printf("[");    
                int offset = b * num_heads * head_dim + h * head_dim;
                for (int hd = 0; hd < head_dim; hd++) {     
                    printf("%f,", __half2float(q[offset + hd]));
                }
                printf("],\n");
            }
            printf("],\n");
        }
    }
    #endif // ROPE_DEBUG
}

void rope_launch(half *q, float rope_theta, int batch, int num_heads, int head_dim, int pos) {
    rope_fwd<<<dim3(batch, num_heads), WARPGROUP_THREADS>>>(q, rope_theta, num_heads, head_dim, pos);
}

} // namespace rope_half
} // namespace ld_infer