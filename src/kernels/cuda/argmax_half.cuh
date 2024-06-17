#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include "utils.cuh"

namespace ld_infer {
namespace argmax_half {

__global__
void argmax_half_fwd(int* output, half* input, int dim) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lid = tid % 32; // lane id

    int offset = b * dim;

    int max_i = lid;
    half max_val = input[offset + max_i];
    
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

#ifdef ARGMAX_DEBUG
    if (thread0()) {
        int batch = gridDim.x;
        printf("argmax:\n");
        printf("[");
        for (int b = 0; b < batch; b++) {
            printf("%d, ", output[b]);
        }
        printf("]\n");
    }
#endif // ARGMAX_DEBUG
}

void argmax_fwd_launch(int* output, half* input, int batch, int dim) {
    argmax_half_fwd<<<batch, WARPGROUP_THREADS>>>(output, input, dim);
}

} // namespace argmax_half
} // namespace ld_infer