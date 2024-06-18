#pragma once

#include <cuda_fp16.h>
#include "utils.cuh"

namespace ld_infer {
namespace cuda {
namespace add_half {
__global__ 
void add_fwd(half* output, half* input) {
    int b = blockIdx.x;
    int dim = WARPGROUP_THREADS * gridDim.y;
    int bid = blockIdx.y;
    int tid = threadIdx.x;
    int offset_o = b * dim + bid * WARPGROUP_THREADS + tid;
    int offset_i = bid * WARPGROUP_THREADS + tid;
    output[offset_o] += input[offset_i];
}

void add_fwd_launch(half* output, half* input, int batch, int out_features) {
    add_fwd<<<dim3(batch, out_features / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(output, input);
}

} // namespace add_half
} // namespace cuda
} // namespace ld_infer