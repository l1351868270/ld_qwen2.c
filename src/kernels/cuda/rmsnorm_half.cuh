#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include "utils.cuh"

namespace ld_infer {
namespace cuda {
namespace rmsnorm_half {

// https://arxiv.org/pdf/1910.07467
__global__
void rmsnorm_fwd(half* o, half* x, half *weight, float rms_norm_eps, int dim) {
    int bidx = blockIdx.x; // batch
    int bidy = blockIdx.y;
    int tid = threadIdx.x; // thread id
    int lid = tid % 32; // lane id
    
    float ss = 0.0f;
    int offset = bidx * dim;
    #pragma unroll
    for (int i = lid; i < dim; i += WARP_THREADS) {
        ss += __half2float(x[offset + i]) * __half2float(x[offset + i]);
    }
    __syncwarp();

    #pragma unroll
    for (int mask = 32 / 2; mask > 0; mask /= 2) {
        ss += __shfl_xor_sync(uint32_t(-1), ss, mask);
        __syncwarp();
    }

    ss /= dim;
    ss += rms_norm_eps;
    ss = rsqrtf(ss);

    int offset_x = bidx * dim + bidy * blockDim.x + tid;
    int offset_w = bidy * blockDim.x + tid;
    int offset_o = bidx * dim + bidy * blockDim.x + tid;
    o[offset_o] = __float2half(__half2float(x[offset_x]) * ss * __half2float(weight[offset_w]));

    #ifdef RMSNORM_DEBUG
    if (thread0()) {
        printf("rmsnorm:\n");
        int batch = gridDim.x;
        for (int b = 0; b < batch; b++) {
            int offset = b * dim;
            printf("[");
            for (int d = 0; d < dim; d++) {
                 printf("%f, ", __half2float(o[offset + d]));
            }
            printf("],\n");
        }
    }
    #endif // CUDA_DEBUG
}
void rmsnorm_fwd_launch(half* o, half* x, half *weight, float rms_norm_eps, int batch, int dim)  {
        rmsnorm_fwd<<<dim3(batch, dim / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(o, x, weight, rms_norm_eps, dim);
}
} // namespace rmsnorm_half
} // namespace cuda
} // namespace ld_infer