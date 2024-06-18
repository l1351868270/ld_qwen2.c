#pragma once

#include <cuda_fp16.h>
#include "utils.cuh"

namespace ld_infer {
namespace cuda {
namespace residual_half {

__global__
void residual_half_fwd(half *x, half *xb, int dim) {
    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;
    int kNThreads = blockDim.x;
    int offset = b * dim + bidy * kNThreads + tid;

    x[offset] += xb[offset];

#ifdef RESIDUAL_DEBUG
    if (thread0()) {
        int batch = gridDim.x;
        printf("residual:\n");
        for (int b = 0; b < batch; b++) {
            printf("[");
            for (int i = 0; i < dim; i++) {
                int offset_x = b * dim + i;
                printf("%f, ", __half2float(x[offset_x]));
            }
            printf("]\n");
        }
    }
#endif // RESIDUAL_DEBUG

}

void residual_fwd_launch(half *x, half *xb, int batch, int dim)
{
    residual_half_fwd<<<dim3(batch, dim / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(x, xb, dim);
}

} // namespace residual_half
} // namespace cuda
} // namespace ld_infer