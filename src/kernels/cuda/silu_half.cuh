#pragma once

#include <cuda_fp16.h>
#include "utils.cuh"

namespace ld_infer {
namespace cuda {
namespace silu_half {

// https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
__global__
void silu_half_fwd(half *hb, half* hb2, int dim) {

    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;

    int offset = b * dim + bidy * WARPGROUP_THREADS + tid;

    float val = __half2float(hb[offset]);
    val *= 1.0f / (1.0f + __expf(-val));
    val *= __half2float(hb2[offset]);
    hb[offset] = __float2half(val);

#ifdef SILU_DEBUG
    if (thread0()) {
        int batch = gridDim.x;
        printf("silu:\n");
        for (int b = 0; b < batch; b++) {
            printf("[");
            for (int i = 0; i < dim; i++) {
                printf("%f, ", __half2float(hb[b * dim + i]));
            }
            printf("]\n");
        }
    }
# endif // SILU_DEBUG

}

void silu_fwd_launch(half *hb, half* hb2, int batch, int dim){
    silu_half_fwd<<<dim3(batch, dim / WARPGROUP_THREADS), WARPGROUP_THREADS>>>(hb, hb2, dim);
}

} // namespace silu_half
} // namespace cuda
} // namespace ld_infer