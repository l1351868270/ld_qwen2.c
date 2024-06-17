#pragma once

#include <concepts>
#include <cuda_fp16.h>
#include "utils.cuh"
namespace ld_infer {
namespace embedding {

template <typename T>
// concept is_half = std::is_same_v<T, half>;
// concept is_float = std::is_same_v<T, float>;
concept all = std::is_same_v<T, half> || std::is_same_v<T, float>;

template <all T>
__global__ 
void embedding_fwd(T *x, T* embed_tokens, int *token, int dim) {
    int bidx = blockIdx.x; // batch
    int bidy = blockIdx.y; // dim = 
    int tidx = threadIdx.x;
    int offset_x = bidx * dim + bidy * blockDim.x + tidx;
    int offset_t = bidy * blockDim.x + tidx;
    x[offset_x] = *(embed_tokens + token[bidx] * dim + offset_t);

    // if (thread0()) {
    //     int batch = gridDim.x;
    //     printf("[");
    //     for (int b = 0; b < batch; b++) {
    //         int offset_x = b * dim;
    //         printf("[");
    //         for (int i = 0; i<dim; i++) {
    //             printf("%f, ", __half2float(x[offset_x + i]));
    //         }
    //         printf("],\n");
    //     }
    //     printf("]\n");
    // }
}

template <all T>
void embedding_fwd_launch(T *x, T* embed_tokens, int *token, int batch, int dim) {
    embedding_fwd<T><<<dim3(batch, dim/WARPGROUP_THREADS), WARPGROUP_THREADS>>>(x, embed_tokens, token, dim);
}

} // namespace embedding
} // namespace ld_infer
