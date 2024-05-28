/*

*/
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

namespace gemv {
constexpr int WARP_THREADS{32};
constexpr int WARPGROUP_THREADS{128};
constexpr int WARPGROUP_WARPS{4};
#define WARP_SIZE 32

template <typename H, typename F>
__global__ void gemm(size_t m, size_t n, size_t k, F alpha,
                               H const* A, size_t lda, H const* B, size_t ldb,
                               F beta, F* C, size_t ldc) {
    auto tid = threadIdx.x;
    const size_t MMA_K{k / WARP_THREADS};
    const size_t kid = tid % WARP_THREADS;
    const size_t row = kid * MMA_K;
    // const size_t col = blockIdx.y * WARPGROUP_WARPS + tid / WARP_THREADS;
    const size_t col = blockIdx.y * WARPGROUP_WARPS + threadIdx.y;

    const float4* A4 = reinterpret_cast<const float4*>(A + row);
    const float4* B4 = reinterpret_cast<const float4*>(B + col * k + row);

    float ss = 0.0;
    for (int i = 0; i < MMA_K >> 3; i++) {
        float4 a = A4[i];
        float4 b = B4[i];
        const half2* a_0 = reinterpret_cast<half2*>(&a.x);
        const half2* a_1 = reinterpret_cast<half2*>(&a.y);
        const half2* a_2 = reinterpret_cast<half2*>(&a.z);
        const half2* a_3 = reinterpret_cast<half2*>(&a.w);
        const half2* b_0 = reinterpret_cast<half2*>(&b.x);
        const half2* b_1 = reinterpret_cast<half2*>(&b.y);
        const half2* b_2 = reinterpret_cast<half2*>(&b.z);
        const half2* b_3 = reinterpret_cast<half2*>(&b.w);
        ss += static_cast<float>(a_0->x) * static_cast<float>(b_0->x);
        ss += static_cast<float>(a_0->y) * static_cast<float>(b_0->y);
        ss += static_cast<float>(a_1->x) * static_cast<float>(b_1->x);
        ss += static_cast<float>(a_1->y) * static_cast<float>(b_1->y);
        ss += static_cast<float>(a_2->x) * static_cast<float>(b_2->x);
        ss += static_cast<float>(a_2->y) * static_cast<float>(b_2->y);
        ss += static_cast<float>(a_3->x) * static_cast<float>(b_3->x);
        ss += static_cast<float>(a_3->y) * static_cast<float>(b_3->y);
    }

    ss += __shfl_down_sync(0xffffffff, ss, 16);
    ss += __shfl_down_sync(0xffffffff, ss, 8);
    ss += __shfl_down_sync(0xffffffff, ss, 4);
    ss += __shfl_down_sync(0xffffffff, ss, 2);
    ss += __shfl_down_sync(0xffffffff, ss, 1);
    __syncthreads();
    if (kid == 0) {
        C[col] = ss;
    }
}

template <typename H, typename F>
void launch_gemm(size_t m, size_t n, size_t k, F const* alpha,
                           H const* A, size_t lda, H const* B, size_t ldb,
                           F const* beta, F* C, size_t ldc,
                           cudaStream_t stream) {
    // const dim3 block_dim{WARPGROUP_THREADS};
    // unsigned int grid =(n + WARPGROUP_WARPS - 1) / WARPGROUP_WARPS;

    // int smem_size = k * sizeof(half);
    // cudaFuncSetAttribute(gemm<H, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    // gemm<H, F><<<dim3(1, grid), dim3(), smem_size>>>(
    //     m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    // gemm<H, F><<<dim3(1, grid), block_dim>>>(
    //     m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    const dim3 block_dim{WARP_THREADS, WARPGROUP_WARPS};
    unsigned int grid = (n + WARPGROUP_WARPS - 1) / WARPGROUP_WARPS;
    gemm<H, F><<<dim3(1, grid), block_dim>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    cudaDeviceSynchronize();
}

} //namespace gemv
#include "gemv_fwd_harness.impl"