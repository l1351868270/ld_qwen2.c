#include "cublas_v2.h"
#include <stdio.h>
template <typename H, typename F>
void launch_simple_gemm_tt(size_t m, size_t n, size_t k, F const* alpha,
                           H const* A, size_t lda, H const* B, size_t ldb,
                           F const* beta, F* C, size_t ldc,
                           cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status = cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                          m, n, k,
                                          alpha, 
                                          A, CUDA_R_16F, k, 
                                          B, CUDA_R_16F, k, 
                                          beta, 
                                          C, CUDA_R_32F, m);
    cudaDeviceSynchronize();
    cublasDestroy(handle);
}

#include "matmul_forward_harness.impl"