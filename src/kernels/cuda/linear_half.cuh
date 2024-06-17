#pragma once

#include <cuda_fp16.h>
#include <cublas_v2.h>
// #include <cublasLt.h>
#include "utils.cuh"
#include "./src/kernels/cuda/add_half.cuh"

namespace ld_infer {
namespace linear_half {

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
void linear_fwd_launch(cublasHandle_t* handle, half* output, half* input, half *weight, half* bias, int batch, int in_features, int out_features) {
    int M = batch;
    int N = out_features;
    int K = in_features;

    // gemv::gemv_fast_v1_fwd(output, input, weight, M, N, K);
    
    float alpha = 1.f;
    float beta = 0.0f;

    // cublasStatus_t status = cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
    //                                       M, N, K,
    //                                       &alpha, 
    //                                       input, CUDA_R_16F, K, 
    //                                       weight, CUDA_R_16F, K, 
    //                                       &beta, 
    //                                       output, CUDA_R_16F, M);

    cublasStatus_t status = cublasSgemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                          N, M, K,
                                          &alpha, 
                                          weight, CUDA_R_16F, K, 
                                          input, CUDA_R_16F, K, 
                                          &beta, 
                                          output, CUDA_R_16F, N);

    if (bias != NULL) {
        ld_infer::add_half::add_fwd_launch(output, bias, batch, out_features);
    }

#ifdef LINEAR_DEBUG
   half *H_input = (half*)malloc(M * K * sizeof(half));
    cudaMemcpy(H_input, input, M * K * sizeof(half), cudaMemcpyDeviceToHost);
    printf("linear_forward input:\n");
    for (int b = 0; b < M; b++) {
        printf("[");
        for (int i = 0; i < K; i++) {
            printf("%f, ", __half2float(H_input[b * K + i]));
        }
        printf("]\n");
    }

    half *H_C = (half*)malloc(M * N * sizeof(half));
    cudaMemcpy(H_C, output, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    printf("linear_forward H_C:\n");
    for (int b = 0; b < M; b++) {
        printf("[");
        for (int i = 0; i < N; i++) {
            printf("%f, ", __half2float(H_C[b * N + i]));
        }
        printf("]\n");
    }
#endif // LINEAR_DEBUG
}
} // namespace linear_half
} // namespace ld_infer
