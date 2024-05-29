/*
refer to https://github.com/wangsiping97/FastGEMV/blob/1fdff6f74aade033c02727a419afd6a4b4bfbc3f/fast_gemv.cu
*/
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

namespace gemv {
#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024
///////////////////////////// REDUCE SUM //////////////////////////////

__device__ bool thread0() {
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
}

__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

__global__ void gemv_fp16(const half* mat, const half* vec, float* res, unsigned int n,
                          unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  const float4* mat4 = reinterpret_cast<const float4*>(mat);
  const float4* vec4 = reinterpret_cast<const float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const half2* mat_h1 = (half2*)&mat_val.x;
      const half2* mat_h2 = (half2*)&mat_val.y;
      const half2* mat_h3 = (half2*)&mat_val.z;
      const half2* mat_h4 = (half2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) * static_cast<float>(mat_h1->x);
      sum += static_cast<float>(vec_h1->y) * static_cast<float>(mat_h1->y);
      sum += static_cast<float>(vec_h2->x) * static_cast<float>(mat_h2->x);
      sum += static_cast<float>(vec_h2->y) * static_cast<float>(mat_h2->y);
      sum += static_cast<float>(vec_h3->x) * static_cast<float>(mat_h3->x);
      sum += static_cast<float>(vec_h3->y) * static_cast<float>(mat_h3->y);
      sum += static_cast<float>(vec_h4->x) * static_cast<float>(mat_h4->x);
      sum += static_cast<float>(vec_h4->y) * static_cast<float>(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = sum;
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = sum;
  }

}

__global__ void gemv_fp16_v1(const half* mat, const half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  const float4* mat4 = reinterpret_cast<const float4*>(mat);
  const float4* vec4 = reinterpret_cast<const float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const half2* mat_h1 = (half2*)&mat_val.x;
      const half2* mat_h2 = (half2*)&mat_val.y;
      const half2* mat_h3 = (half2*)&mat_val.z;
      const half2* mat_h4 = (half2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) * static_cast<float>(mat_h1->x);
      sum += static_cast<float>(vec_h1->y) * static_cast<float>(mat_h1->y);
      sum += static_cast<float>(vec_h2->x) * static_cast<float>(mat_h2->x);
      sum += static_cast<float>(vec_h2->y) * static_cast<float>(mat_h2->y);
      sum += static_cast<float>(vec_h3->x) * static_cast<float>(mat_h3->x);
      sum += static_cast<float>(vec_h3->y) * static_cast<float>(mat_h3->y);
      sum += static_cast<float>(vec_h4->x) * static_cast<float>(mat_h4->x);
      sum += static_cast<float>(vec_h4->y) * static_cast<float>(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }

}

template <typename H, typename F>
void launch_gemm(size_t m, size_t n, size_t k, F const* alpha,
                           H* A, size_t lda, H* B, size_t ldb,
                           F const* beta, F* C, size_t ldc,
                           cudaStream_t stream) {
  int block_dim_y = 4;
  int block_dim_x = 32;
  int num_per_thread = k / block_dim_x;
  dim3 grid_dim(1, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);

  gemv_fp16<<<grid_dim, block_dim>>>(B, A, C,
                                     k, num_per_thread);
  cudaDeviceSynchronize();
}

void gemv_fast_fwd(float* C, half* A, half* B, size_t m, size_t n, size_t k) {
  int block_dim_y = 4;
  int block_dim_x = 32;
  int num_per_thread = k / block_dim_x;
  dim3 grid_dim(m, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_fp16<<<grid_dim, block_dim>>>(B, A, C,
                                     k, num_per_thread);
  cudaDeviceSynchronize();
}

void gemv_fast_v1_fwd(half* C, half* A, half* B, size_t m, size_t n, size_t k) {
  int block_dim_y = 4;
  int block_dim_x = 32;
  int num_per_thread = k / block_dim_x;
  dim3 grid_dim(m, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_fp16_v1<<<grid_dim, block_dim>>>(B, A, C,
                                     k, num_per_thread);
  cudaDeviceSynchronize();
}

} // namespace gemv
#include "gemv_fwd_harness.impl"