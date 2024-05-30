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

struct uint4_2 {
  uint8_t data;

  uint4_2(uint8_t x = 0, uint8_t y = 0) {
    setX(x);
    setY(y);
  }

  __host__ __device__ uint8_t getX() const {
    return data & 0x0F;  // get the lower 4 bits
  }

  __host__ __device__ uint8_t getY() const {
    return (data >> 4) & 0x0F;  // get the upper 4 bits
  }

  __host__ __device__ void setX(uint8_t x) {
    data = (data & 0xF0) | (x & 0x0F);  // set the lower 4 bits
  }

  __host__ __device__ void setY(uint8_t y) {
    data = (data & 0x0F) | ((y & 0x0F) << 4);  // set the upper 4 bits
  }
};

struct half4 {
  half x, y, z, w;
};

struct int8_2 {
  int8_t x, y;
};

struct uint4_2_4 {
  uint4_2 x, y, z, w;
};

__global__ void gemv_quantized_int8(int8_t* mat, half* vec, half* res,
                                    unsigned int n, half* scale, half zero_point,
                                    unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  half4* mat4 = reinterpret_cast<half4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale[row]);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      half4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const int8_2* mat_h1 = (int8_2*)&mat_val.x;
      const int8_2* mat_h2 = (int8_2*)&mat_val.y;
      const int8_2* mat_h3 = (int8_2*)&mat_val.z;
      const int8_2* mat_h4 = (int8_2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) *
             (static_cast<float>(mat_h1->x) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) *
             (static_cast<float>(mat_h1->y) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) *
             (static_cast<float>(mat_h2->x) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) *
             (static_cast<float>(mat_h2->y) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) *
             (static_cast<float>(mat_h3->x) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) *
             (static_cast<float>(mat_h3->y) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) *
             (static_cast<float>(mat_h4->x) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) *
             (static_cast<float>(mat_h4->y) - zero_point_f);
    }
  }

  sum *= scale_f;

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


__global__ void gemv_quantized_int4(uint4_2* mat, half* vec, half* res,
                                    unsigned int n, half* scale, half zero_point,
                                    unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  uint4_2_4* mat4 = reinterpret_cast<uint4_2_4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale[row]);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 4; iter++) {
    unsigned int j = 2 * (start_idx + iter * blockDim.x);
    if (j < n >> 3) {
      float4 vec_val_1 = vec4[j];  // 8 half
      float4 vec_val_2 = vec4[j + 1];
      const half2* vec_h1 = (half2*)&vec_val_1.x;
      const half2* vec_h2 = (half2*)&vec_val_1.y;
      const half2* vec_h3 = (half2*)&vec_val_1.z;
      const half2* vec_h4 = (half2*)&vec_val_1.w;
      const half2* vec_h5 = (half2*)&vec_val_2.x;
      const half2* vec_h6 = (half2*)&vec_val_2.y;
      const half2* vec_h7 = (half2*)&vec_val_2.z;
      const half2* vec_h8 = (half2*)&vec_val_2.w;

      uint4_2_4 mat_val_1 = mat4[row * (n >> 3) + j];
      uint4_2_4 mat_val_2 = mat4[row * (n >> 3) + j + 1];
      const uint4_2* mat_h1 = (uint4_2*)&mat_val_1.x;
      const uint4_2* mat_h2 = (uint4_2*)&mat_val_1.y;
      const uint4_2* mat_h3 = (uint4_2*)&mat_val_1.z;
      const uint4_2* mat_h4 = (uint4_2*)&mat_val_1.w;
      const uint4_2* mat_h5 = (uint4_2*)&mat_val_2.x;
      const uint4_2* mat_h6 = (uint4_2*)&mat_val_2.y;
      const uint4_2* mat_h7 = (uint4_2*)&mat_val_2.z;
      const uint4_2* mat_h8 = (uint4_2*)&mat_val_2.w;

      sum += static_cast<float>(vec_h1->x) *
             (static_cast<float>(mat_h1->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) *
             (static_cast<float>(mat_h1->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) *
             (static_cast<float>(mat_h2->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) *
             (static_cast<float>(mat_h2->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) *
             (static_cast<float>(mat_h3->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) *
             (static_cast<float>(mat_h3->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) *
             (static_cast<float>(mat_h4->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) *
             (static_cast<float>(mat_h4->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h5->x) *
             (static_cast<float>(mat_h5->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h5->y) *
             (static_cast<float>(mat_h5->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h6->x) *
             (static_cast<float>(mat_h6->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h6->y) *
             (static_cast<float>(mat_h6->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h7->x) *
             (static_cast<float>(mat_h7->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h7->y) *
             (static_cast<float>(mat_h7->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h8->x) *
             (static_cast<float>(mat_h8->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h8->y) *
             (static_cast<float>(mat_h8->getY()) - zero_point_f);
    }
  }

  sum *= scale_f;

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

void gemv_fast_q80_fwd(half* C, half* A, int8_t* B, half* scale, size_t m, size_t n, size_t k) {
  int block_dim_y = 4;
  int block_dim_x = 32;
  int num_per_thread = k / block_dim_x;
  dim3 grid_dim(m, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_quantized_int8<<<grid_dim, block_dim>>>(B, A, C,
                                     k, scale, 0.0f, num_per_thread);
  cudaDeviceSynchronize();
}

void gemv_fast_q40_fwd(half* C, half* A, uint8_t* B, half* scale, size_t m, size_t n, size_t k) {
  int block_dim_y = 4;
  int block_dim_x = 32;
  int num_per_thread = k / block_dim_x;
  dim3 grid_dim(m, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  uint4_2 *B_uint4_2 = (uint4_2*)(B);
  gemv_quantized_int4<<<grid_dim, block_dim>>>(B_uint4_2, A, C,
                                     k, scale, 7.0f, num_per_thread);
  cudaDeviceSynchronize();
}

} // namespace gemv
#include "gemv_fwd_harness.impl"