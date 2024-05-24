#include "../kittens/kittens.cuh"

#include <cuda/pipeline>
#include <cooperative_groups.h>

#define NUM_WORKERS (4)
#define MMA_M (kittens::TILE_DIM)
#define MMA_N ((kittens::TILE_DIM) * (NUM_WORKERS))
#define NUM_K 4
#define MMA_K ((kittens::TILE_DIM) * (NUM_K))
#define default_alignment (1024)

using st_hf_1x4_a = kittens::st_hf<1, NUM_K, kittens::ducks::st_layout::swizzle>;
using st_hf_1x4_b = kittens::st_hf<1, NUM_K, kittens::ducks::st_layout::swizzle>;
using st_fl_1x1_c = kittens::st_fl<1, 1, kittens::ducks::st_layout::swizzle>;

using rt_hf_1x4_a = kittens::rt_hf<1, NUM_K, kittens::ducks::rt_layout::row>;
using rt_hf_1x4_b = kittens::rt_hf<1, NUM_K, kittens::ducks::rt_layout::row>;
using rt_fl_1x1_c = kittens::rt_fl<1, 1, kittens::ducks::rt_layout::row>;

template <typename H, typename F>
__global__ void simple_gemm_tt(size_t m, size_t n, size_t k, F alpha,
                               H const* A, size_t lda, H const* B, size_t ldb,
                               F beta, F* C, size_t ldc) {
    auto warpid = kittens::warpid();
    auto laneid = kittens::laneid();
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N + warpid * kittens::TILE_DIM;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator<default_alignment> al((int*)&__shm[0]);
    st_hf_1x4_a (&sA)[2][1] = al.allocate<st_hf_1x4_a, 2, 1>();
    st_hf_1x4_b (&sB)[2][NUM_WORKERS] = al.allocate<st_hf_1x4_b, 2, NUM_WORKERS>();
    st_fl_1x1_c (&sC)[NUM_WORKERS] = al.allocate<st_fl_1x1_c, NUM_WORKERS>();

    rt_hf_1x4_a afrag;
    rt_hf_1x4_b bfrag;
    rt_fl_1x1_c cfrag;
    kittens::zero(cfrag);

    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> ab_barrier;
    if (threadIdx.x == 0) {init(&ab_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> store_barrier;
    if (threadIdx.x == 0) {init(&store_barrier, block.size());}
    block.sync();
    if (warpid == 0) {
        kittens::load_async(sA[tic][0], A + warp_row * k + 0 * MMA_K, k, ab_barrier);
    }
    kittens::load_async(sB[tic][warpid], B + warp_col * k + 0 * MMA_K, k, ab_barrier);

    const int a_tile_elements = st_hf_1x4_a::num_elements;
    const int b_tile_elements = st_hf_1x4_b::num_elements;
    const int c_tile_elements = st_fl_1x1_c::num_elements;


    auto n_tiles  = n / MMA_K;

    if (warp_row >= m && warp_col >= n) {
        return;
    }
    
    #pragma unroll
    for (size_t i = 0; i < n_tiles;i++, tic ^= 1, toc ^=1) {
        ab_barrier.arrive_and_wait();
        if (i < n_tiles - 1) {
            if (warpid == 0) {
                kittens::load_async(sA[toc][0], A + warp_row * k + (i + 1) * MMA_K, k, ab_barrier);
            }
            kittens::load_async(sB[toc][warpid], B + warp_col * k + (i + 1) * MMA_K, k, ab_barrier);
        }

        kittens::load(afrag, sA[tic][0]);
        kittens::load(bfrag, sB[tic][warpid]);
        kittens::mma_ABt(cfrag, afrag, bfrag, cfrag);
    }

    // kittens::store(sC[warpid], cfrag);
    // block.sync();
    // kittens::store_async(C + warp_row * n + warp_col, sC[warpid], n, store_barrier);
    kittens::store(C + warp_row * n + warp_col, cfrag, n);

    // if (kittens::thread0()) {
    // }
}

template <typename H, typename F>
void launch_simple_gemm_tt(size_t m, size_t n, size_t k, F const* alpha,
                           H const* A, size_t lda, H const* B, size_t ldb,
                           F const* beta, F* C, size_t ldc,
                           cudaStream_t stream) {
    const dim3 block_dim{128u};
    unsigned int grid_0 = (unsigned int)(n + MMA_N - 1) / MMA_N;
    unsigned int grid_1 = (unsigned int)(m + MMA_M - 1) / MMA_M;
    const dim3 grid_dim{grid_0, grid_1};                       
    int smem_size = 2 * sizeof(st_hf_1x4_a)
                  + 2 * NUM_WORKERS *sizeof(st_hf_1x4_b)
                  + NUM_WORKERS * sizeof(st_fl_1x1_c)
                  + 2 * default_alignment;
    cudaFuncSetAttribute(simple_gemm_tt<H, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    simple_gemm_tt<H, F><<<grid_dim, block_dim, smem_size>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    cudaDeviceSynchronize();
}

#include "matmul_forward_harness.impl"