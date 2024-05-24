#include "../kittens/kittens.cuh"

#include <cuda/pipeline>
#include <cooperative_groups.h>

#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
#define NUM_WORKERS 4

using st_hf_1x1_a = kittens::st_hf<1, 1, kittens::ducks::st_layout::swizzle>;
using st_hf_1x1_b = kittens::st_hf<1, 1, kittens::ducks::st_layout::swizzle>;
using st_fl_1x1_c = kittens::st_fl<1, 1, kittens::ducks::st_layout::swizzle>;

using rt_hf_1x1_a = kittens::rt_hf<1, 1, kittens::ducks::rt_layout::row>;
using rt_hf_1x1_b = kittens::rt_hf<1, 1, kittens::ducks::rt_layout::row>;
using rt_fl_1x1_c = kittens::rt_fl<1, 1, kittens::ducks::rt_layout::row>;

template <typename H, typename F>
__global__ void simple_gemm_tt(size_t m, size_t n, size_t k, F alpha,
                               H const* A, size_t lda, H const* B, size_t ldb,
                               F beta, F* C, size_t ldc) {
    auto warpid = kittens::warpid();
    auto laneid = kittens::laneid();
    const size_t K_tiles = (k + MMA_K - 1) / MMA_K;
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    st_hf_1x1_a (&sA)[2][NUM_WORKERS] = al.allocate<st_hf_1x1_a, 2, NUM_WORKERS>();
    st_hf_1x1_b (&sB)[2][NUM_WORKERS] = al.allocate<st_hf_1x1_b, 2, NUM_WORKERS>();
    st_fl_1x1_c (&sC)[NUM_WORKERS] = al.allocate<st_fl_1x1_c, NUM_WORKERS>();

    rt_hf_1x1_a afrag;
    rt_hf_1x1_b bfrag;
    rt_fl_1x1_c cfrag;

    kittens::zero(cfrag);

    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> ab_barrier;
    if (threadIdx.x == 0) {init(&ab_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> store_barrier;
    if (threadIdx.x == 0) {init(&store_barrier, block.size());}
    block.sync();

    const int a_tile_elements = st_hf_1x1_a::num_elements;
    const int b_tile_elements = st_hf_1x1_b::num_elements;
    const int c_tile_elements = st_fl_1x1_c::num_elements;

    auto n_tiles  = n/kittens::TILE_DIM;
    auto n_blocks = n_tiles/NUM_WORKERS;
    assert(n_tiles % NUM_WORKERS == 0);

    if (warp_row >= m && warp_col >= n) {
        return;
    }

    #pragma unroll
    for (size_t i = 0; i < K_tiles; i++) {
        kittens::load_async(sA[tic][warpid], A + warp_row * k + i * MMA_K, k, ab_barrier);
        kittens::load_async(sB[tic][warpid], B + warp_col * k + i * MMA_K, k, ab_barrier);
        ab_barrier.arrive_and_wait();

        kittens::load(afrag, sA[tic][warpid]);
        kittens::load(bfrag, sB[tic][warpid]);

        kittens::mma_ABt(cfrag, afrag, bfrag, cfrag);
    }

    kittens::store(sC[warpid], cfrag);
    block.sync();
    kittens::store_async(C + warp_row * n + warp_col, sC[warpid], n, store_barrier);

    // if (kittens::thread0()) {
        // printf("packed_per_thread: %d\n", ar.tiles[0][0].packed_per_thread);
        // printf("height: %d\n", ar.height);
        // printf("width: %d\n", ar.width);

        // printf("max_vec outer_dim: %d\n", max_vec.outer_dim);
        // printf("max_vec inner_dim: %d\n", max_vec.inner_dim);
        // // printf("sizeof: %d\n", max_vec);
        // printf("col_vec_pack: %d\n", cr.tiles[0][0].col_vec_pack);
        // printf("hello\n");
        // for (int i = 0; i < ar.height; i++) {
        //     printf("[");
        //     for (int j = 0; j < ar.width; j++) {
        //         printf("[");
        //         for (int k = 0; k < cr.tiles[0][0].packed_per_thread; k++) {
        //             printf("%f, %f, ", cr.tiles[i][j].data[k].x, cr.tiles[i][j].data[k].y);
        //         }
        //         printf("],\n");
        //     }
        //     printf("],\n");
        // }

        // for (int i = 0; i < cfrag.height; i++) {
        //     printf("[");
        //     for (int j = 0; j < cfrag.width; j++) {
        //         printf("[");
        //         for (int k = 0; k < cfrag.tiles[0][0].packed_per_thread; k++) {
        //             printf("%f, %f, ", cfrag.tiles[i][j].data[k].x, cfrag.tiles[i][j].data[k].y);
        //         }
        //         printf("],\n");
        //     }
        //     printf("],\n");
        // }

        // for (int i = 0; i < 1; i++) {
        //     printf("[");
        //     for (int j = 0; j < sC[warpid].rows; j++) {
        //         printf("[");
        //         for (int k = 0; k < sC[warpid].cols; k++) {
        //             printf("%f, ", sC[warpid].data[j * sC[warpid].cols + k]);
        //         }
        //         printf("],\n");
        //     }
        //     printf("],\n");
        // }
    // }
}

template <typename H, typename F>
void launch_simple_gemm_tt(size_t m, size_t n, size_t k, F const* alpha,
                           H const* A, size_t lda, H const* B, size_t ldb,
                           F const* beta, F* C, size_t ldc,
                           cudaStream_t stream) {
    const dim3 block_dim{32u};
    const dim3 grid_dim{(unsigned int)(n + MMA_N - 1) / MMA_N,
                        (unsigned int)(m + MMA_M - 1) / MMA_M};
    int smem_size = 1024 * sizeof(uint32_t) * 8;
    simple_gemm_tt<H, F><<<grid_dim, block_dim, smem_size>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    cudaDeviceSynchronize();
}

#include "matmul_forward_harness.impl"