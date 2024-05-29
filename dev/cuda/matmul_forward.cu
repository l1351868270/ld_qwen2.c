
/*
refer to https://github.com/HazyResearch/ThunderKittens/tree/main/examples/gemm/A100
*/
#include <stdio.h>
#include "mma.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda/pipeline>
#include <cooperative_groups.h>

#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
#define NUM_WORKERS 4

template <typename H, typename F>
__global__ void simple_gemm_tt(size_t m, size_t n, size_t k, F alpha,
                               H const* A, size_t lda, H const* B, size_t ldb,
                               F beta, F* C, size_t ldc) {

    int warpid = threadIdx.x / ld_qwen2::WARP_THREADS;
    int laneid = threadIdx.x % ld_qwen2::WARP_THREADS;

    const size_t K_tiles = (k + MMA_K - 1) / MMA_K;
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    using shared_tile_ab = ld_qwen2::ld_shared::shared_tile_hf_1x1<ld_qwen2::ld_shared_layout::naive>;
    using shared_tile_c = ld_qwen2::ld_shared::shared_tile_fl_1x1<ld_qwen2::ld_shared_layout::naive>;

    using registers_ab = ld_qwen2::ld_registers_tile::registers_hf_1x1<ld_qwen2::ld_registers_tile_layout::row>;
    using registers_c = ld_qwen2::ld_registers_tile::registers_fl_1x1<ld_qwen2::ld_registers_tile_layout::row>;

    extern __shared__ int __shm[];

    ld_qwen2::ld_shared::shared_allocator<16> al((int*)&__shm[0]);

    // shared_tile_ab *sa = (shared_tile_ab*)__shm;
    shared_tile_ab *sa = al.allocate<shared_tile_ab, 2, NUM_WORKERS>();
    shared_tile_ab *sb = al.allocate<shared_tile_ab, 2, NUM_WORKERS>();
    shared_tile_c *sc = al.allocate<shared_tile_c, 2, NUM_WORKERS>();

    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> ab_barrier;
    if (threadIdx.x == 0) {init(&ab_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> store_barrier;
    if (threadIdx.x == 0) {init(&store_barrier, block.size());}
    block.sync(); 

    const int ab_tile_elements = shared_tile_ab::num_elements;
    const int  c_tile_elements = shared_tile_c::num_elements;
    auto n_tiles  = n/ld_qwen2::TILE_DIM;
    auto n_blocks = n_tiles/NUM_WORKERS;
    assert(n_tiles % NUM_WORKERS == 0);

    // ld_qwen2::g2s_load_async(sa + warpid, A + warpid * ld_qwen2::TILE_DIM, k, ab_barrier);
    // ld_qwen2::g2s_load_async(sb + warpid, B + warpid * ld_qwen2::TILE_DIM, k, ab_barrier);

    ld_qwen2::g2s_load_async(sa + warpid, A, k, ab_barrier);
    ld_qwen2::g2s_load_async(sb + warpid, B, k, ab_barrier);
    ab_barrier.arrive_and_wait(); 
    registers_ab afrag, bfrag; 
    registers_c cfrag;
    cfrag.tiles[0][0].data = {0, 0, 0, 0};
    ld_qwen2::s2r_load(&afrag, sa);
    ld_qwen2::s2r_load(&bfrag, sb);
    
    ab_barrier.arrive_and_wait(); 
    ld_qwen2::fma16816(cfrag.tiles[0][0].data.x, cfrag.tiles[0][0].data.y, cfrag.tiles[0][0].data.z, cfrag.tiles[0][0].data.w,
             afrag.tiles[0][0].data.x, afrag.tiles[0][0].data.y, afrag.tiles[0][0].data.z, afrag.tiles[0][0].data.w,
             bfrag.tiles[0][0].data.x, bfrag.tiles[0][0].data.y,
             cfrag.tiles[0][0].data.x, cfrag.tiles[0][0].data.y, cfrag.tiles[0][0].data.z, cfrag.tiles[0][0].data.w);
    ab_barrier.arrive_and_wait(); 
    if (ld_qwen2::thread0()) {
        
        // // printf("sizeof: %d\n", sizeof(sa[2].data));
        // // printf("sizeof: %d\n", sizeof(o.data));
        // printf("sizeof: %d\n", sizeof(afrag));
        // // printf("sizeof: %d\n", sizeof(cfrag));
        // printf("sizeof: %d\n", sizeof(cfrag));
        // // __bfloat162float
        half2 packed_half2 = make_half2(1.0, 2.0);
        uint32_t packed_uint32 = *(reinterpret_cast<uint32_t*>(&packed_half2));

        half2 q = *(reinterpret_cast<half2*>(&(cfrag.tiles[0][0].data.x)));
        // half2 q = *(reinterpret_cast<half2*>(&packed_uint32));
        half floatVal1 = __low2half(q);
        half floatVal2 = __high2half(q);
        printf("%f, ", __half2float(floatVal1));
        printf("%f, ", __half2float(floatVal2));
        q = *(reinterpret_cast<half2*>(&(cfrag.tiles[0][0].data.y)));
        // half2 q = *(reinterpret_cast<half2*>(&packed_uint32));
        floatVal1 = __low2half(q);
        floatVal2 = __high2half(q);
        printf("%f, ", __half2float(floatVal1));
        printf("%f, ", __half2float(floatVal2));

        q = *(reinterpret_cast<half2*>(&(cfrag.tiles[0][0].data.z)));
        // half2 q = *(reinterpret_cast<half2*>(&packed_uint32));
        floatVal1 = __low2half(q);
        floatVal2 = __high2half(q);
        printf("%f, ", __half2float(floatVal1));
        printf("%f, ", __half2float(floatVal2));

        q = *(reinterpret_cast<half2*>(&(cfrag.tiles[0][0].data.w)));
        // half2 q = *(reinterpret_cast<half2*>(&packed_uint32));
        floatVal1 = __low2half(q);
        floatVal2 = __high2half(q);
        printf("%f, ", __half2float(floatVal1));
        printf("%f, ", __half2float(floatVal2));

        // uint16_t highBits = static_cast<uint16_t>(afrag.tiles[0][0].data.x >> 16);
        // uint16_t lowBits = static_cast<uint16_t>(afrag.tiles[0][0].data.x & 0xFFFF);
        // printf("%d, ", (float)(highBits));
        // printf("%f, ", (float)lowBits);
        // highBits = static_cast<uint16_t>(afrag.tiles[0][0].data.y >> 16);
        // lowBits = static_cast<uint16_t>(afrag.tiles[0][0].data.y & 0xFFFF);
        // printf("%f, ", (float)highBits);
        // printf("%f, ", (float)lowBits);
        // highBits = static_cast<uint16_t>(afrag.tiles[0][0].data.z >> 16);
        // lowBits = static_cast<uint16_t>(afrag.tiles[0][0].data.z & 0xFFFF);
        // printf("%f, ", (float)highBits);
        // printf("%f, ", (float)lowBits);
        // highBits = static_cast<uint16_t>(afrag.tiles[0][0].data.w >> 16);
        // lowBits = static_cast<uint16_t>(afrag.tiles[0][0].data.w & 0xFFFF);
        // printf("%f, ", (float)highBits);
        // printf("%f, ", (float)lowBits);


        printf("\n");
        for (int i = 0; i < ld_qwen2::TILE_DIM; i++) {
            printf("[");
            for (int j = 0; j < ld_qwen2::TILE_DIM; j++) {
                printf("%f, ", __half2float((*(sa))[i * ld_qwen2::TILE_DIM + j]));
                // printf("%f, ", __half2float((*sa)[{i, j}]));
            }
            printf("]\n");
        }
        printf("++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        // for (int i = 0; i < ld_qwen2::TILE_DIM; i++) {
        //     printf("[");
        //     for (int j = 0; j < ld_qwen2::TILE_DIM; j++) {
        //         printf("%f, ", __half2float(A[i * k + 3 * ld_qwen2::TILE_DIM + j]));
        //         // printf("%f, ", __half2float((*sa)[{i, j}]));
        //     }
        //     printf("]\n");
        // }

        for (int i = 0; i < ld_qwen2::TILE_DIM; i++) {
            printf("[");
            for (int j = 0; j < ld_qwen2::TILE_DIM; j++) {
                printf("%f, ", __half2float((*(sb))[i * ld_qwen2::TILE_DIM + j]));
                // printf("%f, ", __half2float((*sa)[{i, j}]));
            }
            printf("]\n");
        }


        printf("hello\n");
    }
}

template <typename H, typename F>
void launch_simple_gemm_tt(size_t m, size_t n, size_t k, F const* alpha,
                           H const* A, size_t lda, H const* B, size_t ldb,
                           F const* beta, F* C, size_t ldc,
                           cudaStream_t stream) {
    const dim3 block_dim{128u};
    const dim3 grid_dim{(unsigned int)(n + MMA_N - 1) / MMA_N,
                        (unsigned int)(m + MMA_M - 1) / MMA_M};
    int smem_size = 1024 * sizeof(uint32_t) * 8;
    simple_gemm_tt<H, F><<<1, 32, smem_size>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    cudaDeviceSynchronize();
}

#include "matmul_forward_harness.impl"