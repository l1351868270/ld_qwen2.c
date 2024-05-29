/*
refer to https://github.com/HazyResearch/ThunderKittens
*/
#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include <concepts>
#include <cuda/pipeline>

namespace ld_qwen2 {

constexpr int TILE_DIM{16};
constexpr int WARP_THREADS{32};
constexpr int WARPGROUP_THREADS{128};
constexpr int WARPGROUP_WARPS{4};

__device__ bool thread0() {
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
}

namespace ld_registers_tile_layout {
    struct row { static constexpr bool is_row = true;  };
    struct col { static constexpr bool is_row = false; };

    template<typename T>
    concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

    template<typename T>
    concept row_layout = all<T> && std::is_same_v<typename T::layout, row>;

    template<typename T>
    concept col_layout = all<T> && std::is_same_v<typename T::layout, col>;
} //namespace ld_registers_tile_layout

namespace ld_registers_tile_base {

    struct identifier {};

    template<typename T> 
    struct packing {
        static __device__ inline constexpr int num() { return 1; }
    };

    template<> struct packing<float> {
        static __device__ inline constexpr int num() { return 1;}
    };

    template<> struct packing<half> {
        static __device__ inline constexpr int num() { return 2;}
    };

    template<typename T, ld_registers_tile_layout::all _layout>
    struct registers_base {
        using identifier = ld_registers_tile_base::identifier;
        using layout = _layout;
        using dtype = T;
        static_assert(
            std::is_same_v<dtype, half> || std::is_same_v<dtype, float>,
            "registers_base only support half and float"
        );

        static constexpr int tile_size = 16;
        static constexpr int rows = tile_size;
        static constexpr int cols = tile_size;
        static constexpr int num_elements = rows * cols;
        static constexpr int elements_per_threads = num_elements / 32;
        static constexpr int packed_per_thread = elements_per_threads / packing<T>::num();

        uint4 data;
    };

    template<typename T> concept all = requires {
        typename T::identifier; 
    } && std::is_same_v<typename T::identifier, identifier>;

} // namespace ld_registers

namespace ld_registers_tile {
    struct identifier {};

    template<typename T, int _height, int _width, ld_registers_tile_layout::all _layout=ld_registers_tile_layout::row>
    struct registers {
        using identifier = ld_registers_tile::identifier; 
        using layout = _layout;
        using dtype = T;

        static constexpr int height = _height;
        static constexpr int width = _width;
        static constexpr int tile_size = ld_registers_tile_base::registers_base<T, layout>::tile_size;
        static constexpr int rows = height * tile_size;
        static constexpr int cols = width * tile_size;
        static constexpr int num_elements = ld_registers_tile_base::registers_base<T, layout>::num_elements * width * height;
        static constexpr int elements_per_threads = ld_registers_tile_base::registers_base<T, layout>::elements_per_threads * width * height;
        static constexpr int packed_per_thread = ld_registers_tile_base::registers_base<T, layout>::packed_per_thread * width * height;

        ld_registers_tile_base::registers_base<dtype, layout> tiles[height][width];
    };

    template<int _height, int _width, ld_registers_tile_layout::all layout=ld_registers_tile_layout::row> using registers_fl = registers<float, _height, _width, layout>;
    template<int _height, int _width, ld_registers_tile_layout::all layout=ld_registers_tile_layout::row> using registers_hf = registers<half, _height, _width, layout>;
 
    template<ld_registers_tile_layout::all layout=ld_registers_tile_layout::row> using registers_fl_1x1 = registers_fl<1, 1, layout>;

    template<ld_registers_tile_layout::all layout=ld_registers_tile_layout::row> using registers_hf_1x1 = registers_hf<1, 1, layout>;

    template<typename T> concept all = requires {
        typename T::identifier; 
    } && std::is_same_v<typename T::identifier, identifier>;

} // namespace ld_registers

//////////////////////////////////////////////////////////////////////////////////////////////////////

namespace ld_shared_layout {

struct naive {};

struct wgmma_naive {};

template<typename T>
concept all = (
    std::is_same_v<T, naive>            ||
    std::is_same_v<T, wgmma_naive>
);
} // namespace ld_shared_layout

namespace ld_shared {

struct identifier {};

template<typename _T, int _height, int _width, ld_shared_layout::all _layout>
struct __align__(16) shared_tile {
    using identifier = identifier;
    using layout = _layout;
    using dtype = _T;
    static constexpr int height = _height;
    static constexpr int width = _width;
    static constexpr int rows = height * ld_qwen2::TILE_DIM;
    static constexpr int cols = width * ld_qwen2::TILE_DIM;
    static constexpr int num_elements = rows * cols;
    dtype data[rows*cols];

    __device__ inline dtype& operator[](const int2 &rowcol) {
        int r = rowcol.x;
        int c = rowcol.y;
        return data[r*cols + c];
    }

    __device__ inline const dtype& operator[](const int2 &rowcol) const{
        int r = rowcol.x;
        int c = rowcol.y;
        return data[r*cols + c];
    }

    __device__ inline dtype& operator[](const int idx) {
        return data[idx];
    }

    __device__ inline const dtype& operator[](const int idx) const {
        return data[idx];
    }
};

template<typename _T, int _subtile_height, int _subtile_width, ld_shared_layout::all _layout>
struct shared_subtile {
    using layout = _layout;
    using dtype = _T;

    static constexpr int height = _subtile_height;
    static constexpr int width = _subtile_width;
    static constexpr int rows = height * ld_qwen2::TILE_DIM;
    static constexpr int cols = width * ld_qwen2::TILE_DIM;
    static constexpr int num_elements = rows * cols;

    dtype *data;
    int row_offset, col_offset;

    __device__ shared_subtile(dtype *src, int _row_offset, int _col_offset) {
        data = src;
        row_offset = _row_offset;
        col_offset = _col_offset;
    }

    __device__ inline dtype& operator[](const int2 &rowcol) {
        return data[(rowcol.x + row_offset) * cols + rowcol.y+col_offset];
    }
};

template<int _height, int _width, ld_shared_layout::all layout=ld_shared_layout::naive> using shared_tile_hf = shared_tile<half, _height, _width, layout>;
template<int _height, int _width, ld_shared_layout::all layout=ld_shared_layout::naive> using shared_tile_fl = shared_tile<float, _height, _width, layout>;


template<ld_shared_layout::all layout=ld_shared_layout::naive> using shared_tile_hf_1x1 = shared_tile_hf<1, 1, layout>;
template<ld_shared_layout::all layout=ld_shared_layout::naive> using shared_tile_fl_1x1 = shared_tile_fl<1, 1, layout>;

template<int default_alignment=-1>
struct shared_allocator {
    int *ptr;
    
    public:
        __device__ shared_allocator(int *_ptr) {
            ptr = _ptr;
        }

        template<typename A, size_t dim> 
        __device__ inline A* allocate() {
            align_ptr<default_alignment>();
            A *p = reinterpret_cast<A*>(ptr);
            ptr += (dim * sizeof(A))/sizeof(int);
            return p;
        }

        template<typename A, size_t dim0, size_t dim1> 
        __device__ inline A* allocate() {
            align_ptr<default_alignment>();
            A *p = reinterpret_cast<A*>(ptr);
            ptr += (dim0 * dim1 * sizeof(A))/sizeof(int);
            return p;
        }


        template<int alignment, typename A, size_t dim> 
        __device__ inline A* allocate() {
            align_ptr<alignment>();
            A *p = reinterpret_cast<A*>(ptr);
            ptr += (dim * sizeof(A))/sizeof(int);
            return p;
        }

        template<int alignment, typename A, size_t dim0, size_t dim1> 
        __device__ inline A* allocate() {
            align_ptr<alignment>();
            A *p = reinterpret_cast<A*>(ptr);
            ptr += (dim0 * dim1 * sizeof(A))/sizeof(int);
            return p;
        }


    private:
        template<int alignment>
        __device__ inline void align_ptr() {
            if constexpr (alignment > 0) {
                uint64_t p = reinterpret_cast<uint64_t>(ptr);
                ptr = (int*)(p + (alignment-(p%alignment)));
            }
        }
};

// template<typename T> concept all = std::is_same_v<T, shared_tile>;


template<typename T> concept all = std::is_same_v<typename T::identifier, identifier>;

} // namespace ld_shared



template<ld_shared::all ST>
__device__ static inline void g2s_load_async(ST *dst, const half *src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    
    int laneid = threadIdx.x % 32;

    int elem_per_memcpy = sizeof(float4)/sizeof(half);
    int memcpy_per_row = dst->cols / elem_per_memcpy;
    int total_calls = dst->height * dst->width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst->cols;
        
        cuda::memcpy_async(
            (void*)(&((*dst)[{row, col}])),
            (void*)(&src[row*row_stride + col]),
            cuda::aligned_size_t<16>(sizeof(float4)),
            barrier
        );
    }
}

template<ld_registers_tile::all RT, ld_shared::all ST>
__device__ inline static void s2r_load(RT *dst, const ST *src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T  = RT::dtype;
    using U  = ST::dtype;

    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst->height; i++) {
        #pragma unroll
        for(int j = 0; j < dst->width; j++) {
            if constexpr (std::is_same_v<typename RT::layout, ld_registers_tile_layout::row>) {
                // handle the row-major layout
                int row = i*dst->tile_size + (laneid / 4);
                int col = j*dst->tile_size + 2*(laneid % 4);
                dst->tiles[i][j].data.x = *(uint32_t*)(&(*src)[{row+0, col+0}]);
                dst->tiles[i][j].data.y = *(uint32_t*)(&(*src)[{row+8, col+0}]);
                dst->tiles[i][j].data.z = *(uint32_t*)(&(*src)[{row+0, col+8}]);
                dst->tiles[i][j].data.w = *(uint32_t*)(&(*src)[{row+8, col+8}]);
            }
            else {
                // handle the column-major layout
                int row = i*dst->tile_size + 2*(laneid % 4);
                int col = j*dst->tile_size + (laneid / 4);

                half2 tmp;
                tmp.x = (*src)[{row+0, col+0}];
                tmp.y = (*src)[{row+1, col+0}];
                dst->tiles[i][j].data.x = *(reinterpret_cast<uint32_t*>(&tmp));
                tmp.x = (*src)[{row+0, col+8}];
                tmp.y = (*src)[{row+1, col+8}];
                dst->tiles[i][j].data.y = *(reinterpret_cast<uint32_t*>(&tmp));
                tmp.x = (*src)[{row+8, col+0}];
                tmp.y = (*src)[{row+9, col+0}];
                dst->tiles[i][j].data.z = *(reinterpret_cast<uint32_t*>(&tmp));
                tmp.x = (*src)[{row+8, col+8}];
                tmp.y = (*src)[{row+9, col+8}];
                dst->tiles[i][j].data.w = *(reinterpret_cast<uint32_t*>(&tmp));
            }
        }
    }
}

    // // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
    // __device__ static inline void 
    // fma16816(float      & d0, float      & d1, float      & d2, float      & d3,
    //          float const& a0, float const& a1, float const& a2, float const& a3,
    //          float const& b0, float const& b1,
    //          float const& c0, float const& c1, float const& c2, float const& c3){

    //         asm volatile(
    //             "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    //             "{%0,  %1,  %2,  %3},"
    //             "{%4,  %5,  %6,  %7},"
    //             "{%8,  %9},"
    //             "{%10, %11, %12, %13};\n"
    //             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    //             :  "r"(*(uint32_t*)(&a0)),  "r"(*(uint32_t*)(&a1)),  "r"(*(uint32_t*)(&a2)),  "r"(*(uint32_t*)(&a3)),
    //                "r"(*(uint32_t*)(&b0)),  "r"(*(uint32_t*)(&b1)),
    //                "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
    // }

    __device__ static inline void 
    fma16816(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
             uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
             uint32_t const& b0, uint32_t const& b1,
             uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3){

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
                   "r"(b0),  "r"(b1),
                   "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
    }

    // __device__ static inline void fma_AB_base(registers::registers_base<float, registers::row> &d,
    //                                           const registers::registers_base<half, registers::row> &a,
    //                                           const registers::registers_base<half, registers::col> &b, // in col-major mode
    //                                           const registers::registers_base<float, registers::row> &c) {
    //     fma16816(
    //         d.data[0], d.data[1], d.data[2], d.data[3], 
    //         a.data[0], a.data[1], a.data[2], a.data[3],
    //         b.data[0], b.data[1],
    //         c.data[0], c.data[1], c.data[2], c.data[3]
    //     );

    //     fma16816(
    //         d.data[4], d.data[5], d.data[6], d.data[7],
    //         a.data[4], a.data[5], a.data[6], a.data[7],
    //         b.data[2], b.data[3],
    //         c.data[4], c.data[5], c.data[6], c.data[7]
    //     );
    // }

// template<int N, int K, int M>
// __device__ static inline void fma_AB(registers::registers_fl<N, M, registers::row> &d,
//                                const registers::registers_hf<N, K, registers::row> &a,
//                                const registers::registers_hf<K, M, registers::col> &b,
//                                const registers::registers_fl<N, M, registers::row> &c) {
//     #pragma unroll
//     for(int n = 0; n < N; n++) {
//         #pragma unroll
//         for(int m = 0; m < M; m++) {
//             fma_AB_base(
//                 d.tiles[n][m],
//                 a.tiles[n][0],
//                 b.tiles[0][m],
//                 c.tiles[n][m]
//             );
//             #pragma unroll
//             for(int k = 1; k < K; k++) {
//                 fma_AB_base(
//                     d.tiles[n][m],
//                     a.tiles[n][k],
//                     b.tiles[k][m],
//                     d.tiles[n][m]
//                 );
//             }
//         }
//     }
// }

// template<row_layout RT, typename U>
// __device__ inline static void gr_load(RT &dst, const U *src, const int row_stride) {
//     // using T2 = RT::dtype;
//     // using U2 = base_types::packing<U>::packed_type;
//     // int laneid = kittens::laneid();
//     // int warphalf = (laneid & 16) > 0;
//     // int warphalflaneid = laneid % 16;
//     // #pragma unroll
//     // for(int i = 0; i < dst.height; i++) {
//     //     int row_0to3 = i*dst.tile_size + (warphalflaneid / 4);
//     //     #pragma unroll
//     //     for(int j = 0; j < dst.width; j++) {
//     //         int col = j*dst.tile_size + warphalf*8 + 2*(laneid % 4);
//     //         T2 transfers[2];
//     //         transfers[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[(row_0to3+0)*row_stride + col]));
//     //         transfers[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[(row_0to3+4)*row_stride + col]));
//     //         transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
//     //         dst.tiles[i][j].data[0] = transfers[0];
//     //         dst.tiles[i][j].data[2] = transfers[1];
//     //     }
//     //     #pragma unroll
//     //     for(int j = 0; j < dst.width; j++) {
//     //         int col = j*dst.tile_size + warphalf*8 + 2*(laneid % 4);
//     //         T2 transfers[2];
//     //         transfers[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[(row_0to3+ 8)*row_stride + col]));
//     //         transfers[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[(row_0to3+12)*row_stride + col]));
//     //         transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
//     //         dst.tiles[i][j].data[1] = transfers[0];
//     //         dst.tiles[i][j].data[3] = transfers[1];
//     //     }
//     // }
// }

}