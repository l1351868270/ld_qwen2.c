#pragma once

#include <stdint.h>
#include "utils.h"
#include <omp.h>

#ifdef AVX512_FWD
#include <immintrin.h>
#endif

#ifdef NEON_FWD
#include <arm_neon.h>
#endif

namespace ld_infer {
namespace cpu {
namespace argmax {

template <typename TS>
void argmaxV1_fwd(int* output, TS* input, int batch, int dim) {
    for (int b = 0; b < batch; b++) {
        int max_idx = 0;
        TS max_val = std::numeric_limits<TS>::lowest();

        for (int d = 0; d < dim; d++) {
           if (input[b * dim + d] > max_val) {
               max_val = input[b * dim + d];
               max_idx = d;
           }
        }
        output[b] = max_idx;
    }

#ifdef ARGMAX_DEBUG
    printf("argmax:\n");
    printf("[");
    for (int b = 0; b < batch; b++) {
        printf("%d, %f", output[b], input[b * dim + output[b]]);
    }
    printf("]\n");
#endif // ARGMAX_DEBUG

}

#ifdef AVX512_FWD
template <typename TS>
void argmax_avx512_fwd(int* output, TS* input, int batch, int dim) {
    #pragma omp parallel for
    for (int b = 0; b < batch; b++) {
        int max_i = 0;
        TS max_v = std::numeric_limits<TS>::min();
        for (int d = 0; d < dim; d += 16) {
            __m512 max_avx = _mm512_loadu_ps(input + b * dim + d); 
            for (int i = 0; i < 16; i++) {
                if (max_avx[i] > max_v) {
                    max_v = max_avx[i];
                    max_i = d + i;
                }
            } 
        }
        output[b] = max_i;
    }

#ifdef ARGMAX_DEBUG
    printf("argmax:\n");
    printf("[");
    for (int b = 0; b < batch; b++) {
        printf("%d, %f", output[b], input[b * dim + output[b]]);
    }
    printf("]\n");
#endif // ARGMAX_DEBUG

}
#endif

#ifdef NEON_FWD
template <typename TS>
void argmax_neon_fwd(int* output, TS* input, int batch, int dim) {
    #pragma omp parallel for
    for (int b = 0; b < batch; b++) {
        int max_i = 0;
        for (int d = 1; d < dim; d += 4) {
            float32x4_t max_neon = vld1q_f32(input + b * dim + d); 
            for (int i = 0; i < 4; i++) {
                if (max_neon[i] > input[b * dim + max_i]) {
                    max_i = d + i;
                }
            } 
        }
        output[b] = max_i;
    }

#ifdef ARGMAX_DEBUG
    printf("argmax:\n");
    printf("[");
    for (int b = 0; b < batch; b++) {
        printf("%d, %f", output[b], input[b * dim + output[b]]);
    }
    printf("]\n");
#endif // ARGMAX_DEBUG

}
#endif

template <typename TS>
void argmax_fwd(int* output, TS* input, int batch, int dim) {
#ifdef AVX512_FWD
    // double tdata = omp_get_wtime();
    argmax_avx512_fwd(output, input, batch, dim);
    // tdata = omp_get_wtime() - tdata;
    // printf("batch=%d, dim=%d, in %f secs\n", batch, dim, tdata);
#elif NEON_FWD
    argmax_neon_fwd(output, input, batch, dim);
#else
    // double tdata = omp_get_wtime();
    argmaxV1_fwd(output, input, batch, dim);
    // tdata = omp_get_wtime() - tdata;
    // printf("batch=%d, dim=%d, in %f secs\n", batch, dim, tdata);
#endif
}

} // namespace argmax
} // namespace cpu
} // namespace ld_infer