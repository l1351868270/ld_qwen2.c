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

typedef struct {
    int idx;
    float val;
} argmax_t;

argmax_t argmax_op(argmax_t a, argmax_t b) {
    return (a.val > b.val) ? a : b;
}

#pragma omp declare reduction(argmax_reduction : argmax_t : omp_out = argmax_op(omp_out, omp_in)) initializer(omp_priv = {INT_MIN, -1})

void argmaxV1_fwd(int* output, float* input, int batch, int dim) {
    for (int b = 0; b < batch; b++) {
        int max_idx = 0;
        float max_val = std::numeric_limits<float>::lowest();
        argmax_t max_a = {max_idx, max_val};
        #pragma omp parallel for reduction(argmax_reduction : max_a)
        for (int d = 0; d < dim; d++) {
           if (input[b * dim + d] > max_a.val) {
               max_a.val = input[b * dim + d];
               max_a.idx = d;
           }
        }
        output[b] = max_a.idx;
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
void argmax_avx512_fwd(int* output, float* input, int batch, int dim) {
    #pragma omp parallel for
    for (int b = 0; b < batch; b++) {
        int max_i = 0;
        float max_v = std::numeric_limits<float>::min();
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
void argmax_neon_fwd(int* output, float* input, int batch, int dim) {
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

void argmax_fwd(int* output, float* input, int batch, int dim) {

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