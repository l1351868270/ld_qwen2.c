#pragma once

#include <stdint.h>
#include "utils.h"

#ifdef AVX512_FWD
#include <immintrin.h>
#endif

#ifdef NEON_FWD
#include <arm_neon.h>
#endif

namespace ld_infer {
namespace cpu {
namespace argmax {

void argmaxV1_fwd(int* output, float* input, int batch, int dim) {
    int b;
    #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int max_i = 0;
        for (int d = 1; d < dim; d++) {
           if (input[b * dim + d] > input[b * dim + max_i]) {
               max_i = d;
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

#ifdef AVX512_FWD
void argmax_avx512_fwd(int* output, float* input, int batch, int dim) {
    int b;
    #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int max_i = 0;
        for (int d = 1; d < dim; d += 16) {
            __m512 max_avx = _mm512_loadu_ps(input + b * dim + d); 
            for (int i = 0; i < 16; i++) {
                if (max_avx[i] > input[b * dim + max_i]) {
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
    int b;
    #pragma omp parallel for private(b)
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
    argmax_avx512_fwd(output, input, batch, dim);
#elif NEON_FWD
    argmax_neon_fwd(output, input, batch, dim);
#else
    argmaxV1_fwd(output, input, batch, dim);
#endif
}

} // namespace argmax
} // namespace cpu
} // namespace ld_infer