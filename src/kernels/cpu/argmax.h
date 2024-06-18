#pragma once

#include <stdint.h>
#include "utils.h"
#ifdef AVX512_FWD
#include <immintrin.h>
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
#endif

void argmax_fwd(int* output, float* input, int batch, int dim) {
    argmaxV1_fwd(output, input, batch, dim);
}

} // namespace argmax
} // namespace cpu
} // namespace ld_infer