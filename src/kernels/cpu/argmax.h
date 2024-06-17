#pragma once

#include <stdint.h>
#include "utils.h"

namespace ld_infer {
namespace cpu {
namespace argmax {

void argmax_fwd(int* output, float* input, int batch, int dim) {
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

} // namespace argmax
} // namespace cpu
} // namespace ld_infer