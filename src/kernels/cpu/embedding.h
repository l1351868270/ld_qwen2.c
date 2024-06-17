#pragma once

#include <concepts>
#include "utils.h"
namespace ld_infer {
namespace cpu {
namespace embedding {

template <typename T>
void embedding_fwd(T *x, T* embed_tokens, int *token, int batch, int dim) {
    int elem_per_cpu = dim / NUM_CPUS;
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        int t;
        #pragma omp parallel for private(t)
        for (t = 0; t < NUM_CPUS; t++) {
            for (int i = 0; i < elem_per_cpu; i++) {
                int offset_x = b * dim + t * elem_per_cpu + i;
                int offset_t = token[b] * dim + t * elem_per_cpu + i;
                x[offset_x] = *(embed_tokens + offset_t);
            }
        }
    }
#ifdef EMBEDDING_DEBUG
    printf("token[b]:%d \n", token[0]);
    printf("[");
    for (int b = 0; b < batch; b++) {
        int offset_x = b * dim;
        printf("[");
        for (int i = 0; i<dim; i++) {
            printf("%f, ", x[offset_x + i]);
        }
        printf("],\n");
    }
    printf("]\n");
#endif // EMBEDDING_DEBUG

}

} // namespace embedding
} // namespace cpu
} // namespace ld_infer
