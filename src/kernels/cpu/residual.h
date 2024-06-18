#pragma once

#include "utils.h"

namespace ld_infer {
namespace cpu {
namespace residual {

void residualV1_fwd(float *x, float *xb, int batch, int dim) {
    int elem_per_cpu = dim / NUM_CPUS;
    int b;
    #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int t;
        #pragma omp parallel for private(t)
        for (t = 0; t < NUM_CPUS; t++) {
            for (int i = 0; i < elem_per_cpu; i++) {
                int offset = b * dim + t * elem_per_cpu + i;
                x[offset] += xb[offset];
            }
        }
    }

#ifdef RESIDUAL_DEBUG
    printf("residual:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < dim; i++) {
            int offset_x = b * dim + i;
            printf("%f, ", x[offset_x]);
        }
        printf("]\n");
    }
#endif // RESIDUAL_DEBUG

}

void residualV2_fwd(float *x, float *xb, int batch, int dim) {
    int b;
    #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < dim; d++) {
            int offset = b * dim + d;
            x[offset] += xb[offset];
        }
    }

#ifdef RESIDUAL_DEBUG
    printf("residual:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < dim; i++) {
            int offset_x = b * dim + i;
            printf("%f, ", x[offset_x]);
        }
        printf("]\n");
    }
#endif // RESIDUAL_DEBUG

}

void residual_fwd(float *x, float *xb, int batch, int dim) {
    residualV2_fwd(x, xb, batch, dim);
}

} // namespace residual
} // namespace cpu
} // namespace ld_infer