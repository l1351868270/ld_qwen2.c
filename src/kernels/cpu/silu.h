#pragma once

#include <math.h>
#include "utils.h"

namespace ld_infer {
namespace cpu {
namespace silu {

// https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
void siluV1_fwd(float *hb, float* hb2, int batch, int dim) {
    int elem_per_cpu = dim / utils::NUM_CPUS;
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int t;
        #pragma omp parallel for private(t)
        for (t = 0; t < utils::NUM_CPUS; t++) {
            for (int i = 0; i < elem_per_cpu; i++) {
                int offset = b * dim + t * elem_per_cpu + i;
                float val = hb[offset];
                val *= 1.0f / (1.0f + expf(-val));
                val *= hb2[offset];
                hb[offset] = val;
            }
        }
    }

#ifdef SILU_DEBUG
    printf("silu:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < dim; i++) {
            printf("%f, ", hb[b * dim + i]);
        }
        printf("]\n");
    }
# endif // SILU_DEBUG

}

void siluV2_fwd(float *hb, float* hb2, int batch, int dim) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < dim; d++) {
            int offset = b * dim + d;
            float val = hb[offset];
            val *= 1.0f / (1.0f + expf(-val));
            val *= hb2[offset];
            hb[offset] = val;
        }
    }

#ifdef SILU_DEBUG
    printf("silu:\n");
    for (int b = 0; b < batch; b++) {
        printf("[");
        for (int i = 0; i < dim; i++) {
            printf("%f, ", hb[b * dim + i]);
        }
        printf("]\n");
    }
# endif // SILU_DEBUG

}

void silu_fwd(float *hb, float* hb2, int batch, int dim) {
#ifdef OPENMP_V1
    siluV1_fwd(hb, hb2, batch, dim);
#else
    siluV2_fwd(hb, hb2, batch, dim);
#endif
}

} // namespace silu
} // namespace cpu
} // namespace ld_infer