#pragma once

#include <stdint.h>
#include <math.h>
#include "utils.h"

namespace ld_infer {
namespace cpu {
namespace rmsnorm {

// https://arxiv.org/pdf/1910.07467
void rmsnorm_fwd(float* o, float* x, float *weight, float rms_norm_eps, int batch, int dim) {
    int b;
    #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
            
        int offset = b * dim;
        float ss = 0.0f;
        for (int d = 0; d < dim; d++) {
            ss += x[offset + d] * x[ offset + d];
        }
        ss /= dim;
        ss += rms_norm_eps;
        ss = 1.0f / sqrtf(ss);

        int elem_per_cpu = dim / NUM_CPUS;
        
        int t;
        #pragma omp parallel for private(t)
        for (t = 0; t < NUM_CPUS; t++) {
            for (int d = 0; d < elem_per_cpu; d++) {
                int offset_o = b * dim + t * elem_per_cpu + d;
                int offset_w = t * elem_per_cpu + d;
                o[offset_o] = x[offset_o] * ss * weight[offset_w];
            }
        }
    }

#ifdef RMSNORM_DEBUG
    printf("rmsnorm:\n");
    for (int b = 0; b < batch; b++) {
        int offset = b * dim;
        printf("[");
        for (int d = 0; d < dim; d++) {
                printf("%f, ", o[offset + d]);
        }
        printf("],\n");
    }
#endif // RMSNORM_DEBUG

}

} // namespace rmsnorm
} // namespace cpu
} // namespace ld_infer