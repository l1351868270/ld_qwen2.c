#pragma once

#include "utils.h"

#ifdef AVX512_FWD
#include <immintrin.h>
#endif

#ifdef NEON_FWD
#include <arm_neon.h>
#endif

namespace ld_infer {
namespace cpu {
namespace residual {

void residualV1_fwd(float *x, float *xb, int batch, int dim) {
    int elem_per_cpu = dim / utils::NUM_CPUS;
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int t;
        #pragma omp parallel for private(t)
        for (t = 0; t < utils::NUM_CPUS; t++) {
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
    // int b;
    // #pragma omp parallel for private(b)
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

#ifdef AVX512_FWD
void residual_avx512_fwd(float *x, float *xb, int batch, int dim) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < dim; d += 16) {
            int offset = b * dim + d;
            _mm512_storeu_ps(x + offset, _mm512_add_ps(_mm512_loadu_ps(x + offset), _mm512_loadu_ps(xb + offset)));
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
#endif

#ifdef NEON_FWD
void residual_neon_fwd(float *x, float *xb, int batch, int dim) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < dim; d += 4) {
            int offset = b * dim + d;
            vst1q_f32(x + offset, vaddq_f32(vld1q_f32(x + offset), vld1q_f32(xb + offset)));
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
#endif

void residual_fwd(float *x, float *xb, int batch, int dim) {
#ifdef AVX512_FWD
    residual_avx512_fwd(x, xb, batch, dim);
#elif NEON_FWD
    residual_neon_fwd(x, xb, batch, dim);
#elif OPENMP_V1
    residualV1_fwd(x, xb, batch, dim);
#else
    residualV2_fwd(x, xb, batch, dim);
#endif
}

} // namespace residual
} // namespace cpu
} // namespace ld_infer