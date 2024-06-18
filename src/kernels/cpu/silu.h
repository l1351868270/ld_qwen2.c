#pragma once

#include <math.h>
#include "utils.h"

#ifdef AVX512_FWD
#include <immintrin.h>
#endif

#ifdef NEON_FWD
#include <arm_neon.h>
#endif

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

#ifdef AVX512_FWD
void silu_avx512_fwd(float *hb, float* hb2, int batch, int dim) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < dim; d += 16) {
            int offset = b * dim + d;
            __m512 val_avx = _mm512_loadu_ps(hb + offset);
            for(int i = 0; i < 16; i++) {
                val_avx[i] *= 1.0f / (1.0f + expf(-val_avx[i]));
            }
            _mm512_storeu_ps(hb + offset, _mm512_mul_ps(val_avx, _mm512_loadu_ps(hb2 + offset)));
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
#endif

#ifdef NEON_FWD
void silu_neon_fwd(float *hb, float* hb2, int batch, int dim) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < dim; d += 4) {
            int offset = b * dim + d;
            float32x4_t val_neon = vld1q_f32(hb + offset);
            for(int i = 0; i < 4; i++) {
                val_neon[i] *= 1.0f / (1.0f + expf(-val_neon[i]));
            }
            vst1q_f32(hb + offset, vmulq_f32(val_neon, vld1q_f32(hb2 + offset)));
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
#endif

void silu_fwd(float *hb, float* hb2, int batch, int dim) {
#ifdef AVX512_FWD
    silu_avx512_fwd(hb, hb2, batch, dim);
#elif NEON_FWD
    silu_neon_fwd(hb, hb2, batch, dim);
#elif OPENMP_V1
    siluV1_fwd(hb, hb2, batch, dim);
#else
    siluV2_fwd(hb, hb2, batch, dim);
#endif
}

} // namespace silu
} // namespace cpu
} // namespace ld_infer