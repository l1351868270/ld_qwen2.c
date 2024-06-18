#pragma once

#include <stdint.h>
#include <math.h>
#include "utils.h"

#ifdef AVX512_FWD
#include <immintrin.h>
#endif

namespace ld_infer {
namespace cpu {
namespace rmsnorm {

// https://arxiv.org/pdf/1910.07467
void rmsnormV1_fwd(float* o, float* x, float *weight, float rms_norm_eps, int batch, int dim) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
            
        int offset = b * dim;
        float ss = 0.0f;
        for (int d = 0; d < dim; d++) {
            ss += x[offset + d] * x[ offset + d];
        }
        ss /= dim;
        ss += rms_norm_eps;
        ss = 1.0f / sqrtf(ss);

        int elem_per_cpu = dim / utils::NUM_CPUS;
        
        int t;
        #pragma omp parallel for private(t)
        for (t = 0; t < utils::NUM_CPUS; t++) {
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

void rmsnormV2_fwd(float* o, float* x, float *weight, float rms_norm_eps, int batch, int dim) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
            
        int offset = b * dim;
        float ss = 0.0f;
        for (int d = 0; d < dim; d++) {
            ss += x[offset + d] * x[ offset + d];
        }
        ss /= dim;
        ss += rms_norm_eps;
        ss = 1.0f / sqrtf(ss);
        
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < dim; d++) {
            int offset_o = b * dim + d;
            int offset_w = d;
            o[offset_o] = x[offset_o] * ss * weight[offset_w];
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

#ifdef AVX512_FWD
void rmsnorm_avx512_fwd(float* o, float* x, float *weight, float rms_norm_eps, int batch, int dim) {
    // int b;
    // #pragma omp parallel for private(b)
    for (int b = 0; b < batch; b++) {
            
        int offset = b * dim;
        float ss = 0.0f;
        __m512 a_avx;
        __m512 d_avx;
        for (int d = 0; d < dim; d += 16) {
            a_avx = _mm512_loadu_ps(x + offset + d);  
            d_avx = _mm512_mul_ps(a_avx, a_avx);
            ss += _mm512_reduce_add_ps(d_avx); 
        }
        ss /= dim;
        ss += rms_norm_eps;
        ss = 1.0f / sqrtf(ss);
        
        int d;
        #pragma omp parallel for private(d)
        for (d = 0; d < dim; d += 16) {
            int offset_o = b * dim + d;
            int offset_w = d;
            _mm512_storeu_ps(o + offset_o, ss * _mm512_mul_ps(_mm512_loadu_ps(x + offset_o), _mm512_loadu_ps(weight + offset_w)));
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
#endif

void rmsnorm_fwd(float* o, float* x, float *weight, float rms_norm_eps, int batch, int dim) {
#ifdef AVX512_FWD
    rmsnorm_avx512_fwd(o, x, weight, rms_norm_eps, batch, dim);
#elif OPENMP_V1
    rmsnormV1_fwd(o, x, weight, rms_norm_eps, batch, dim);
#else
    rmsnormV2_fwd(o, x, weight, rms_norm_eps, batch, dim);
#endif
}

} // namespace rmsnorm
} // namespace cpu
} // namespace ld_infer