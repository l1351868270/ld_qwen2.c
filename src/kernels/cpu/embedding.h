#pragma once

#include <concepts>
#include "utils.h"
#include <math.h>
#ifdef AVX512_FWD
#include <immintrin.h>
#endif

#ifdef NEON_FWD
#include <arm_neon.h>
#endif

namespace ld_infer {
namespace cpu {
namespace embedding {

template<typename T>
concept all = std::is_same_v<typename T::identifier, float>;
template <typename TS, typename TW>
void embeddingV1_fwd(TS *x, TW* embed_tokens, int *token, int batch, int dim) {
    int elem_per_cpu = dim / utils::NUM_CPUS;
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < utils::NUM_CPUS; t++) {
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

template <typename TS, typename TW>
void embeddingV2_fwd(TS *x, TW* embed_tokens, int *token, int batch, int dim) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int d = 0; d < dim; d++) {
            int offset_x = b * dim + d;
            int offset_t = token[b] * dim + d;
            x[offset_x] = *(embed_tokens + offset_t);
        }
    }
#ifdef EMBEDDING_DEBUG
    printf("token[b]:%d \n", token[0]);
    printf("[");
    for (int b = 0; b < batch; b++) {
        int offset_x = b * dim;
        printf("[");
        for (int i = 0; i < dim; i++) {
            printf("%f, ", x[offset_x + i]);
        }
        printf("],\n");
    }
    printf("]\n");
#endif // EMBEDDING_DEBUG

}

#ifdef AVX512_FWD
template <typename TS, typename TW>
void embedding_avx512_fwd(TS *x, TW* embed_tokens, int *token, int batch, int dim) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int d = 0; d < dim; d += 16) {
            int offset_x = b * dim + d;
            int offset_t = token[b] * dim + d;
            _mm512_storeu_ps(x + offset_x, _mm512_loadu_ps(embed_tokens + offset_t));
        }
    }
#ifdef EMBEDDING_DEBUG
    printf("token[b]:%d \n", token[0]);
    printf("[");
    for (int b = 0; b < batch; b++) {
        int offset_x = b * dim;
        printf("[");
        for (int i = 0; i < dim; i++) {
            printf("%f, ", x[offset_x + i]);
        }
        printf("],\n");
    }
    printf("]\n");
#endif // EMBEDDING_DEBUG

}
#endif

#ifdef NEON_FWD
template <typename TS, typename TW>
void embedding_neon_fwd(TS *x, TW* embed_tokens, int *token, int batch, int dim) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int d = 0; d < dim; d += 4) {
            int offset_x = b * dim + d;
            int offset_t = token[b] * dim + d;
            x[offset_x] = *(embed_tokens + offset_t);
            vst1q_f32(x + offset_x, vld1q_f32(embed_tokens + offset_t));
        }
    }
#ifdef EMBEDDING_DEBUG
    printf("token[b]:%d \n", token[0]);
    printf("[");
    for (int b = 0; b < batch; b++) {
        int offset_x = b * dim;
        printf("[");
        for (int i = 0; i < dim; i++) {
            printf("%f, ", x[offset_x + i]);
        }
        printf("],\n");
    }
    printf("]\n");
#endif // EMBEDDING_DEBUG

}
#endif

#ifdef ENABLE_MUTI
template <typename TS, typename TW>
void embedding_muti_fwd(TS *x, TW* embed_tokens, int *token, int batch, int dim, int partial_rank = 0, int partial_size = 1) {
    int elem_per_partial = dim / partial_size;
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int d = 0; d < elem_per_partial; d++) {
            int offset_x = b * elem_per_partial + d;
            int offset_t = token[b] * dim + partial_rank * elem_per_partial + d;
            x[offset_x] = *(embed_tokens + offset_t);
        }
    }

#ifdef EMBEDDING_DEBUG
    printf("token[b]:%d \n", token[0]);
    printf("[");
    for (int b = 0; b < batch; b++) {
        int offset_x = b * dim;
        printf("[");
        for (int i = 0; i < dim; i++) {
            printf("%f, ", x[offset_x + i]);
        }
        printf("],\n");
    }
    printf("]\n");
#endif // EMBEDDING_DEBUG
}
#endif

template <typename TS, typename TW>
void embedding_fwd(TS *x, TW* embed_tokens, int *token, int batch, int dim) {
#ifdef AVX512_FWD
    embedding_avx512_fwd<TS, TW>(x, embed_tokens, token, batch, dim);
#elif NEON_FWD
    embedding_neon_fwd<TS, TW>(x, embed_tokens, token, batch, dim);
#elif OPENMP_V1
    embeddingV1_fwd<TS, TW>(x, embed_tokens, token, batch, dim);
#elif OPENMP_V1
    embeddingV1_fwd<TS, TW>(x, embed_tokens, token, batch, dim);
#else
    embeddingV2_fwd<TS, TW>(x, embed_tokens, token, batch, dim);
#endif
}
} // namespace embedding
} // namespace cpu
} // namespace ld_infer
