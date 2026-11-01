/* L1 kernel table. Ops never call ISA-specific code directly; they go through
 * ctx->k, which is filled at nc_init() from the reference table and then
 * overridden entry by entry with whatever the running CPU supports. */
#ifndef NC_KERNELS_H
#define NC_KERNELS_H

#include "nc.h"

typedef struct nc_kernels {
    const char *name;

    /* contiguous elementwise: o[i] = a[i] op b[i] */
    void (*add_f32)(int64_t n, const float *a, const float *b, float *o);
    void (*sub_f32)(int64_t n, const float *a, const float *b, float *o);
    void (*mul_f32)(int64_t n, const float *a, const float *b, float *o);
    void (*div_f32)(int64_t n, const float *a, const float *b, float *o);

    /* y[i] += alpha * x[i]   (gradient accumulation workhorse) */
    void (*axpy_f32)(int64_t n, float alpha, const float *x, float *y);
    /* o[i] = s * x[i] */
    void (*scale_f32)(int64_t n, float s, const float *x, float *o);
    void (*relu_f32)(int64_t n, const float *x, float *o);
    float (*sum_f32)(int64_t n, const float *x);
    /* transcendental maps, y may alias x. SIMD tiers use a polynomial exp
     * (relative error ~1e-7), the reference uses libm. */
    void (*exp_f32)(int64_t n, const float *x, float *y);
    void (*sigmoid_f32)(int64_t n, const float *x, float *y);
    void (*tanh_f32)(int64_t n, const float *x, float *y);
    /* erf for GELU: SIMD tiers have absolute error <= 5e-7 (fine next to the
     * 1 in Phi(x) = (1 + erf(x / sqrt 2)) / 2, not for tiny erf values) */
    void (*erf_f32)(int64_t n, const float *x, float *y);

    /* C = alpha * op(A) * op(B) + beta * C, row-major.
     * op(A) is M x K, op(B) is K x N. ta/tb = 1 means the stored matrix is
     * the transpose (A stored K x M, B stored N x K). */
    void (*gemm_f32)(int ta, int tb, int64_t M, int64_t N, int64_t K, float alpha,
                     const float *A, int64_t lda, const float *B, int64_t ldb, float beta,
                     float *C, int64_t ldc);

    /* C[i, j] = sum_k A[i, k] * B[j, k]; A [M, K] and B [N, K] int8 row-major,
     * C [M, N] int32 (dynamic-quantized linear). */
    void (*gemm_s8_nt)(int64_t M, int64_t N, int64_t K, const int8_t *A, int64_t lda, const int8_t *B,
                       int64_t ldb, int32_t *C, int64_t ldc);
} nc_kernels;

const nc_kernels *nc__kernels_ref(void);
/* Each returns NULL when not compiled in for this target. */
const nc_kernels *nc__kernels_neon(void);
const nc_kernels *nc__kernels_avx2(void); /* x86-64 AVX2 + FMA */

/* Picks the best table for `cpu`. The result is a merged static table. */
const nc_kernels *nc__select_kernels(const nc_cpu_info *cpu, int force_reference);

#endif /* NC_KERNELS_H */
