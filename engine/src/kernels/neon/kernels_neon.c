/* ARM NEON kernels: tiers T1 (ARMv7 + NEON) and T2 (AArch64).
 * Entries left NULL fall back to the reference table. */
#include "../kernels.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

#if defined(__aarch64__)
#define NC_FMA_N(acc, b, s) vfmaq_n_f32((acc), (b), (s))
#define NC_FMA(acc, a, b) vfmaq_f32((acc), (a), (b))
#else
#define NC_FMA_N(acc, b, s) vmlaq_n_f32((acc), (b), (s))
#define NC_FMA(acc, a, b) vmlaq_f32((acc), (a), (b))
#endif

static inline float hsum(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
#endif
}

static void add_f32(int64_t n, const float *a, const float *b, float *o) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(o + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) o[i] = a[i] + b[i];
}
static void sub_f32(int64_t n, const float *a, const float *b, float *o) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(o + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) o[i] = a[i] - b[i];
}
static void mul_f32(int64_t n, const float *a, const float *b, float *o) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(o + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) o[i] = a[i] * b[i];
}
static void axpy_f32(int64_t n, float alpha, const float *x, float *y) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, NC_FMA_N(vld1q_f32(y + i), vld1q_f32(x + i), alpha));
    for (; i < n; i++) y[i] += alpha * x[i];
}
static void scale_f32(int64_t n, float s, const float *x, float *o) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(o + i, vmulq_n_f32(vld1q_f32(x + i), s));
    for (; i < n; i++) o[i] = s * x[i];
}
static void relu_f32(int64_t n, const float *x, float *o) {
    const float32x4_t z = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(o + i, vmaxq_f32(vld1q_f32(x + i), z));
    for (; i < n; i++) o[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
static float sum_f32(int64_t n, const float *x) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vaddq_f32(acc0, vld1q_f32(x + i));
        acc1 = vaddq_f32(acc1, vld1q_f32(x + i + 4));
    }
    float s = hsum(vaddq_f32(acc0, acc1));
    for (; i < n; i++) s += x[i];
    return s;
}

static inline void store4(float *c, float32x4_t acc, float alpha, float beta) {
    float32x4_t r = vmulq_n_f32(acc, alpha);
    if (beta != 0.0f) r = NC_FMA_N(r, vld1q_f32(c), beta);
    vst1q_f32(c, r);
}

static inline void store1(float *c, float s, float alpha, float beta) {
    *c = alpha * s + (beta == 0.0f ? 0.0f : beta * *c);
}

#define AT(i, k) (ta ? A[(k) * lda + (i)] : A[(i) * lda + (k)])

static void gemm_cell(int ta, int64_t i, int64_t j, int64_t K, float alpha, const float *A,
                      int64_t lda, const float *B, int64_t ldb, float beta, float *C,
                      int64_t ldc) {
    float s = 0.0f;
    for (int64_t k = 0; k < K; k++) s += AT(i, k) * B[k * ldb + j];
    store1(C + i * ldc + j, s, alpha, beta);
}

/* C = alpha * A @ B^T + beta * C with B stored N x K (the nn.Linear forward).
 * Every C element is a contiguous dot product; 4 columns share each A load. */
static void gemm_nt(int64_t M, int64_t N, int64_t K, float alpha, const float *A, int64_t lda,
                    const float *B, int64_t ldb, float beta, float *C, int64_t ldc) {
    for (int64_t i = 0; i < M; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        int64_t j = 0;
        for (; j + 4 <= N; j += 4) {
            const float *b0 = B + j * ldb, *b1 = b0 + ldb, *b2 = b1 + ldb, *b3 = b2 + ldb;
            float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0), s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
            int64_t k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t av = vld1q_f32(a + k);
                s0 = NC_FMA(s0, av, vld1q_f32(b0 + k));
                s1 = NC_FMA(s1, av, vld1q_f32(b1 + k));
                s2 = NC_FMA(s2, av, vld1q_f32(b2 + k));
                s3 = NC_FMA(s3, av, vld1q_f32(b3 + k));
            }
            float r0 = hsum(s0), r1 = hsum(s1), r2 = hsum(s2), r3 = hsum(s3);
            for (; k < K; k++) {
                r0 += a[k] * b0[k];
                r1 += a[k] * b1[k];
                r2 += a[k] * b2[k];
                r3 += a[k] * b3[k];
            }
            store1(c + j, r0, alpha, beta);
            store1(c + j + 1, r1, alpha, beta);
            store1(c + j + 2, r2, alpha, beta);
            store1(c + j + 3, r3, alpha, beta);
        }
        for (; j < N; j++) {
            const float *b = B + j * ldb;
            float32x4_t s = vdupq_n_f32(0);
            int64_t k = 0;
            for (; k + 4 <= K; k += 4) s = NC_FMA(s, vld1q_f32(a + k), vld1q_f32(b + k));
            float r = hsum(s);
            for (; k < K; k++) r += a[k] * b[k];
            store1(c + j, r, alpha, beta);
        }
    }
}

/* 4x8 register-blocked micro-kernel: each B row segment is loaded once and
 * reused for 4 rows of C kept in 8 accumulator registers. */
static void gemm_f32(int ta, int tb, int64_t M, int64_t N, int64_t K, float alpha,
                     const float *A, int64_t lda, const float *B, int64_t ldb, float beta,
                     float *C, int64_t ldc) {
    if (tb) {
        if (!ta) gemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else nc__kernels_ref()->gemm_f32(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    int64_t i = 0;
    for (; i + 4 <= M; i += 4) {
        int64_t j = 0;
        for (; j + 8 <= N; j += 8) {
            float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
            for (int64_t k = 0; k < K; k++) {
                const float *b = B + k * ldb + j;
                float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4);
                float a0 = AT(i, k), a1 = AT(i + 1, k), a2 = AT(i + 2, k), a3 = AT(i + 3, k);
                c00 = NC_FMA_N(c00, b0, a0); c01 = NC_FMA_N(c01, b1, a0);
                c10 = NC_FMA_N(c10, b0, a1); c11 = NC_FMA_N(c11, b1, a1);
                c20 = NC_FMA_N(c20, b0, a2); c21 = NC_FMA_N(c21, b1, a2);
                c30 = NC_FMA_N(c30, b0, a3); c31 = NC_FMA_N(c31, b1, a3);
            }
            float *c = C + i * ldc + j;
            store4(c, c00, alpha, beta);           store4(c + 4, c01, alpha, beta);
            store4(c + ldc, c10, alpha, beta);     store4(c + ldc + 4, c11, alpha, beta);
            store4(c + 2 * ldc, c20, alpha, beta); store4(c + 2 * ldc + 4, c21, alpha, beta);
            store4(c + 3 * ldc, c30, alpha, beta); store4(c + 3 * ldc + 4, c31, alpha, beta);
        }
        for (; j < N; j++)
            for (int64_t r = 0; r < 4; r++)
                gemm_cell(ta, i + r, j, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    for (; i < M; i++)
        for (int64_t j = 0; j < N; j++)
            gemm_cell(ta, i, j, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#undef AT

static const nc_kernels neon_table = {
    .name = "neon",
    .add_f32 = add_f32,
    .sub_f32 = sub_f32,
    .mul_f32 = mul_f32,
    .div_f32 = NULL, /* no vector divide on ARMv7; ref is auto-vectorized on A64 */
    .axpy_f32 = axpy_f32,
    .scale_f32 = scale_f32,
    .relu_f32 = relu_f32,
    .sum_f32 = sum_f32,
    .gemm_f32 = gemm_f32,
};

const nc_kernels *nc__kernels_neon(void) { return &neon_table; }

#else

const nc_kernels *nc__kernels_neon(void) { return NULL; }

#endif
