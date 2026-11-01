/* x86-64 AVX2 + FMA kernels. Compiled with per-function target attributes,
 * so the rest of the build stays baseline x86-64 and these are only used
 * when the CPU reports AVX2 and FMA at runtime. */
#include "../kernels.h"

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#include <string.h>

#define NC_AVX2 __attribute__((target("avx2,fma")))

NC_AVX2 static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

NC_AVX2 static inline int32_t hsum8i(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

#define BINARY(nm, vop, sop)                                                          \
    NC_AVX2 static void nm(int64_t n, const float *a, const float *b, float *o) {     \
        int64_t i = 0;                                                                \
        for (; i + 8 <= n; i += 8)                                                    \
            _mm256_storeu_ps(o + i, vop(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i))); \
        for (; i < n; i++) o[i] = a[i] sop b[i];                                      \
    }
BINARY(add_f32, _mm256_add_ps, +)
BINARY(sub_f32, _mm256_sub_ps, -)
BINARY(mul_f32, _mm256_mul_ps, *)
BINARY(div_f32, _mm256_div_ps, /)
#undef BINARY

NC_AVX2 static void axpy_f32(int64_t n, float alpha, const float *x, float *y) {
    __m256 va = _mm256_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += alpha * x[i];
}

NC_AVX2 static void scale_f32(int64_t n, float s, const float *x, float *o) {
    __m256 vs = _mm256_set1_ps(s);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(o + i, _mm256_mul_ps(vs, _mm256_loadu_ps(x + i)));
    for (; i < n; i++) o[i] = s * x[i];
}

NC_AVX2 static void relu_f32(int64_t n, const float *x, float *o) {
    __m256 z = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(o + i, _mm256_max_ps(_mm256_loadu_ps(x + i), z));
    for (; i < n; i++) o[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

NC_AVX2 static float sum_f32(int64_t n, const float *x) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_add_ps(a0, _mm256_loadu_ps(x + i));
        a1 = _mm256_add_ps(a1, _mm256_loadu_ps(x + i + 8));
    }
    float s = hsum8(_mm256_add_ps(a0, a1));
    for (; i < n; i++) s += x[i];
    return s;
}


/* exp: x = n ln2 + r, e^r by a degree-6 polynomial, 2^n built in the exponent
 * bits. Inputs are clamped to the finite float range. */
#define NC_EXP_HI 88.3762626f
#define NC_EXP_LO -87.3365448f
#define NC_LOG2E 1.44269504088896341f
#define NC_LN2_HI 0.693359375f
#define NC_LN2_LO -2.12194440e-4f
#define NC_EXP_P0 1.9875691500e-4f
#define NC_EXP_P1 1.3981999507e-3f
#define NC_EXP_P2 8.3334519073e-3f
#define NC_EXP_P3 4.1665795894e-2f
#define NC_EXP_P4 1.6666665459e-1f
#define NC_EXP_P5 5.0000001201e-1f
/* tanh: odd Taylor polynomial below |x| < 0.25 (no cancellation), else
 * (1 - e) / (1 + e) with e = exp(-2|x|) */
#define NC_TANH_SMALL 0.25f
#define NC_TANH_T3 -0.333333333f
#define NC_TANH_T5 0.133333333f
#define NC_TANH_T7 -0.053968254f
#define NC_TANH_T9 0.021869489f

NC_AVX2 static inline __m256 exp8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(NC_EXP_LO)), _mm256_set1_ps(NC_EXP_HI));
    __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(NC_LOG2E), _mm256_set1_ps(0.5f)));
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(NC_LN2_HI), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(NC_LN2_LO), x);
    __m256 y = _mm256_set1_ps(NC_EXP_P0);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(NC_EXP_P1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(NC_EXP_P2));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(NC_EXP_P3));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(NC_EXP_P4));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(NC_EXP_P5));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

NC_AVX2 static inline __m256 sigmoid8(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

NC_AVX2 static inline __m256 tanh8(__m256 x) {
    __m256 sign = _mm256_set1_ps(-0.0f), one = _mm256_set1_ps(1.0f);
    __m256 ax = _mm256_andnot_ps(sign, x);
    __m256 e = exp8(_mm256_mul_ps(ax, _mm256_set1_ps(-2.0f)));
    __m256 big = _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));
    big = _mm256_or_ps(big, _mm256_and_ps(sign, x)); /* copy the sign of x */
    __m256 z = _mm256_mul_ps(x, x);
    __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(NC_TANH_T9), z, _mm256_set1_ps(NC_TANH_T7));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(NC_TANH_T5));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(NC_TANH_T3));
    p = _mm256_fmadd_ps(_mm256_mul_ps(p, z), x, x);
    return _mm256_blendv_ps(big, p, _mm256_cmp_ps(ax, _mm256_set1_ps(NC_TANH_SMALL), _CMP_LT_OQ));
}

/* erf: rational approximation in t = 1 / (1 + p|x|) times exp(-x^2)
 * (1.5e-7 in exact arithmetic, <= 5e-7 in float), sign restored afterwards */
#define NC_ERF_P 0.3275911f
#define NC_ERF_A1 0.254829592f
#define NC_ERF_A2 -0.284496736f
#define NC_ERF_A3 1.421413741f
#define NC_ERF_A4 -1.453152027f
#define NC_ERF_A5 1.061405429f

NC_AVX2 static inline __m256 erf8(__m256 x) {
    __m256 sign = _mm256_set1_ps(-0.0f), one = _mm256_set1_ps(1.0f);
    __m256 ax = _mm256_andnot_ps(sign, x);
    __m256 t = _mm256_div_ps(one, _mm256_fmadd_ps(ax, _mm256_set1_ps(NC_ERF_P), one));
    __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(NC_ERF_A5), t, _mm256_set1_ps(NC_ERF_A4));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(NC_ERF_A3));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(NC_ERF_A2));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(NC_ERF_A1));
    __m256 e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(ax, ax)));
    __m256 r = _mm256_fnmadd_ps(_mm256_mul_ps(p, t), e, one);
    return _mm256_or_ps(r, _mm256_and_ps(sign, x));
}

/* The tail goes through the same vector code (padded), so every element gets
 * the same arithmetic regardless of its position. */
#define MAP8(nm, fn)                                                       \
    NC_AVX2 static void nm(int64_t n, const float *x, float *y) {          \
        int64_t i = 0;                                                     \
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, fn(_mm256_loadu_ps(x + i))); \
        if (i < n) {                                                       \
            float buf[8] = {0};                                            \
            memcpy(buf, x + i, (size_t)(n - i) * sizeof(float));           \
            _mm256_storeu_ps(buf, fn(_mm256_loadu_ps(buf)));               \
            memcpy(y + i, buf, (size_t)(n - i) * sizeof(float));           \
        }                                                                  \
    }
MAP8(exp_f32, exp8)
MAP8(sigmoid_f32, sigmoid8)
MAP8(tanh_f32, tanh8)
MAP8(erf_f32, erf8)
#undef MAP8

/* ------------------------------------------------------------------------ */
/* GEMM, B not transposed: register-blocked R x 16 micro-tiles (R <= 4) over */
/* KC x NB panels of B that stay in L2.                                      */
/* ------------------------------------------------------------------------ */

#define AT(i, k) (ta ? A[(k) * lda + (i)] : A[(i) * lda + (k)])

NC_AVX2 static inline void store8(float *c, __m256 acc, __m256 valpha, float beta) {
    __m256 r = _mm256_mul_ps(acc, valpha);
    if (beta != 0.0f) r = _mm256_fmadd_ps(_mm256_set1_ps(beta), _mm256_loadu_ps(c), r);
    _mm256_storeu_ps(c, r);
}

/* R rows x (8 * W) cols, W in {1, 2}; R and W are constants after inlining. */
NC_AVX2 static inline void micro(int R, int W, int ta, int64_t i, int64_t j, int64_t kb, int64_t ke, float alpha,
                                 const float *A, int64_t lda, const float *B, int64_t ldb, float beta, float *C,
                                 int64_t ldc) {
    __m256 acc[4][2];
    for (int r = 0; r < R; r++)
        for (int w = 0; w < W; w++) acc[r][w] = _mm256_setzero_ps();
    for (int64_t k = kb; k < ke; k++) {
        const float *b = B + k * ldb + j;
        __m256 b0 = _mm256_loadu_ps(b);
        __m256 b1 = W > 1 ? _mm256_loadu_ps(b + 8) : b0;
        for (int r = 0; r < R; r++) {
            __m256 a = _mm256_set1_ps(AT(i + r, k));
            acc[r][0] = _mm256_fmadd_ps(a, b0, acc[r][0]);
            if (W > 1) acc[r][1] = _mm256_fmadd_ps(a, b1, acc[r][1]);
        }
    }
    __m256 va = _mm256_set1_ps(alpha);
    for (int r = 0; r < R; r++)
        for (int w = 0; w < W; w++) store8(C + (i + r) * ldc + j + 8 * w, acc[r][w], va, beta);
}

NC_AVX2 static void cell(int ta, int64_t i, int64_t j, int64_t kb, int64_t ke, float alpha, const float *A,
                         int64_t lda, const float *B, int64_t ldb, float beta, float *C, int64_t ldc) {
    float s = 0.0f;
    for (int64_t k = kb; k < ke; k++) s += AT(i, k) * B[k * ldb + j];
    float *c = C + i * ldc + j;
    *c = alpha * s + (beta == 0.0f ? 0.0f : beta * *c);
}

#define MICRO_ROWS(R)                                                                                  \
    do {                                                                                               \
        int64_t j = jb;                                                                                \
        for (; j + 16 <= je; j += 16) micro(R, 2, ta, i, j, kb, ke, alpha, A, lda, B, ldb, bt, C, ldc); \
        for (; j + 8 <= je; j += 8) micro(R, 1, ta, i, j, kb, ke, alpha, A, lda, B, ldb, bt, C, ldc);  \
        for (; j < je; j++)                                                                            \
            for (int r = 0; r < R; r++) cell(ta, i + r, j, kb, ke, alpha, A, lda, B, ldb, bt, C, ldc); \
    } while (0)

NC_AVX2 static void gemm_nn(int ta, int64_t M, int64_t N, int64_t K, float alpha, const float *A, int64_t lda,
                            const float *B, int64_t ldb, float beta, float *C, int64_t ldc) {
    enum { KC = 256, NB = 512 };
    if (K == 0) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++) C[i * ldc + j] = beta == 0.0f ? 0.0f : beta * C[i * ldc + j];
        return;
    }
    for (int64_t jb = 0; jb < N; jb += NB) {
        int64_t je = N - jb < NB ? N : jb + NB;
        for (int64_t kb = 0; kb < K; kb += KC) {
            int64_t ke = K - kb < KC ? K : kb + KC;
            float bt = kb == 0 ? beta : 1.0f; /* later K panels accumulate */
            int64_t i = 0;
            for (; i + 4 <= M; i += 4) MICRO_ROWS(4);
            for (; i < M; i++) MICRO_ROWS(1);
        }
    }
}

#undef MICRO_ROWS

/* B stored N x K (nn.Linear forward): dot products, 4 columns share each A
 * load; blocks of B rows are reused across all rows of A. */
NC_AVX2 static void gemm_nt(int64_t M, int64_t N, int64_t K, float alpha, const float *A, int64_t lda,
                            const float *B, int64_t ldb, float beta, float *C, int64_t ldc) {
    enum { JB = 64 };
    for (int64_t jb = 0; jb < N; jb += JB) {
        int64_t je = N - jb < JB ? N : jb + JB;
        for (int64_t i = 0; i < M; i++) {
            const float *a = A + i * lda;
            float *c = C + i * ldc;
            int64_t j = jb;
            for (; j + 4 <= je; j += 4) {
                const float *b0 = B + j * ldb, *b1 = b0 + ldb, *b2 = b1 + ldb, *b3 = b2 + ldb;
                __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
                int64_t k = 0;
                for (; k + 8 <= K; k += 8) {
                    __m256 av = _mm256_loadu_ps(a + k);
                    s0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0 + k), s0);
                    s1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1 + k), s1);
                    s2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b2 + k), s2);
                    s3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b3 + k), s3);
                }
                float r[4] = {hsum8(s0), hsum8(s1), hsum8(s2), hsum8(s3)};
                for (; k < K; k++) {
                    r[0] += a[k] * b0[k];
                    r[1] += a[k] * b1[k];
                    r[2] += a[k] * b2[k];
                    r[3] += a[k] * b3[k];
                }
                for (int q = 0; q < 4; q++) c[j + q] = alpha * r[q] + (beta == 0.0f ? 0.0f : beta * c[j + q]);
            }
            for (; j < je; j++) {
                const float *b = B + j * ldb;
                __m256 s = _mm256_setzero_ps();
                int64_t k = 0;
                for (; k + 8 <= K; k += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), s);
                float r = hsum8(s);
                for (; k < K; k++) r += a[k] * b[k];
                c[j] = alpha * r + (beta == 0.0f ? 0.0f : beta * c[j]);
            }
        }
    }
}

NC_AVX2 static void gemm_f32(int ta, int tb, int64_t M, int64_t N, int64_t K, float alpha, const float *A,
                             int64_t lda, const float *B, int64_t ldb, float beta, float *C, int64_t ldc) {
    if (!tb) gemm_nn(ta, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (!ta) gemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    else nc__kernels_ref()->gemm_f32(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#undef AT

/* int8 dot products via sign-extension to int16 and madd (exact in int32). */
NC_AVX2 static inline __m256i dot16(const int8_t *a, const int8_t *b, __m256i acc) {
    __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)a));
    __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)b));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
}

NC_AVX2 static void gemm_s8_nt(int64_t M, int64_t N, int64_t K, const int8_t *A, int64_t lda, const int8_t *B,
                               int64_t ldb, int32_t *C, int64_t ldc) {
    for (int64_t i = 0; i < M; i++) {
        const int8_t *a = A + i * lda;
        int64_t j = 0;
        for (; j + 4 <= N; j += 4) {
            const int8_t *b0 = B + j * ldb, *b1 = b0 + ldb, *b2 = b1 + ldb, *b3 = b2 + ldb;
            __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
            int64_t k = 0;
            for (; k + 16 <= K; k += 16) {
                s0 = dot16(a + k, b0 + k, s0);
                s1 = dot16(a + k, b1 + k, s1);
                s2 = dot16(a + k, b2 + k, s2);
                s3 = dot16(a + k, b3 + k, s3);
            }
            int32_t r[4] = {hsum8i(s0), hsum8i(s1), hsum8i(s2), hsum8i(s3)};
            for (; k < K; k++) {
                r[0] += a[k] * b0[k];
                r[1] += a[k] * b1[k];
                r[2] += a[k] * b2[k];
                r[3] += a[k] * b3[k];
            }
            for (int q = 0; q < 4; q++) C[i * ldc + j + q] = r[q];
        }
        for (; j < N; j++) {
            const int8_t *b = B + j * ldb;
            __m256i s = _mm256_setzero_si256();
            int64_t k = 0;
            for (; k + 16 <= K; k += 16) s = dot16(a + k, b + k, s);
            int32_t r = hsum8i(s);
            for (; k < K; k++) r += a[k] * b[k];
            C[i * ldc + j] = r;
        }
    }
}

static const nc_kernels avx2_table = {
    .name = "avx2",
    .add_f32 = add_f32,
    .sub_f32 = sub_f32,
    .mul_f32 = mul_f32,
    .div_f32 = div_f32,
    .axpy_f32 = axpy_f32,
    .scale_f32 = scale_f32,
    .relu_f32 = relu_f32,
    .sum_f32 = sum_f32,
    .exp_f32 = exp_f32,
    .sigmoid_f32 = sigmoid_f32,
    .tanh_f32 = tanh_f32,
    .erf_f32 = erf_f32,
    .gemm_f32 = gemm_f32,
    .gemm_s8_nt = gemm_s8_nt,
};

const nc_kernels *nc__kernels_avx2(void) { return &avx2_table; }

#else

const nc_kernels *nc__kernels_avx2(void) { return NULL; }

#endif
