/* Portable reference kernels (tier T0). These define correct behaviour for
 * every other kernel table and must build on any C11 compiler. Loops are kept
 * simple enough for compilers to auto-vectorize. */
#include "../kernels.h"

#include <math.h>

static void add_f32(int64_t n, const float *a, const float *b, float *o) {
    for (int64_t i = 0; i < n; i++) o[i] = a[i] + b[i];
}
static void sub_f32(int64_t n, const float *a, const float *b, float *o) {
    for (int64_t i = 0; i < n; i++) o[i] = a[i] - b[i];
}
static void mul_f32(int64_t n, const float *a, const float *b, float *o) {
    for (int64_t i = 0; i < n; i++) o[i] = a[i] * b[i];
}
static void div_f32(int64_t n, const float *a, const float *b, float *o) {
    for (int64_t i = 0; i < n; i++) o[i] = a[i] / b[i];
}
static void axpy_f32(int64_t n, float alpha, const float *x, float *y) {
    for (int64_t i = 0; i < n; i++) y[i] += alpha * x[i];
}
static void scale_f32(int64_t n, float s, const float *x, float *o) {
    for (int64_t i = 0; i < n; i++) o[i] = s * x[i];
}
static void relu_f32(int64_t n, const float *x, float *o) {
    for (int64_t i = 0; i < n; i++) o[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
static float sum_f32(int64_t n, const float *x) {
    /* pairwise-ish: 4 partial sums limit rounding error growth */
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += x[i];
        s1 += x[i + 1];
        s2 += x[i + 2];
        s3 += x[i + 3];
    }
    for (; i < n; i++) s0 += x[i];
    return (s0 + s1) + (s2 + s3);
}

static void exp_f32(int64_t n, const float *x, float *y) {
    for (int64_t i = 0; i < n; i++) y[i] = expf(x[i]);
}
static void sigmoid_f32(int64_t n, const float *x, float *y) {
    for (int64_t i = 0; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
}
static void tanh_f32(int64_t n, const float *x, float *y) {
    for (int64_t i = 0; i < n; i++) y[i] = tanhf(x[i]);
}

static void erf_f32(int64_t n, const float *x, float *y) {
    for (int64_t i = 0; i < n; i++) y[i] = erff(x[i]);
}

static void gemm_f32(int ta, int tb, int64_t M, int64_t N, int64_t K, float alpha,
                     const float *A, int64_t lda, const float *B, int64_t ldb, float beta,
                     float *C, int64_t ldc) {
    for (int64_t i = 0; i < M; i++) {
        float *c = C + i * ldc;
        if (beta == 0.0f) {
            for (int64_t j = 0; j < N; j++) c[j] = 0.0f;
        } else if (beta != 1.0f) {
            for (int64_t j = 0; j < N; j++) c[j] *= beta;
        }
    }
    if (!tb) {
        /* i-k-j order: the inner loop streams a row of B into a row of C.
         * Blocking K and N keeps the B panel (KC x NB floats, 256 KiB) in
         * cache while every row of A passes over it. */
        enum { KC = 256, NB = 256 };
        for (int64_t jb = 0; jb < N; jb += NB) {
            int64_t nb = N - jb < NB ? N - jb : NB;
            for (int64_t kb = 0; kb < K; kb += KC) {
                int64_t ke = K - kb < KC ? K : kb + KC;
                for (int64_t i = 0; i < M; i++) {
                    float *c = C + i * ldc + jb;
                    for (int64_t k = kb; k < ke; k++) {
                        float a = alpha * (ta ? A[k * lda + i] : A[i * lda + k]);
                        if (a == 0.0f) continue;
                        const float *b = B + k * ldb + jb;
                        for (int64_t j = 0; j < nb; j++) c[j] += a * b[j];
                    }
                }
            }
        }
    } else if (!ta) {
        /* B stored N x K: each C element is a contiguous dot product. A
         * block of B rows is reused by every row of A. */
        enum { JB = 64 };
        for (int64_t jb = 0; jb < N; jb += JB) {
            int64_t je = N - jb < JB ? N : jb + JB;
            for (int64_t i = 0; i < M; i++) {
                const float *a = A + i * lda;
                float *c = C + i * ldc;
                for (int64_t j = jb; j < je; j++) {
                    const float *b = B + j * ldb;
                    float s = 0.0f;
                    for (int64_t k = 0; k < K; k++) s += a[k] * b[k];
                    c[j] += alpha * s;
                }
            }
        }
    } else {
        for (int64_t i = 0; i < M; i++) {
            float *c = C + i * ldc;
            for (int64_t j = 0; j < N; j++) {
                float s = 0.0f;
                for (int64_t k = 0; k < K; k++) s += A[k * lda + i] * B[j * ldb + k];
                c[j] += alpha * s;
            }
        }
    }
}

static void gemm_s8_nt(int64_t M, int64_t N, int64_t K, const int8_t *A, int64_t lda, const int8_t *B,
                       int64_t ldb, int32_t *C, int64_t ldc) {
    for (int64_t i = 0; i < M; i++) {
        const int8_t *a = A + i * lda;
        for (int64_t j = 0; j < N; j++) {
            const int8_t *b = B + j * ldb;
            int32_t s = 0;
            for (int64_t k = 0; k < K; k++) s += (int32_t)a[k] * (int32_t)b[k];
            C[i * ldc + j] = s;
        }
    }
}

static const nc_kernels ref_table = {
    .name = "ref",
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

const nc_kernels *nc__kernels_ref(void) { return &ref_table; }
