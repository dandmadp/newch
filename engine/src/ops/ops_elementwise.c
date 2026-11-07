/* Elementwise ops: broadcasting binary ops and unary math. */
#include "../core/nc_internal.h"

#include <math.h>
#include <stdio.h>

enum { BIN_ADD, BIN_SUB, BIN_MUL, BIN_DIV };

/* Elementwise work is split into chunks of at least NC_EW_GRAIN elements across the
 * thread pool; every element is computed by exactly one thread, so results do not
 * depend on the thread count. */
#define NC_EW_GRAIN 32768


/* ------------------------------------------------------------------------ */
/* Row functions for nc__iterate                                             */
/* ------------------------------------------------------------------------ */

#define ROW3(nm, expr)                                                                      \
    static void nm(int64_t n, float *const *p, const int64_t *s, const void *u) {           \
        (void)u;                                                                            \
        const float *a = p[0], *b = p[1];                                                   \
        float *o = p[2];                                                                    \
        for (int64_t i = 0; i < n; i++) o[i * s[2]] = (expr);                               \
    }
ROW3(row_add, a[i * s[0]] + b[i * s[1]])
ROW3(row_sub, a[i * s[0]] - b[i * s[1]])
ROW3(row_mul, a[i * s[0]] * b[i * s[1]])
ROW3(row_div, a[i * s[0]] / b[i * s[1]])
#undef ROW3

/* t += alpha * g */
static void row_acc(int64_t n, float *const *p, const int64_t *s, const void *u) {
    float alpha = *(const float *)u;
    for (int64_t i = 0; i < n; i++) p[1][i * s[1]] += alpha * p[0][i * s[0]];
}
/* t += g * x */
static void row_acc_mul(int64_t n, float *const *p, const int64_t *s, const void *u) {
    (void)u;
    for (int64_t i = 0; i < n; i++) p[2][i * s[2]] += p[0][i * s[0]] * p[1][i * s[1]];
}
/* t += g / x */
static void row_acc_div(int64_t n, float *const *p, const int64_t *s, const void *u) {
    (void)u;
    for (int64_t i = 0; i < n; i++) p[2][i * s[2]] += p[0][i * s[0]] / p[1][i * s[1]];
}
/* t -= g * a / b^2 */
static void row_acc_div_b(int64_t n, float *const *p, const int64_t *s, const void *u) {
    (void)u;
    for (int64_t i = 0; i < n; i++) {
        float b = p[2][i * s[2]];
        p[3][i * s[3]] -= p[0][i * s[0]] * p[1][i * s[1]] / (b * b);
    }
}

/* ------------------------------------------------------------------------ */
/* Binary                                                                    */
/* ------------------------------------------------------------------------ */

static nc_status binary_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *a = n->in[0], *b = n->in[1];
    if (!nc__expect_f32(n, 0) || !nc__expect_f32(n, 1)) return NC_ERR_DTYPE;
    int r = a->ndim > b->ndim ? a->ndim : b->ndim;
    for (int i = 0; i < r; i++) {
        int ia = i - (r - a->ndim), ib = i - (r - b->ndim);
        int64_t da = ia < 0 ? 1 : a->shape[ia], db = ib < 0 ? 1 : b->shape[ib];
        if (da != db && da != 1 && db != 1) {
            char sa[96], sb[96];
            nc__error(a->ctx, NC_ERR_SHAPE, "%s: shapes %s and %s are not broadcastable", n->op->name,
                      nc__shape_str(a->ndim, a->shape, sa, sizeof sa),
                      nc__shape_str(b->ndim, b->shape, sb, sizeof sb));
            return NC_ERR_SHAPE;
        }
        shape[i] = da == 1 ? db : da;
    }
    *dt = NC_F32;
    *nd = r;
    return NC_OK;
}

typedef struct {
    void (*f)(int64_t, const float *, const float *, float *);
    const float *a, *b;
    float *o;
} bin_job;

static void bin_fwd_range(void *arg, int64_t b, int64_t e) {
    const bin_job *j = (const bin_job *)arg;
    j->f(e - b, j->a + b, j->b + b, j->o + b);
}

/* backward rows over a flat range when every operand has the output's contiguous layout */
typedef struct {
    nc_row_fn fn;
    int nops;
    float *base[4];
    const void *user;
} flat_job;

static void flat_range(void *arg, int64_t b, int64_t e) {
    const flat_job *j = (const flat_job *)arg;
    float *p[4];
    int64_t st[4] = {1, 1, 1, 1};
    for (int k = 0; k < j->nops; k++) p[k] = j->base[k] + b;
    j->fn(e - b, p, st, j->user);
}

static nc_status binary_fwd(nc_node *n) {
    nc_tensor *a = n->in[0], *b = n->in[1], *o = n->out;
    const nc_kernels *k = o->ctx->k;
    int kind = n->op->kind;
    if (nc__same_shape(a, b) && nc_is_contiguous(a) && nc_is_contiguous(b)) {
        void (*f)(int64_t, const float *, const float *, float *) =
            kind == BIN_ADD ? k->add_f32 : kind == BIN_SUB ? k->sub_f32 : kind == BIN_MUL ? k->mul_f32 : k->div_f32;
        bin_job j = {f, (const float *)a->data, (const float *)b->data, (float *)o->data};
        nc__parallel_for(o->ctx, nc_numel(o), NC_EW_GRAIN, bin_fwd_range, &j);
        return NC_OK;
    }
    int64_t st[3][NC_MAX_DIMS];
    nc__bcast_strides(a, o->ndim, o->shape, st[0]);
    nc__bcast_strides(b, o->ndim, o->shape, st[1]);
    memcpy(st[2], o->stride, sizeof st[2]);
    float *base[3] = {(float *)a->data, (float *)b->data, (float *)o->data};
    nc_row_fn fn = kind == BIN_ADD ? row_add : kind == BIN_SUB ? row_sub : kind == BIN_MUL ? row_mul : row_div;
    nc__iterate(3, base, (const int64_t(*)[NC_MAX_DIMS])st, o->ndim, o->shape, fn, NULL);
    return NC_OK;
}

static void binary_bwd(nc_node *n) {
    nc_tensor *a = n->in[0], *b = n->in[1], *o = n->out;
    float *g = (float *)o->grad->data;
    int nd = o->ndim, kind = n->op->kind;
    int64_t sg[NC_MAX_DIMS], sa[NC_MAX_DIMS], sb[NC_MAX_DIMS];
    memcpy(sg, o->stride, sizeof sg);
    nc__bcast_strides(a, nd, o->shape, sa);
    nc__bcast_strides(b, nd, o->shape, sb);

    int flat = nc__same_shape(a, o) && nc__same_shape(b, o) && nc_is_contiguous(a) && nc_is_contiguous(b);
    for (int which = 0; which < 2; which++) {
        nc_tensor *x = n->in[which];
        if (!nc__needs_grad(x)) continue;
        nc_tensor *gx = nc__grad_buf(x);
        if (!gx) return;
        if (flat && kind != BIN_ADD && kind != BIN_SUB) {
            flat_job j;
            if (kind == BIN_MUL || which == 0) {
                j = (flat_job){kind == BIN_MUL ? row_acc_mul : row_acc_div, 3,
                               {g, (float *)(which == 0 ? b : a)->data, (float *)gx->data, NULL}, NULL};
            } else {
                j = (flat_job){row_acc_div_b, 4, {g, (float *)a->data, (float *)b->data, (float *)gx->data}, NULL};
            }
            nc__parallel_for(o->ctx, nc_numel(o), NC_EW_GRAIN, flat_range, &j);
            continue;
        }
        int64_t st[4][NC_MAX_DIMS];
        memcpy(st[0], sg, sizeof sg);
        int64_t sgx[NC_MAX_DIMS];
        nc__bcast_strides_shape(x->ndim, x->shape, nd, o->shape, sgx);

        if (kind == BIN_ADD || kind == BIN_SUB) {
            float alpha = (kind == BIN_SUB && which == 1) ? -1.0f : 1.0f;
            if (nc__same_shape(x, o)) {
                o->ctx->k->axpy_f32(nc_numel(o), alpha, g, (float *)gx->data);
                continue;
            }
            memcpy(st[1], sgx, sizeof sgx);
            float *base[2] = {g, (float *)gx->data};
            nc__iterate(2, base, (const int64_t(*)[NC_MAX_DIMS])st, nd, o->shape, row_acc, &alpha);
        } else if (kind == BIN_MUL || which == 0) {
            /* d(a*b)/da = b, d(a*b)/db = a, d(a/b)/da = 1/b */
            nc_tensor *other = which == 0 ? b : a;
            memcpy(st[1], which == 0 ? sb : sa, sizeof sa);
            memcpy(st[2], sgx, sizeof sgx);
            float *base[3] = {g, (float *)other->data, (float *)gx->data};
            nc__iterate(3, base, (const int64_t(*)[NC_MAX_DIMS])st, nd, o->shape,
                        kind == BIN_MUL ? row_acc_mul : row_acc_div, NULL);
        } else {
            /* d(a/b)/db = -a / b^2 */
            memcpy(st[1], sa, sizeof sa);
            memcpy(st[2], sb, sizeof sb);
            memcpy(st[3], sgx, sizeof sgx);
            float *base[4] = {g, (float *)a->data, (float *)b->data, (float *)gx->data};
            nc__iterate(4, base, (const int64_t(*)[NC_MAX_DIMS])st, nd, o->shape, row_acc_div_b, NULL);
        }
    }
}

#define BINARY_OP(nm, KIND) \
    const nc_op_def nc_op_##nm = {#nm, NC_OPF_ELEMENTWISE, KIND, binary_infer, NULL, binary_fwd, binary_bwd};
BINARY_OP(add, BIN_ADD)
BINARY_OP(sub, BIN_SUB)
BINARY_OP(mul, BIN_MUL)
BINARY_OP(div, BIN_DIV)
#undef BINARY_OP

static nc_tensor *binary(const nc_op_def *op, nc_tensor *a, nc_tensor *b) {
    nc_tensor *in[2] = {a, b};
    return nc__apply(op, in, 2, NULL);
}

nc_tensor *nc_add(nc_tensor *a, nc_tensor *b) { return binary(&nc_op_add, a, b); }
nc_tensor *nc_sub(nc_tensor *a, nc_tensor *b) { return binary(&nc_op_sub, a, b); }
nc_tensor *nc_mul(nc_tensor *a, nc_tensor *b) { return binary(&nc_op_mul, a, b); }
nc_tensor *nc_div(nc_tensor *a, nc_tensor *b) { return binary(&nc_op_div, a, b); }

/* ------------------------------------------------------------------------ */
/* Unary                                                                     */
/* ------------------------------------------------------------------------ */

enum {
    UN_NEG, UN_EXP, UN_LOG, UN_SQRT, UN_RELU, UN_SIGMOID, UN_TANH, UN_SCALE, UN_ADD_SCALAR, UN_POW,
    UN_LEAKY_RELU, UN_RELU6, UN_HARDSIGMOID, UN_HARDSWISH, UN_SILU, UN_GELU, UN_GELU_TANH
};

/* GELU: exact form uses erf, the approximation tanh (as PyTorch's approximate="tanh"). */
#define NC_SQRT1_2 0.70710678118654752f
#define NC_INV_SQRT_2PI 0.39894228040143268f
#define NC_SQRT_2_PI 0.79788456080286536f
#define NC_GELU_C 0.044715f

static nc_status unary_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    if (!nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    *dt = NC_F32;
    *nd = n->in[0]->ndim;
    memcpy(shape, n->in[0]->shape, sizeof(int64_t) * (size_t)*nd);
    return NC_OK;
}

static void unary_fwd_span(const nc_kernels *k, int kind, float s, const float *x, float *y, int64_t N) {
    switch (kind) {
    case UN_NEG: k->scale_f32(N, -1.0f, x, y); break;
    case UN_EXP: k->exp_f32(N, x, y); break;
    case UN_LOG: for (int64_t i = 0; i < N; i++) y[i] = logf(x[i]); break;
    case UN_SQRT: for (int64_t i = 0; i < N; i++) y[i] = sqrtf(x[i]); break;
    case UN_RELU: k->relu_f32(N, x, y); break;
    case UN_SIGMOID: k->sigmoid_f32(N, x, y); break;
    case UN_TANH: k->tanh_f32(N, x, y); break;
    case UN_SCALE: k->scale_f32(N, s, x, y); break;
    case UN_ADD_SCALAR: for (int64_t i = 0; i < N; i++) y[i] = x[i] + s; break;
    case UN_POW: for (int64_t i = 0; i < N; i++) y[i] = powf(x[i], s); break;
    /* Piecewise-linear activations: only compares and multiplies, so they stay
     * cheap on small cores and keep a bounded range for int8. */
    case UN_LEAKY_RELU: for (int64_t i = 0; i < N; i++) y[i] = x[i] > 0.0f ? x[i] : s * x[i]; break;
    case UN_RELU6: for (int64_t i = 0; i < N; i++) y[i] = fminf(fmaxf(x[i], 0.0f), 6.0f); break;
    case UN_HARDSIGMOID:
        for (int64_t i = 0; i < N; i++) y[i] = fminf(fmaxf(x[i] * (1.0f / 6.0f) + 0.5f, 0.0f), 1.0f);
        break;
    case UN_HARDSWISH:
        for (int64_t i = 0; i < N; i++) y[i] = x[i] * fminf(fmaxf(x[i] + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
        break;
    case UN_SILU:
        k->sigmoid_f32(N, x, y);
        for (int64_t i = 0; i < N; i++) y[i] *= x[i];
        break;
    case UN_GELU:
        k->scale_f32(N, NC_SQRT1_2, x, y);
        k->erf_f32(N, y, y);
        for (int64_t i = 0; i < N; i++) y[i] = 0.5f * x[i] * (1.0f + y[i]);
        break;
    case UN_GELU_TANH:
        for (int64_t i = 0; i < N; i++) y[i] = NC_SQRT_2_PI * (x[i] + NC_GELU_C * x[i] * x[i] * x[i]);
        k->tanh_f32(N, y, y);
        for (int64_t i = 0; i < N; i++) y[i] = 0.5f * x[i] * (1.0f + y[i]);
        break;
    }
}

typedef struct {
    const nc_kernels *k;
    int kind;
    float s;
    const float *x, *y, *g;
    float *out, *gi, *tmp, *tmp2;
} un_job;

static void unary_fwd_range(void *arg, int64_t b, int64_t e) {
    const un_job *j = (const un_job *)arg;
    unary_fwd_span(j->k, j->kind, j->s, j->x + b, j->out + b, e - b);
}

static nc_status unary_fwd(nc_node *n) {
    const float *x = nc__contig_f32(n->in[0]);
    if (!x) return NC_ERR_OOM;
    un_job j = {n->out->ctx->k, n->op->kind, n->p.f[0], x, NULL, NULL, (float *)n->out->data, NULL, NULL, NULL};
    nc__parallel_for(n->out->ctx, nc_numel(n->out), NC_EW_GRAIN, unary_fwd_range, &j);
    return NC_OK;
}

static void unary_bwd_span(const nc_kernels *k, int kind, float s, const float *x, const float *y,
                           const float *g, float *gi, float *tmp, float *tmp2, int64_t N) {
    if (kind == UN_SILU) {
        k->sigmoid_f32(N, x, tmp);
    } else if (kind == UN_GELU) {
        k->scale_f32(N, NC_SQRT1_2, x, tmp);
        k->erf_f32(N, tmp, tmp);
        for (int64_t i = 0; i < N; i++) tmp2[i] = -0.5f * x[i] * x[i];
        k->exp_f32(N, tmp2, tmp2);
    } else if (kind == UN_GELU_TANH) {
        for (int64_t i = 0; i < N; i++) tmp[i] = NC_SQRT_2_PI * (x[i] + NC_GELU_C * x[i] * x[i] * x[i]);
        k->tanh_f32(N, tmp, tmp);
    }
    switch (kind) {
    case UN_NEG: k->axpy_f32(N, -1.0f, g, gi); break;
    case UN_EXP: for (int64_t i = 0; i < N; i++) gi[i] += g[i] * y[i]; break;
    case UN_LOG: for (int64_t i = 0; i < N; i++) gi[i] += g[i] / x[i]; break;
    case UN_SQRT: for (int64_t i = 0; i < N; i++) gi[i] += g[i] * 0.5f / y[i]; break;
    case UN_RELU: for (int64_t i = 0; i < N; i++) gi[i] += x[i] > 0.0f ? g[i] : 0.0f; break;
    case UN_SIGMOID: for (int64_t i = 0; i < N; i++) gi[i] += g[i] * y[i] * (1.0f - y[i]); break;
    case UN_TANH: for (int64_t i = 0; i < N; i++) gi[i] += g[i] * (1.0f - y[i] * y[i]); break;
    case UN_SCALE: k->axpy_f32(N, s, g, gi); break;
    case UN_ADD_SCALAR: k->axpy_f32(N, 1.0f, g, gi); break;
    case UN_POW: for (int64_t i = 0; i < N; i++) gi[i] += g[i] * s * powf(x[i], s - 1.0f); break;
    /* kinks take the PyTorch convention (open intervals) */
    case UN_LEAKY_RELU: for (int64_t i = 0; i < N; i++) gi[i] += x[i] > 0.0f ? g[i] : s * g[i]; break;
    case UN_RELU6: for (int64_t i = 0; i < N; i++) gi[i] += (x[i] > 0.0f && x[i] < 6.0f) ? g[i] : 0.0f; break;
    case UN_HARDSIGMOID:
        for (int64_t i = 0; i < N; i++) gi[i] += (x[i] > -3.0f && x[i] < 3.0f) ? g[i] * (1.0f / 6.0f) : 0.0f;
        break;
    case UN_HARDSWISH:
        for (int64_t i = 0; i < N; i++)
            gi[i] += x[i] < -3.0f ? 0.0f : x[i] <= 3.0f ? g[i] * (x[i] * (1.0f / 3.0f) + 0.5f) : g[i];
        break;
    case UN_SILU:
        for (int64_t i = 0; i < N; i++) gi[i] += g[i] * tmp[i] * (1.0f + x[i] * (1.0f - tmp[i]));
        break;
    case UN_GELU:
        for (int64_t i = 0; i < N; i++) gi[i] += g[i] * (0.5f * (1.0f + tmp[i]) + x[i] * NC_INV_SQRT_2PI * tmp2[i]);
        break;
    case UN_GELU_TANH:
        for (int64_t i = 0; i < N; i++) {
            float v = x[i];
            float t = tmp[i];
            float dt = (1.0f - t * t) * NC_SQRT_2_PI * (1.0f + 3.0f * NC_GELU_C * v * v);
            gi[i] += g[i] * (0.5f * (1.0f + t) + 0.5f * v * dt);
        }
        break;
    }
}

static void unary_bwd_range(void *arg, int64_t b, int64_t e) {
    const un_job *j = (const un_job *)arg;
    unary_bwd_span(j->k, j->kind, j->s, j->x + b, j->y + b, j->g + b, j->gi + b, j->tmp ? j->tmp + b : NULL,
                   j->tmp2 ? j->tmp2 + b : NULL, e - b);
}

static void unary_bwd(nc_node *n) {
    nc_tensor *in = n->in[0];
    nc_tensor *gt = nc__grad_buf(in);
    const float *x = nc__contig_f32(in);
    if (!gt || !x) return;
    int64_t N = nc_numel(in);
    int kind = n->op->kind;
    float *tmp = NULL, *tmp2 = NULL; /* sigmoid / erf / tanh values recomputed for SiLU and GELU */
    if (kind == UN_SILU || kind == UN_GELU || kind == UN_GELU_TANH) {
        tmp = (float *)nc__arena_alloc(in->ctx, (size_t)(N ? N : 1) * sizeof(float));
        if (!tmp) return;
        if (kind == UN_GELU) {
            tmp2 = (float *)nc__arena_alloc(in->ctx, (size_t)(N ? N : 1) * sizeof(float));
            if (!tmp2) return;
        }
    }
    un_job j = {in->ctx->k, kind, n->p.f[0], x, (const float *)n->out->data, (const float *)n->out->grad->data,
                NULL, (float *)gt->data, tmp, tmp2};
    nc__parallel_for(in->ctx, N, NC_EW_GRAIN, unary_bwd_range, &j);
}

#define UNARY_OP(nm, KIND) \
    const nc_op_def nc_op_##nm = {#nm, NC_OPF_ELEMENTWISE, KIND, unary_infer, NULL, unary_fwd, unary_bwd};
UNARY_OP(neg, UN_NEG)
UNARY_OP(exp, UN_EXP)
UNARY_OP(log, UN_LOG)
UNARY_OP(sqrt, UN_SQRT)
UNARY_OP(relu, UN_RELU)
UNARY_OP(sigmoid, UN_SIGMOID)
UNARY_OP(tanh, UN_TANH)
UNARY_OP(scale, UN_SCALE)
UNARY_OP(add_scalar, UN_ADD_SCALAR)
UNARY_OP(pow_scalar, UN_POW)
UNARY_OP(leaky_relu, UN_LEAKY_RELU)
UNARY_OP(relu6, UN_RELU6)
UNARY_OP(hardsigmoid, UN_HARDSIGMOID)
UNARY_OP(hardswish, UN_HARDSWISH)
UNARY_OP(silu, UN_SILU)
UNARY_OP(gelu, UN_GELU)
UNARY_OP(gelu_tanh, UN_GELU_TANH)
#undef UNARY_OP

static nc_tensor *unary(const nc_op_def *op, nc_tensor *x, float s) {
    nc_op_params p = {{0}, {0}};
    p.f[0] = s;
    return nc__apply(op, &x, 1, &p);
}

nc_tensor *nc_neg(nc_tensor *x) { return unary(&nc_op_neg, x, 0); }
nc_tensor *nc_exp(nc_tensor *x) { return unary(&nc_op_exp, x, 0); }
nc_tensor *nc_log(nc_tensor *x) { return unary(&nc_op_log, x, 0); }
nc_tensor *nc_sqrt(nc_tensor *x) { return unary(&nc_op_sqrt, x, 0); }
nc_tensor *nc_relu(nc_tensor *x) { return unary(&nc_op_relu, x, 0); }
nc_tensor *nc_sigmoid(nc_tensor *x) { return unary(&nc_op_sigmoid, x, 0); }
nc_tensor *nc_tanh(nc_tensor *x) { return unary(&nc_op_tanh, x, 0); }
nc_tensor *nc_scale(nc_tensor *x, float s) { return unary(&nc_op_scale, x, s); }
nc_tensor *nc_add_scalar(nc_tensor *x, float s) { return unary(&nc_op_add_scalar, x, s); }
nc_tensor *nc_pow_scalar(nc_tensor *x, float p) { return unary(&nc_op_pow_scalar, x, p); }
nc_tensor *nc_leaky_relu(nc_tensor *x, float negative_slope) { return unary(&nc_op_leaky_relu, x, negative_slope); }
nc_tensor *nc_relu6(nc_tensor *x) { return unary(&nc_op_relu6, x, 0); }
nc_tensor *nc_hardsigmoid(nc_tensor *x) { return unary(&nc_op_hardsigmoid, x, 0); }
nc_tensor *nc_hardswish(nc_tensor *x) { return unary(&nc_op_hardswish, x, 0); }
nc_tensor *nc_silu(nc_tensor *x) { return unary(&nc_op_silu, x, 0); }
nc_tensor *nc_gelu(nc_tensor *x, int approximate_tanh) {
    return unary(approximate_tanh ? &nc_op_gelu_tanh : &nc_op_gelu, x, 0);
}

/* ------------------------------------------------------------------------ */
/* Comparisons -> i32 0/1. f32 and i32 are both 4 bytes, so nc__iterate's     */
/* float pointers walk either; the row function reads them by dtype.          */
/* ------------------------------------------------------------------------ */

typedef struct {
    int op;
    nc_dtype da, db;
} cmp_info;

static void row_cmp(int64_t n, float *const *p, const int64_t *s, const void *u) {
    const cmp_info *c = (const cmp_info *)u;
    for (int64_t i = 0; i < n; i++) {
        const float *pa = p[0] + i * s[0], *pb = p[1] + i * s[1];
        double a = c->da == NC_I32 ? (double)*(const int32_t *)(const void *)pa : (double)*pa;
        double b = c->db == NC_I32 ? (double)*(const int32_t *)(const void *)pb : (double)*pb;
        int r = c->op == NC_CMP_EQ   ? a == b
                : c->op == NC_CMP_NE ? a != b
                : c->op == NC_CMP_LT ? a < b
                : c->op == NC_CMP_LE ? a <= b
                : c->op == NC_CMP_GT ? a > b
                                     : a >= b;
        *(int32_t *)(void *)(p[2] + i * s[2]) = r;
    }
}

static nc_status compare_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    for (int k = 0; k < 2; k++) {
        if (n->in[k]->dtype != NC_F32 && n->in[k]->dtype != NC_I32) {
            nc__error(n->in[k]->ctx, NC_ERR_DTYPE, "compare: inputs must be f32 or i32, got %s",
                      nc__dtype_name(n->in[k]->dtype));
            return NC_ERR_DTYPE;
        }
    }
    nc_tensor *a = n->in[0], *b = n->in[1];
    int r = a->ndim > b->ndim ? a->ndim : b->ndim;
    for (int i = 0; i < r; i++) {
        int ia = i - (r - a->ndim), ib = i - (r - b->ndim);
        int64_t da = ia < 0 ? 1 : a->shape[ia], db = ib < 0 ? 1 : b->shape[ib];
        if (da != db && da != 1 && db != 1) {
            char sa[96], sb[96];
            nc__error(a->ctx, NC_ERR_SHAPE, "compare: shapes %s and %s are not broadcastable",
                      nc__shape_str(a->ndim, a->shape, sa, sizeof sa), nc__shape_str(b->ndim, b->shape, sb, sizeof sb));
            return NC_ERR_SHAPE;
        }
        shape[i] = da == 1 ? db : da;
    }
    *dt = NC_I32;
    *nd = r;
    return NC_OK;
}

static nc_status compare_fwd(nc_node *n) {
    nc_tensor *a = n->in[0], *b = n->in[1], *o = n->out;
    int64_t st[3][NC_MAX_DIMS];
    nc__bcast_strides(a, o->ndim, o->shape, st[0]);
    nc__bcast_strides(b, o->ndim, o->shape, st[1]);
    memcpy(st[2], o->stride, sizeof st[2]);
    float *base[3] = {(float *)a->data, (float *)b->data, (float *)o->data};
    cmp_info c = {(int)n->p.i[0], a->dtype, b->dtype};
    nc__iterate(3, base, (const int64_t(*)[NC_MAX_DIMS])st, o->ndim, o->shape, row_cmp, &c);
    return NC_OK;
}

const nc_op_def nc_op_compare = {"compare", NC_OPF_NONDIFF, 0, compare_infer, NULL, compare_fwd, NULL};

nc_tensor *nc_compare(nc_tensor *a, nc_tensor *b, nc_cmp op) {
    /* 1-byte integers (u8 data, i8) compare as i32: exact, and keeps the row loop 4-byte */
    if (a && (a->dtype == NC_U8 || a->dtype == NC_I8)) a = nc_to_dtype(a, NC_I32);
    if (b && (b->dtype == NC_U8 || b->dtype == NC_I8)) b = nc_to_dtype(b, NC_I32);
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = op;
    nc_tensor *in[2] = {a, b};
    return nc__apply(&nc_op_compare, in, 2, &p);
}
