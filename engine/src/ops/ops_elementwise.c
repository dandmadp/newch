/* Elementwise ops: broadcasting binary ops and unary math. */
#include "../core/nc_internal.h"

#include <math.h>
#include <stdio.h>

enum { BIN_ADD, BIN_SUB, BIN_MUL, BIN_DIV };

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

static nc_status binary_fwd(nc_node *n) {
    nc_tensor *a = n->in[0], *b = n->in[1], *o = n->out;
    const nc_kernels *k = o->ctx->k;
    int kind = n->op->kind;
    if (nc__same_shape(a, b) && nc_is_contiguous(a) && nc_is_contiguous(b)) {
        void (*f)(int64_t, const float *, const float *, float *) =
            kind == BIN_ADD ? k->add_f32 : kind == BIN_SUB ? k->sub_f32 : kind == BIN_MUL ? k->mul_f32 : k->div_f32;
        f(nc_numel(o), (const float *)a->data, (const float *)b->data, (float *)o->data);
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

    for (int which = 0; which < 2; which++) {
        nc_tensor *x = n->in[which];
        if (!nc__needs_grad(x)) continue;
        nc_tensor *gx = nc__grad_buf(x);
        if (!gx) return;
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

enum { UN_NEG, UN_EXP, UN_LOG, UN_SQRT, UN_RELU, UN_SIGMOID, UN_TANH, UN_SCALE, UN_ADD_SCALAR, UN_POW };

static nc_status unary_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    if (!nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    *dt = NC_F32;
    *nd = n->in[0]->ndim;
    memcpy(shape, n->in[0]->shape, sizeof(int64_t) * (size_t)*nd);
    return NC_OK;
}

static nc_status unary_fwd(nc_node *n) {
    const nc_kernels *k = n->out->ctx->k;
    const float *x = nc__contig_f32(n->in[0]);
    if (!x) return NC_ERR_OOM;
    float *y = (float *)n->out->data;
    int64_t N = nc_numel(n->out);
    float s = n->p.f[0];
    switch (n->op->kind) {
    case UN_NEG: k->scale_f32(N, -1.0f, x, y); break;
    case UN_EXP: for (int64_t i = 0; i < N; i++) y[i] = expf(x[i]); break;
    case UN_LOG: for (int64_t i = 0; i < N; i++) y[i] = logf(x[i]); break;
    case UN_SQRT: for (int64_t i = 0; i < N; i++) y[i] = sqrtf(x[i]); break;
    case UN_RELU: k->relu_f32(N, x, y); break;
    case UN_SIGMOID: for (int64_t i = 0; i < N; i++) y[i] = 1.0f / (1.0f + expf(-x[i])); break;
    case UN_TANH: for (int64_t i = 0; i < N; i++) y[i] = tanhf(x[i]); break;
    case UN_SCALE: k->scale_f32(N, s, x, y); break;
    case UN_ADD_SCALAR: for (int64_t i = 0; i < N; i++) y[i] = x[i] + s; break;
    case UN_POW: for (int64_t i = 0; i < N; i++) y[i] = powf(x[i], s); break;
    }
    return NC_OK;
}

static void unary_bwd(nc_node *n) {
    nc_tensor *in = n->in[0];
    nc_tensor *gt = nc__grad_buf(in);
    const float *x = nc__contig_f32(in);
    if (!gt || !x) return;
    const nc_kernels *k = in->ctx->k;
    const float *y = (const float *)n->out->data;
    const float *g = (const float *)n->out->grad->data;
    float *gi = (float *)gt->data;
    int64_t N = nc_numel(in);
    float s = n->p.f[0];
    switch (n->op->kind) {
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
    }
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
