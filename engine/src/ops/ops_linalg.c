/* Matrix multiply: a [..., M, K] @ b [K, N] -> [..., M, N].
 * Leading dims of `a` are folded into M. Transposed views are passed to GEMM
 * as transpose flags instead of being copied. */
#include "../core/nc_internal.h"

#include <stdio.h>

/* A 2-D operand as GEMM sees it: data, transpose flag and leading dim. */
typedef struct {
    const float *p;
    int trans;
    int64_t ld;
} gemm_operand;

/* rows x cols logical matrix stored in t (2-D) */
static gemm_operand as_gemm(nc_tensor *t, int64_t rows, int64_t cols) {
    gemm_operand g;
    if (t->ndim == 2 && t->stride[1] == 1 && t->stride[0] >= cols) {
        g.p = (const float *)t->data, g.trans = 0, g.ld = t->stride[0];
    } else if (t->ndim == 2 && t->stride[0] == 1 && t->stride[1] >= rows) {
        g.p = (const float *)t->data, g.trans = 1, g.ld = t->stride[1];
    } else {
        g.p = nc__contig_f32(t), g.trans = 0, g.ld = cols;
    }
    return g;
}

static nc_status matmul_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *a = n->in[0], *b = n->in[1];
    if (!nc__expect_f32(n, 0) || !nc__expect_f32(n, 1)) return NC_ERR_DTYPE;
    if (a->ndim < 2 || b->ndim != 2 || a->shape[a->ndim - 1] != b->shape[0]) {
        char sa[96], sb[96];
        nc__error(a->ctx, NC_ERR_SHAPE, "matmul: cannot multiply %s by %s (expected [..., M, K] @ [K, N])",
                  nc__shape_str(a->ndim, a->shape, sa, sizeof sa),
                  nc__shape_str(b->ndim, b->shape, sb, sizeof sb));
        return NC_ERR_SHAPE;
    }
    *dt = NC_F32;
    *nd = a->ndim;
    memcpy(shape, a->shape, sizeof(int64_t) * (size_t)a->ndim);
    shape[a->ndim - 1] = b->shape[1];
    return NC_OK;
}

static void dims(const nc_node *n, int64_t *M, int64_t *K, int64_t *N) {
    const nc_tensor *a = n->in[0], *b = n->in[1];
    *K = b->shape[0];
    *N = b->shape[1];
    *M = nc_numel(a) / (*K ? *K : 1);
}

static gemm_operand a_operand(nc_tensor *a, int64_t M, int64_t K) {
    if (a->ndim == 2) return as_gemm(a, M, K);
    gemm_operand g = {nc__contig_f32(a), 0, K}; /* fold leading dims */
    return g;
}

static nc_status matmul_fwd(nc_node *n) {
    int64_t M, K, N;
    dims(n, &M, &K, &N);
    gemm_operand A = a_operand(n->in[0], M, K), B = as_gemm(n->in[1], K, N);
    if (!A.p || !B.p) return NC_ERR_OOM;
    nc__gemm(n->out->ctx, A.trans, B.trans, M, N, K, 1.0f, A.p, A.ld, B.p, B.ld, 0.0f,
                             (float *)n->out->data, N);
    return NC_OK;
}

static void matmul_bwd(nc_node *n) {
    nc_tensor *a = n->in[0], *b = n->in[1];
    int64_t M, K, N;
    dims(n, &M, &K, &N);
    const float *g = (const float *)n->out->grad->data; /* M x N */
    if (nc__needs_grad(a)) {
        /* dA (M x K) += dC (M x N) @ B^T */
        nc_tensor *ga = nc__grad_buf(a);
        gemm_operand B = as_gemm(b, K, N);
        if (ga && B.p) nc__gemm(a->ctx, 0, !B.trans, M, K, N, 1.0f, g, N, B.p, B.ld, 1.0f, (float *)ga->data, K);
    }
    if (nc__needs_grad(b)) {
        /* dB (K x N) += A^T @ dC */
        nc_tensor *gb = nc__grad_buf(b);
        gemm_operand A = a_operand(a, M, K);
        if (gb && A.p) nc__gemm(a->ctx, !A.trans, 0, K, N, M, 1.0f, A.p, A.ld, g, N, 1.0f, (float *)gb->data, N);
    }
}

const nc_op_def nc_op_matmul = {"matmul", 0, 0, matmul_infer, NULL, matmul_fwd, matmul_bwd};

nc_tensor *nc_matmul(nc_tensor *a, nc_tensor *b) {
    nc_tensor *in[2] = {a, b};
    return nc__apply(&nc_op_matmul, in, 2, NULL);
}

/* ------------------------------------------------------------------------ */
/* linear: y = x @ W^T + b, W stored [out, in] (same layout as PyTorch).     */
/* One GEMM per pass, no transposed temporaries.                             */
/* ------------------------------------------------------------------------ */

static nc_status linear_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    for (int i = 0; i < n->n_in; i++)
        if (!nc__expect_f32(n, i)) return NC_ERR_DTYPE;
    if (x->ndim < 1 || w->ndim != 2 || x->shape[x->ndim - 1] != w->shape[1] ||
        (b && (b->ndim != 1 || b->shape[0] != w->shape[0]))) {
        char sx[96], sw[96], sb[96] = "none";
        if (b) nc__shape_str(b->ndim, b->shape, sb, sizeof sb);
        nc__error(x->ctx, NC_ERR_SHAPE,
                  "linear: expected x [..., in], weight [out, in], bias [out]; got %s, %s, %s",
                  nc__shape_str(x->ndim, x->shape, sx, sizeof sx), nc__shape_str(w->ndim, w->shape, sw, sizeof sw), sb);
        return NC_ERR_SHAPE;
    }
    *dt = NC_F32;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    shape[x->ndim - 1] = w->shape[0];
    return NC_OK;
}

static nc_status linear_fwd(nc_node *n) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    int64_t in = w->shape[1], out = w->shape[0], M = in ? nc_numel(x) / in : 0;
    const float *X = nc__contig_f32(x), *W = nc__contig_f32(w), *B = b ? nc__contig_f32(b) : NULL;
    float *Y = (float *)n->out->data;
    if (!X || !W || (b && !B)) return NC_ERR_OOM;
    if (B)
        for (int64_t r = 0; r < M; r++) memcpy(Y + r * out, B, (size_t)out * sizeof(float));
    nc__gemm(x->ctx, 0, 1, M, out, in, 1.0f, X, in, W, in, B ? 1.0f : 0.0f, Y, out);
    return NC_OK;
}

static void linear_bwd(nc_node *n) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    const nc_kernels *k = x->ctx->k;
    int64_t in = w->shape[1], out = w->shape[0], M = in ? nc_numel(x) / in : 0;
    const float *g = (const float *)n->out->grad->data; /* M x out */
    if (nc__needs_grad(x)) {
        nc_tensor *gx = nc__grad_buf(x);
        const float *W = nc__contig_f32(w);
        if (gx && W) nc__gemm(x->ctx, 0, 0, M, in, out, 1.0f, g, out, W, in, 1.0f, (float *)gx->data, in);
    }
    if (nc__needs_grad(w)) {
        nc_tensor *gw = nc__grad_buf(w);
        const float *X = nc__contig_f32(x);
        if (gw && X) nc__gemm(x->ctx, 1, 0, out, in, M, 1.0f, g, out, X, in, 1.0f, (float *)gw->data, in);
    }
    if (b && nc__needs_grad(b)) {
        nc_tensor *gb = nc__grad_buf(b);
        if (gb)
            for (int64_t r = 0; r < M; r++) k->axpy_f32(out, 1.0f, g + r * out, (float *)gb->data);
    }
}

const nc_op_def nc_op_linear = {"linear", 0, 0, linear_infer, NULL, linear_fwd, linear_bwd};

nc_tensor *nc_linear_fn(nc_tensor *x, nc_tensor *weight, nc_tensor *bias) {
    nc_tensor *in[3] = {x, weight, bias};
    return nc__apply(&nc_op_linear, in, bias ? 3 : 2, NULL);
}
