/* Matrix multiply. a [..., M, K] @ b [K, N] -> [..., M, N] folds the leading
 * dims of `a` into M (one GEMM); a 3-D+ `b` goes to the batched op below.
 * Transposed views are passed to GEMM as transpose flags instead of being
 * copied. */
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

/* ------------------------------------------------------------------------ */
/* Batched matmul: a [..., M, K] @ b [..., K, N] with broadcast batch dims    */
/* (torch.matmul for 3-D+ operands). One GEMM per batch matrix; transposed    */
/* views (k.transpose(-2, -1) in attention) go to GEMM as flags, not copies.  */
/* ------------------------------------------------------------------------ */

typedef struct {
    int nb;                      /* batch dims of the output */
    int64_t shape[NC_MAX_DIMS];  /* output batch shape */
    int64_t count;               /* number of batch matrices */
} batch_dims;

static int batch_shape(const nc_tensor *a, const nc_tensor *b, batch_dims *bd) {
    int na = a->ndim - 2, nb = b->ndim - 2;
    bd->nb = na > nb ? na : nb;
    bd->count = 1;
    for (int i = 0; i < bd->nb; i++) {
        int ia = i - (bd->nb - na), ib = i - (bd->nb - nb);
        int64_t da = ia < 0 ? 1 : a->shape[ia], db = ib < 0 ? 1 : b->shape[ib];
        if (da != db && da != 1 && db != 1) return 0;
        bd->shape[i] = da == 1 ? db : da;
        bd->count *= bd->shape[i];
    }
    return 1;
}

/* per output batch dim: element stride of t's batch matrices (0 = broadcast) */
static void batch_strides(const nc_tensor *t, const int64_t *st, const batch_dims *bd, int64_t *out) {
    int tb = t->ndim - 2;
    for (int i = 0; i < bd->nb; i++) {
        int j = i - (bd->nb - tb);
        out[i] = (j < 0 || t->shape[j] == 1) ? 0 : st[j];
    }
}

static int64_t batch_offset(const batch_dims *bd, const int64_t *st, int64_t idx) {
    int64_t off = 0;
    for (int i = bd->nb - 1; i >= 0; i--) {
        off += (idx % bd->shape[i]) * st[i];
        idx /= bd->shape[i];
    }
    return off;
}

typedef struct {
    const float *p;
    int trans;
    int64_t ld;
    int64_t bst[NC_MAX_DIMS];
} bmm_operand;

static int bmm_operand_of(nc_tensor *t, const batch_dims *bd, bmm_operand *o) {
    int nd = t->ndim;
    int64_t rows = t->shape[nd - 2], cols = t->shape[nd - 1];
    const int64_t *st = t->stride;
    int64_t cs[NC_MAX_DIMS];
    if (t->stride[nd - 1] == 1 && t->stride[nd - 2] >= cols) {
        o->p = (const float *)t->data, o->trans = 0, o->ld = t->stride[nd - 2];
    } else if (t->stride[nd - 2] == 1 && t->stride[nd - 1] >= rows) {
        o->p = (const float *)t->data, o->trans = 1, o->ld = t->stride[nd - 1];
    } else {
        o->p = nc__contig_f32(t), o->trans = 0, o->ld = cols;
        if (!o->p) return 0;
        nc__contig_strides(nd, t->shape, cs);
        st = cs;
    }
    batch_strides(t, st, bd, o->bst);
    return 1;
}

static nc_status bmm_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *a = n->in[0], *b = n->in[1];
    batch_dims bd;
    if (!nc__expect_f32(n, 0) || !nc__expect_f32(n, 1)) return NC_ERR_DTYPE;
    if (a->ndim < 2 || b->ndim < 2 || a->shape[a->ndim - 1] != b->shape[b->ndim - 2] || !batch_shape(a, b, &bd)) {
        char sa[96], sb[96];
        nc__error(a->ctx, NC_ERR_SHAPE, "matmul: cannot multiply %s by %s (expected [..., M, K] @ [..., K, N])",
                  nc__shape_str(a->ndim, a->shape, sa, sizeof sa),
                  nc__shape_str(b->ndim, b->shape, sb, sizeof sb));
        return NC_ERR_SHAPE;
    }
    *dt = NC_F32;
    *nd = bd.nb + 2;
    memcpy(shape, bd.shape, sizeof(int64_t) * (size_t)bd.nb);
    shape[bd.nb] = a->shape[a->ndim - 2];
    shape[bd.nb + 1] = b->shape[b->ndim - 1];
    return NC_OK;
}

/* Small batch matrices run one per thread (a plain GEMM each), large ones
 * one after another with a threaded GEMM. Either way each output element is
 * computed by a single thread, so results do not depend on the thread count. */
typedef struct {
    const nc_kernels *k;
    const batch_dims *bd;
    int ta, tb;
    int64_t M, N, K, lda, ldb, ldc;
    const float *A, *B;
    const int64_t *sa, *sb, *sc; /* batch strides */
    float *C;
} bmm_job;

static void bmm_range(void *arg, int64_t begin, int64_t end) {
    const bmm_job *j = (const bmm_job *)arg;
    for (int64_t i = begin; i < end; i++)
        j->k->gemm_f32(j->ta, j->tb, j->M, j->N, j->K, 1.0f, j->A + batch_offset(j->bd, j->sa, i), j->lda,
                       j->B + batch_offset(j->bd, j->sb, i), j->ldb, 1.0f, j->C + batch_offset(j->bd, j->sc, i),
                       j->ldc);
}

/* C_i += op(A_i) @ op(B_i) for every batch index i. `split` = 0 when two
 * indices may write the same C block (a broadcast grad): keep them serial. */
static void bmm_run(nc_ctx *ctx, bmm_job *j, int split) {
    double work = (double)j->M * (double)j->N * (double)(j->K ? j->K : 1);
    if (split && j->bd->count > 1 && work < (double)(1 << 21) && ctx->n_threads > 1) {
        int64_t grain = (int64_t)((double)(1 << 18) / (work + 1.0)) + 1;
        nc__parallel_for(ctx, j->bd->count, grain, bmm_range, j);
        return;
    }
    for (int64_t i = 0; i < j->bd->count; i++)
        nc__gemm(ctx, j->ta, j->tb, j->M, j->N, j->K, 1.0f, j->A + batch_offset(j->bd, j->sa, i), j->lda,
                 j->B + batch_offset(j->bd, j->sb, i), j->ldb, 1.0f, j->C + batch_offset(j->bd, j->sc, i), j->ldc);
}

static int any_broadcast(const batch_dims *bd, const int64_t *st) {
    for (int i = 0; i < bd->nb; i++)
        if (st[i] == 0 && bd->shape[i] > 1) return 1;
    return 0;
}

static nc_status bmm_fwd(nc_node *n) {
    nc_tensor *a = n->in[0], *b = n->in[1], *o = n->out;
    batch_dims bd;
    batch_shape(a, b, &bd);
    bmm_operand A, B;
    if (!bmm_operand_of(a, &bd, &A) || !bmm_operand_of(b, &bd, &B)) return NC_ERR_OOM;
    int64_t M = o->shape[bd.nb], N = o->shape[bd.nb + 1], K = a->shape[a->ndim - 1];
    int64_t sc[NC_MAX_DIMS];
    batch_strides(o, o->stride, &bd, sc);
    memset(o->data, 0, (size_t)nc_numel(o) * sizeof(float));
    bmm_job j = {o->ctx->k, &bd, A.trans, B.trans, M, N, K, A.ld, B.ld, N, A.p, B.p, A.bst, B.bst, sc,
                 (float *)o->data};
    bmm_run(o->ctx, &j, 1);
    return NC_OK;
}

static void bmm_bwd(nc_node *n) {
    nc_tensor *a = n->in[0], *b = n->in[1], *o = n->out;
    nc_ctx *ctx = o->ctx;
    batch_dims bd;
    batch_shape(a, b, &bd);
    int64_t M = o->shape[bd.nb], N = o->shape[bd.nb + 1], K = a->shape[a->ndim - 1];
    const float *g = (const float *)o->grad->data;
    int64_t sg[NC_MAX_DIMS], cs[NC_MAX_DIMS], sgrad[NC_MAX_DIMS];
    nc__contig_strides(o->ndim, o->shape, cs);
    batch_strides(o, cs, &bd, sg);
    bmm_operand A, B;
    if (nc__needs_grad(a)) { /* dA_i (M x K) += dC_i @ B_i^T */
        nc_tensor *ga = nc__grad_buf(a);
        if (ga && bmm_operand_of(b, &bd, &B)) {
            nc__contig_strides(a->ndim, a->shape, cs);
            batch_strides(a, cs, &bd, sgrad);
            bmm_job j = {ctx->k, &bd, 0, !B.trans, M, K, N, N, B.ld, K, g, B.p, sg, B.bst, sgrad, (float *)ga->data};
            bmm_run(ctx, &j, !any_broadcast(&bd, sgrad));
        }
    }
    if (nc__needs_grad(b)) { /* dB_i (K x N) += A_i^T @ dC_i */
        nc_tensor *gb = nc__grad_buf(b);
        if (gb && bmm_operand_of(a, &bd, &A)) {
            nc__contig_strides(b->ndim, b->shape, cs);
            batch_strides(b, cs, &bd, sgrad);
            bmm_job j = {ctx->k, &bd, !A.trans, 0, K, N, M, A.ld, N, N, A.p, g, A.bst, sg, sgrad, (float *)gb->data};
            bmm_run(ctx, &j, !any_broadcast(&bd, sgrad));
        }
    }
}

const nc_op_def nc_op_bmm = {"bmm", 0, 0, bmm_infer, NULL, bmm_fwd, bmm_bwd};

nc_tensor *nc_matmul(nc_tensor *a, nc_tensor *b) {
    nc_tensor *in[2] = {a, b};
    return nc__apply(b && b->ndim > 2 ? &nc_op_bmm : &nc_op_matmul, in, 2, NULL);
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
