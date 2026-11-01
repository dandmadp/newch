/* 2-D convolution (im2col + GEMM) and pooling on NCHW tensors.
 *
 * Convolution keeps one im2col buffer per call (C*KH*KW x OH*OW floats) and
 * rebuilds it in backward instead of storing it per sample, trading a little
 * compute for a much smaller activation footprint. */
#include "../core/nc_internal.h"

#include <math.h>
#include <stdio.h>

typedef struct {
    int64_t N, C, H, W, OC, KH, KW, OH, OW;
    int sh, sw, ph, pw;
} conv_geom;

static int64_t out_size(int64_t in, int64_t k, int s, int p) { return (in + 2 * p - k) / s + 1; }

/* params: i[0..3] = stride_h, stride_w, pad_h, pad_w; pools add i[4..5] = k_h, k_w */
static conv_geom geom(const nc_node *n, int64_t kh, int64_t kw, int64_t oc) {
    const nc_tensor *x = n->in[0];
    conv_geom g;
    g.N = x->shape[0], g.C = x->shape[1], g.H = x->shape[2], g.W = x->shape[3];
    g.sh = (int)n->p.i[0], g.sw = (int)n->p.i[1], g.ph = (int)n->p.i[2], g.pw = (int)n->p.i[3];
    g.KH = kh, g.KW = kw, g.OC = oc;
    g.OH = out_size(g.H, kh, g.sh, g.ph);
    g.OW = out_size(g.W, kw, g.sw, g.pw);
    return g;
}

static nc_status check_window(nc_node *n, int64_t kh, int64_t kw) {
    nc_tensor *x = n->in[0];
    if (n->p.i[0] < 1 || n->p.i[1] < 1 || n->p.i[2] < 0 || n->p.i[3] < 0 || kh < 1 || kw < 1) {
        nc__error(x->ctx, NC_ERR_ARG, "%s: stride must be >= 1, padding >= 0, kernel >= 1", n->op->name);
        return NC_ERR_ARG;
    }
    if (x->ndim != 4) {
        char sx[96];
        nc__error(x->ctx, NC_ERR_SHAPE, "%s: expected input [N, C, H, W], got %s", n->op->name,
                  nc__shape_str(x->ndim, x->shape, sx, sizeof sx));
        return NC_ERR_SHAPE;
    }
    conv_geom g = geom(n, kh, kw, 0);
    if (g.OH < 1 || g.OW < 1) {
        nc__error(x->ctx, NC_ERR_SHAPE, "%s: kernel %lldx%lld does not fit input %lldx%lld with padding %d,%d",
                  n->op->name, (long long)kh, (long long)kw, (long long)g.H, (long long)g.W, g.ph, g.pw);
        return NC_ERR_SHAPE;
    }
    return NC_OK;
}

/* col [C*KH*KW, OH*OW] from one image x [C, H, W] */
static void im2col(const conv_geom *g, const float *x, float *col) {
    int64_t L = g->OH * g->OW;
    for (int64_t c = 0; c < g->C; c++)
        for (int64_t ki = 0; ki < g->KH; ki++)
            for (int64_t kj = 0; kj < g->KW; kj++) {
                float *row = col + ((c * g->KH + ki) * g->KW + kj) * L;
                for (int64_t oy = 0; oy < g->OH; oy++) {
                    int64_t iy = oy * g->sh - g->ph + ki;
                    float *r = row + oy * g->OW;
                    if (iy < 0 || iy >= g->H) {
                        for (int64_t ox = 0; ox < g->OW; ox++) r[ox] = 0.0f;
                        continue;
                    }
                    const float *src = x + (c * g->H + iy) * g->W;
                    for (int64_t ox = 0; ox < g->OW; ox++) {
                        int64_t ix = ox * g->sw - g->pw + kj;
                        r[ox] = (ix >= 0 && ix < g->W) ? src[ix] : 0.0f;
                    }
                }
            }
}

/* accumulate col back into dx [C, H, W] */
static void col2im(const conv_geom *g, const float *col, float *dx) {
    int64_t L = g->OH * g->OW;
    for (int64_t c = 0; c < g->C; c++)
        for (int64_t ki = 0; ki < g->KH; ki++)
            for (int64_t kj = 0; kj < g->KW; kj++) {
                const float *row = col + ((c * g->KH + ki) * g->KW + kj) * L;
                for (int64_t oy = 0; oy < g->OH; oy++) {
                    int64_t iy = oy * g->sh - g->ph + ki;
                    if (iy < 0 || iy >= g->H) continue;
                    float *dst = dx + (c * g->H + iy) * g->W;
                    const float *r = row + oy * g->OW;
                    for (int64_t ox = 0; ox < g->OW; ox++) {
                        int64_t ix = ox * g->sw - g->pw + kj;
                        if (ix >= 0 && ix < g->W) dst[ix] += r[ox];
                    }
                }
            }
}

/* ------------------------------------------------------------------------ */
/* conv2d: x [N, C, H, W], w [OC, C, KH, KW], b [OC] -> [N, OC, OH, OW]      */
/* ------------------------------------------------------------------------ */

static nc_status conv_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    for (int i = 0; i < n->n_in; i++)
        if (!nc__expect_f32(n, i)) return NC_ERR_DTYPE;
    if (w->ndim != 4 || x->ndim != 4 || x->shape[1] != w->shape[1] || (b && (b->ndim != 1 || b->shape[0] != w->shape[0]))) {
        char sx[96], sw[96];
        nc__error(x->ctx, NC_ERR_SHAPE, "conv2d: expected x [N, C, H, W], weight [OC, C, KH, KW], bias [OC]; got %s and %s",
                  nc__shape_str(x->ndim, x->shape, sx, sizeof sx), nc__shape_str(w->ndim, w->shape, sw, sizeof sw));
        return NC_ERR_SHAPE;
    }
    nc_status st = check_window(n, w->shape[2], w->shape[3]);
    if (st != NC_OK) return st;
    conv_geom g = geom(n, w->shape[2], w->shape[3], w->shape[0]);
    *dt = NC_F32;
    *nd = 4;
    shape[0] = g.N, shape[1] = g.OC, shape[2] = g.OH, shape[3] = g.OW;
    return NC_OK;
}

static nc_status conv_fwd(nc_node *n) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    nc_ctx *ctx = x->ctx;
    conv_geom g = geom(n, w->shape[2], w->shape[3], w->shape[0]);
    int64_t CKK = g.C * g.KH * g.KW, L = g.OH * g.OW;
    const float *X = nc__contig_f32(x), *Wt = nc__contig_f32(w), *B = b ? nc__contig_f32(b) : NULL;
    float *col = (float *)nc__arena_alloc(ctx, (size_t)(CKK * L) * sizeof(float));
    if (!X || !Wt || !col || (b && !B)) return NC_ERR_OOM;
    float *Y = (float *)n->out->data;
    for (int64_t s = 0; s < g.N; s++) {
        float *y = Y + s * g.OC * L;
        im2col(&g, X + s * g.C * g.H * g.W, col);
        if (B)
            for (int64_t o = 0; o < g.OC; o++)
                for (int64_t l = 0; l < L; l++) y[o * L + l] = B[o];
        nc__gemm(ctx, 0, 0, g.OC, L, CKK, 1.0f, Wt, CKK, col, L, B ? 1.0f : 0.0f, y, L);
    }
    return NC_OK;
}

static void conv_bwd(nc_node *n) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    nc_ctx *ctx = x->ctx;
    conv_geom g = geom(n, w->shape[2], w->shape[3], w->shape[0]);
    int64_t CKK = g.C * g.KH * g.KW, L = g.OH * g.OW;
    const float *G = (const float *)n->out->grad->data;
    const float *X = nc__contig_f32(x), *Wt = nc__contig_f32(w);
    float *col = (float *)nc__arena_alloc(ctx, (size_t)(CKK * L) * sizeof(float));
    if (!X || !Wt || !col) return;
    nc_tensor *gx = nc__needs_grad(x) ? nc__grad_buf(x) : NULL;
    nc_tensor *gw = nc__needs_grad(w) ? nc__grad_buf(w) : NULL;
    nc_tensor *gb = b && nc__needs_grad(b) ? nc__grad_buf(b) : NULL;
    for (int64_t s = 0; s < g.N; s++) {
        const float *gs = G + s * g.OC * L;
        if (gw) {
            /* dW [OC, CKK] += dY [OC, L] @ col^T */
            im2col(&g, X + s * g.C * g.H * g.W, col);
            nc__gemm(ctx, 0, 1, g.OC, CKK, L, 1.0f, gs, L, col, L, 1.0f, (float *)gw->data, CKK);
        }
        if (gx) {
            /* dcol [CKK, L] = W^T @ dY, then scatter back into dx */
            nc__gemm(ctx, 1, 0, CKK, L, g.OC, 1.0f, Wt, CKK, gs, L, 0.0f, col, L);
            col2im(&g, col, (float *)gx->data + s * g.C * g.H * g.W);
        }
        if (gb)
            for (int64_t o = 0; o < g.OC; o++) ((float *)gb->data)[o] += ctx->k->sum_f32(L, gs + o * L);
    }
}

const nc_op_def nc_op_conv2d = {"conv2d", 0, 0, conv_infer, NULL, conv_fwd, conv_bwd};

nc_tensor *nc_conv2d(nc_tensor *x, nc_tensor *weight, nc_tensor *bias, int stride_h, int stride_w, int pad_h,
                     int pad_w) {
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = stride_h, p.i[1] = stride_w, p.i[2] = pad_h, p.i[3] = pad_w;
    nc_tensor *in[3] = {x, weight, bias};
    return nc__apply(&nc_op_conv2d, in, bias ? 3 : 2, &p);
}

/* ------------------------------------------------------------------------ */
/* Pooling                                                                   */
/* ------------------------------------------------------------------------ */

enum { POOL_MAX, POOL_AVG };

static nc_status pool_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    if (!nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    int64_t kh = n->p.i[4], kw = n->p.i[5];
    nc_status st = check_window(n, kh, kw);
    if (st != NC_OK) return st;
    if (n->p.i[2] * 2 > kh || n->p.i[3] * 2 > kw) {
        nc__error(n->in[0]->ctx, NC_ERR_ARG, "%s: padding must be at most half the kernel size", n->op->name);
        return NC_ERR_ARG;
    }
    conv_geom g = geom(n, kh, kw, 0);
    *dt = NC_F32;
    *nd = 4;
    shape[0] = g.N, shape[1] = g.C, shape[2] = g.OH, shape[3] = g.OW;
    return NC_OK;
}

static nc_status pool_fwd(nc_node *n) {
    nc_tensor *x = n->in[0];
    conv_geom g = geom(n, n->p.i[4], n->p.i[5], 0);
    const float *X = nc__contig_f32(x);
    float *Y = (float *)n->out->data;
    int is_max = n->op->kind == POOL_MAX;
    int32_t *arg = NULL;
    if (!X) return NC_ERR_OOM;
    if (is_max) {
        arg = (int32_t *)nc__arena_alloc(x->ctx, (size_t)nc_numel(n->out) * sizeof(int32_t));
        if (!arg) return NC_ERR_OOM;
        n->saved = arg;
    }
    float inv = 1.0f / (float)(g.KH * g.KW);
    for (int64_t plane = 0; plane < g.N * g.C; plane++) {
        const float *xp = X + plane * g.H * g.W;
        for (int64_t oy = 0; oy < g.OH; oy++)
            for (int64_t ox = 0; ox < g.OW; ox++) {
                float best = -INFINITY, acc = 0.0f;
                int32_t bi = -1;
                for (int64_t ki = 0; ki < g.KH; ki++) {
                    int64_t iy = oy * g.sh - g.ph + ki;
                    if (iy < 0 || iy >= g.H) continue;
                    for (int64_t kj = 0; kj < g.KW; kj++) {
                        int64_t ix = ox * g.sw - g.pw + kj;
                        if (ix < 0 || ix >= g.W) continue;
                        float v = xp[iy * g.W + ix];
                        acc += v;
                        if (v > best || bi < 0) best = v, bi = (int32_t)(iy * g.W + ix);
                    }
                }
                int64_t o = (plane * g.OH + oy) * g.OW + ox;
                if (is_max) {
                    Y[o] = best;
                    arg[o] = bi;
                } else {
                    Y[o] = acc * inv; /* count_include_pad=True */
                }
            }
    }
    return NC_OK;
}

static void pool_bwd(nc_node *n) {
    nc_tensor *x = n->in[0];
    nc_tensor *gt = nc__grad_buf(x);
    if (!gt) return;
    conv_geom g = geom(n, n->p.i[4], n->p.i[5], 0);
    const float *G = (const float *)n->out->grad->data;
    float *gx = (float *)gt->data;
    if (n->op->kind == POOL_MAX) {
        const int32_t *arg = (const int32_t *)n->saved;
        int64_t per_plane = g.OH * g.OW;
        for (int64_t o = 0, total = nc_numel(n->out); o < total; o++)
            gx[(o / per_plane) * g.H * g.W + arg[o]] += G[o];
        return;
    }
    float inv = 1.0f / (float)(g.KH * g.KW);
    for (int64_t plane = 0; plane < g.N * g.C; plane++) {
        float *gp = gx + plane * g.H * g.W;
        for (int64_t oy = 0; oy < g.OH; oy++)
            for (int64_t ox = 0; ox < g.OW; ox++) {
                float v = G[(plane * g.OH + oy) * g.OW + ox] * inv;
                for (int64_t ki = 0; ki < g.KH; ki++) {
                    int64_t iy = oy * g.sh - g.ph + ki;
                    if (iy < 0 || iy >= g.H) continue;
                    for (int64_t kj = 0; kj < g.KW; kj++) {
                        int64_t ix = ox * g.sw - g.pw + kj;
                        if (ix >= 0 && ix < g.W) gp[iy * g.W + ix] += v;
                    }
                }
            }
    }
}

const nc_op_def nc_op_max_pool2d = {"max_pool2d", 0, POOL_MAX, pool_infer, NULL, pool_fwd, pool_bwd};
const nc_op_def nc_op_avg_pool2d = {"avg_pool2d", 0, POOL_AVG, pool_infer, NULL, pool_fwd, pool_bwd};

static nc_tensor *pool(const nc_op_def *op, nc_tensor *x, int kh, int kw, int sh, int sw, int ph, int pw) {
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = sh, p.i[1] = sw, p.i[2] = ph, p.i[3] = pw, p.i[4] = kh, p.i[5] = kw;
    return nc__apply(op, &x, 1, &p);
}

nc_tensor *nc_max_pool2d(nc_tensor *x, int k_h, int k_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    return pool(&nc_op_max_pool2d, x, k_h, k_w, stride_h, stride_w, pad_h, pad_w);
}

nc_tensor *nc_avg_pool2d(nc_tensor *x, int k_h, int k_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    return pool(&nc_op_avg_pool2d, x, k_h, k_w, stride_h, stride_w, pad_h, pad_w);
}
