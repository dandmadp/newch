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
    int sh, sw, ph, pw, dh, dw; /* stride, padding, dilation */
    int64_t G;                  /* groups: C/G input and OC/G output channels each */
} conv_geom;

static int64_t out_size(int64_t in, int64_t k, int s, int p, int d) { return (in + 2 * p - d * (k - 1) - 1) / s + 1; }

/* params: i[0..3] = stride_h, stride_w, pad_h, pad_w; pools add i[4..5] = k_h, k_w;
 * conv adds i[4..5] = dilation_h, dilation_w and i[6] = groups */
static conv_geom geom(const nc_node *n, int64_t kh, int64_t kw, int64_t oc) {
    const nc_tensor *x = n->in[0];
    conv_geom g;
    g.N = x->shape[0], g.C = x->shape[1], g.H = x->shape[2], g.W = x->shape[3];
    g.sh = (int)n->p.i[0], g.sw = (int)n->p.i[1], g.ph = (int)n->p.i[2], g.pw = (int)n->p.i[3];
    g.dh = g.dw = 1;
    g.G = 1;
    g.KH = kh, g.KW = kw, g.OC = oc;
    g.OH = out_size(g.H, kh, g.sh, g.ph, 1);
    g.OW = out_size(g.W, kw, g.sw, g.pw, 1);
    return g;
}

static conv_geom conv_geom_of(const nc_node *n) {
    const nc_tensor *w = n->in[1];
    conv_geom g = geom(n, w->shape[2], w->shape[3], w->shape[0]);
    g.dh = (int)n->p.i[4], g.dw = (int)n->p.i[5];
    g.G = n->p.i[6];
    g.OH = out_size(g.H, g.KH, g.sh, g.ph, g.dh);
    g.OW = out_size(g.W, g.KW, g.sw, g.pw, g.dw);
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
    conv_geom g = geom(n, kh, kw, 0); /* kh, kw: the dilated extent for conv */
    if (g.OH < 1 || g.OW < 1) {
        nc__error(x->ctx, NC_ERR_SHAPE, "%s: kernel %lldx%lld does not fit input %lldx%lld with padding %d,%d",
                  n->op->name, (long long)kh, (long long)kw, (long long)g.H, (long long)g.W, g.ph, g.pw);
        return NC_ERR_SHAPE;
    }
    return NC_OK;
}

/* col [Cg*KH*KW, OH*OW] from Cg channels x [Cg, H, W] of one image */
static void im2col(const conv_geom *g, int64_t Cg, const float *x, float *col) {
    int64_t L = g->OH * g->OW;
    for (int64_t c = 0; c < Cg; c++)
        for (int64_t ki = 0; ki < g->KH; ki++)
            for (int64_t kj = 0; kj < g->KW; kj++) {
                float *row = col + ((c * g->KH + ki) * g->KW + kj) * L;
                for (int64_t oy = 0; oy < g->OH; oy++) {
                    int64_t iy = oy * g->sh - g->ph + ki * g->dh;
                    float *r = row + oy * g->OW;
                    if (iy < 0 || iy >= g->H) {
                        for (int64_t ox = 0; ox < g->OW; ox++) r[ox] = 0.0f;
                        continue;
                    }
                    const float *src = x + (c * g->H + iy) * g->W;
                    for (int64_t ox = 0; ox < g->OW; ox++) {
                        int64_t ix = ox * g->sw - g->pw + kj * g->dw;
                        r[ox] = (ix >= 0 && ix < g->W) ? src[ix] : 0.0f;
                    }
                }
            }
}

/* accumulate col back into dx [Cg, H, W] */
static void col2im(const conv_geom *g, int64_t Cg, const float *col, float *dx) {
    int64_t L = g->OH * g->OW;
    for (int64_t c = 0; c < Cg; c++)
        for (int64_t ki = 0; ki < g->KH; ki++)
            for (int64_t kj = 0; kj < g->KW; kj++) {
                const float *row = col + ((c * g->KH + ki) * g->KW + kj) * L;
                for (int64_t oy = 0; oy < g->OH; oy++) {
                    int64_t iy = oy * g->sh - g->ph + ki * g->dh;
                    if (iy < 0 || iy >= g->H) continue;
                    float *dst = dx + (c * g->H + iy) * g->W;
                    const float *r = row + oy * g->OW;
                    for (int64_t ox = 0; ox < g->OW; ox++) {
                        int64_t ix = ox * g->sw - g->pw + kj * g->dw;
                        if (ix >= 0 && ix < g->W) dst[ix] += r[ox];
                    }
                }
            }
}

/* ------------------------------------------------------------------------ */
/* conv2d: x [N, C, H, W], w [OC, C/G, KH, KW], b [OC] -> [N, OC, OH, OW]    */
/* Groups run one im2col + GEMM each; depthwise (one input and one output    */
/* channel per group) skips im2col and convolves each plane directly, one    */
/* plane per thread.                                                         */
/* ------------------------------------------------------------------------ */

static nc_status conv_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    for (int i = 0; i < n->n_in; i++)
        if (!nc__expect_f32(n, i)) return NC_ERR_DTYPE;
    int64_t G = n->p.i[6];
    if (G < 1 || n->p.i[4] < 1 || n->p.i[5] < 1) {
        nc__error(x->ctx, NC_ERR_ARG, "conv2d: dilation and groups must be >= 1");
        return NC_ERR_ARG;
    }
    if (w->ndim != 4 || x->ndim != 4 || x->shape[1] != w->shape[1] * G || w->shape[0] % G != 0 ||
        (b && (b->ndim != 1 || b->shape[0] != w->shape[0]))) {
        char sx[96], sw[96];
        nc__error(x->ctx, NC_ERR_SHAPE,
                  "conv2d: expected x [N, C, H, W], weight [OC, C/groups, KH, KW] with OC divisible by groups (%lld), "
                  "bias [OC]; got %s and %s",
                  (long long)G, nc__shape_str(x->ndim, x->shape, sx, sizeof sx),
                  nc__shape_str(w->ndim, w->shape, sw, sizeof sw));
        return NC_ERR_SHAPE;
    }
    nc_status st = check_window(n, n->p.i[4] * (w->shape[2] - 1) + 1, n->p.i[5] * (w->shape[3] - 1) + 1);
    if (st != NC_OK) return st;
    conv_geom g = conv_geom_of(n);
    *dt = NC_F32;
    *nd = 4;
    shape[0] = g.N, shape[1] = g.OC, shape[2] = g.OH, shape[3] = g.OW;
    return NC_OK;
}

static int is_depthwise(const conv_geom *g) { return g->G == g->C && g->OC == g->C; }

typedef struct {
    const conv_geom *g;
    const nc_kernels *k;
    const float *X, *W, *B, *G; /* G: output grad (backward) */
    float *Y, *dX, *dW, *dB;
} dw_job;

/* Output columns [*lo, *hi) whose input column ox*s + off lies inside [0, in): the loops below
 * then need no bounds checks, and stride 1 rows go through the SIMD axpy kernel. */
static void col_range(int64_t out, int64_t in, int s, int64_t off, int64_t *lo, int64_t *hi) {
    *lo = off >= 0 ? 0 : (-off + s - 1) / s;
    *hi = in - 1 - off < 0 ? 0 : (in - 1 - off) / s + 1;
    if (*hi > out) *hi = out;
}

/* depthwise forward: planes [begin, end) of N*C */
static void dw_fwd_range(void *arg, int64_t begin, int64_t end) {
    const dw_job *j = (const dw_job *)arg;
    const conv_geom *g = j->g;
    for (int64_t plane = begin; plane < end; plane++) {
        int64_t c = plane % g->C;
        const float *xp = j->X + plane * g->H * g->W, *wk = j->W + c * g->KH * g->KW;
        float *yp = j->Y + plane * g->OH * g->OW;
        float bias = j->B ? j->B[c] : 0.0f;
        for (int64_t oy = 0; oy < g->OH; oy++) {
            float *yr = yp + oy * g->OW;
            for (int64_t ox = 0; ox < g->OW; ox++) yr[ox] = bias;
            for (int64_t ki = 0; ki < g->KH; ki++) {
                int64_t iy = oy * g->sh - g->ph + ki * g->dh;
                if (iy < 0 || iy >= g->H) continue;
                const float *xr = xp + iy * g->W;
                for (int64_t kj = 0; kj < g->KW; kj++) {
                    int64_t off = kj * g->dw - g->pw, lo, hi;
                    col_range(g->OW, g->W, g->sw, off, &lo, &hi);
                    float wv = wk[ki * g->KW + kj];
                    if (lo >= hi) continue;
                    if (g->sw == 1) j->k->axpy_f32(hi - lo, wv, xr + lo + off, yr + lo);
                    else
                        for (int64_t ox = lo; ox < hi; ox++) yr[ox] += wv * xr[ox * g->sw + off];
                }
            }
        }
    }
}

/* depthwise input grad: planes [begin, end) of N*C (each dx plane has one writer) */
static void dw_dx_range(void *arg, int64_t begin, int64_t end) {
    const dw_job *j = (const dw_job *)arg;
    const conv_geom *g = j->g;
    for (int64_t plane = begin; plane < end; plane++) {
        int64_t c = plane % g->C;
        const float *gp = j->G + plane * g->OH * g->OW, *wk = j->W + c * g->KH * g->KW;
        float *dxp = j->dX + plane * g->H * g->W;
        for (int64_t oy = 0; oy < g->OH; oy++) {
            const float *gr = gp + oy * g->OW;
            for (int64_t ki = 0; ki < g->KH; ki++) {
                int64_t iy = oy * g->sh - g->ph + ki * g->dh;
                if (iy < 0 || iy >= g->H) continue;
                float *dxr = dxp + iy * g->W;
                for (int64_t kj = 0; kj < g->KW; kj++) {
                    int64_t off = kj * g->dw - g->pw, lo, hi;
                    col_range(g->OW, g->W, g->sw, off, &lo, &hi);
                    float wv = wk[ki * g->KW + kj];
                    if (lo >= hi) continue;
                    if (g->sw == 1) j->k->axpy_f32(hi - lo, wv, gr + lo, dxr + lo + off);
                    else
                        for (int64_t ox = lo; ox < hi; ox++) dxr[ox * g->sw + off] += wv * gr[ox];
                }
            }
        }
    }
}

/* depthwise weight and bias grads: channels [begin, end), summed over the batch in order */
static void dw_dw_range(void *arg, int64_t begin, int64_t end) {
    const dw_job *j = (const dw_job *)arg;
    const conv_geom *g = j->g;
    for (int64_t c = begin; c < end; c++) {
        float *dwk = j->dW ? j->dW + c * g->KH * g->KW : NULL;
        for (int64_t s = 0; s < g->N; s++) {
            int64_t plane = s * g->C + c;
            const float *xp = j->X + plane * g->H * g->W, *gp = j->G + plane * g->OH * g->OW;
            if (j->dB) j->dB[c] += j->k->sum_f32(g->OH * g->OW, gp);
            if (!dwk) continue;
            for (int64_t ki = 0; ki < g->KH; ki++)
                for (int64_t kj = 0; kj < g->KW; kj++) {
                    int64_t off = kj * g->dw - g->pw, lo, hi;
                    col_range(g->OW, g->W, g->sw, off, &lo, &hi);
                    float acc = 0.0f;
                    for (int64_t oy = 0; oy < g->OH && lo < hi; oy++) {
                        int64_t iy = oy * g->sh - g->ph + ki * g->dh;
                        if (iy < 0 || iy >= g->H) continue;
                        const float *xr = xp + iy * g->W + off, *gr = gp + oy * g->OW;
                        if (g->sw == 1) { /* SIMD: products into a small buffer, then a vector sum */
                            float tmp[256];
                            for (int64_t ox = lo; ox < hi; ox += 256) {
                                int64_t m = hi - ox < 256 ? hi - ox : 256;
                                j->k->mul_f32(m, xr + ox, gr + ox, tmp);
                                acc += j->k->sum_f32(m, tmp);
                            }
                        } else
                            for (int64_t ox = lo; ox < hi; ox++) acc += xr[ox * g->sw] * gr[ox];
                    }
                    dwk[ki * g->KW + kj] += acc;
                }
        }
    }
}

static int64_t dw_grain(const conv_geom *g) {
    int64_t work = g->OH * g->OW * g->KH * g->KW;
    return work >= 16384 ? 1 : 16384 / (work ? work : 1);
}

static nc_status conv_fwd(nc_node *n) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    nc_ctx *ctx = x->ctx;
    conv_geom g = conv_geom_of(n);
    int64_t Cg = g.C / g.G, OCg = g.OC / g.G, CKK = Cg * g.KH * g.KW, L = g.OH * g.OW;
    const float *X = nc__contig_f32(x), *Wt = nc__contig_f32(w), *B = b ? nc__contig_f32(b) : NULL;
    if (!X || !Wt || (b && !B)) return NC_ERR_OOM;
    float *Y = (float *)n->out->data;
    if (is_depthwise(&g)) {
        dw_job j = {&g, ctx->k, X, Wt, B, NULL, Y, NULL, NULL, NULL};
        nc__parallel_for(ctx, g.N * g.C, dw_grain(&g), dw_fwd_range, &j);
        return NC_OK;
    }
    float *col = (float *)nc__arena_alloc(ctx, (size_t)(CKK * L) * sizeof(float));
    if (!col) return NC_ERR_OOM;
    for (int64_t s = 0; s < g.N; s++) {
        for (int64_t gi = 0; gi < g.G; gi++) {
            float *y = Y + (s * g.OC + gi * OCg) * L;
            im2col(&g, Cg, X + (s * g.C + gi * Cg) * g.H * g.W, col);
            if (B)
                for (int64_t o = 0; o < OCg; o++)
                    for (int64_t l = 0; l < L; l++) y[o * L + l] = B[gi * OCg + o];
            nc__gemm(ctx, 0, 0, OCg, L, CKK, 1.0f, Wt + gi * OCg * CKK, CKK, col, L, B ? 1.0f : 0.0f, y, L);
        }
    }
    return NC_OK;
}

static void conv_bwd(nc_node *n) {
    nc_tensor *x = n->in[0], *w = n->in[1], *b = n->n_in > 2 ? n->in[2] : NULL;
    nc_ctx *ctx = x->ctx;
    conv_geom g = conv_geom_of(n);
    int64_t Cg = g.C / g.G, OCg = g.OC / g.G, CKK = Cg * g.KH * g.KW, L = g.OH * g.OW;
    const float *G = (const float *)n->out->grad->data;
    const float *X = nc__contig_f32(x), *Wt = nc__contig_f32(w);
    if (!X || !Wt) return;
    nc_tensor *gx = nc__needs_grad(x) ? nc__grad_buf(x) : NULL;
    nc_tensor *gw = nc__needs_grad(w) ? nc__grad_buf(w) : NULL;
    nc_tensor *gb = b && nc__needs_grad(b) ? nc__grad_buf(b) : NULL;
    if (is_depthwise(&g)) {
        dw_job j = {&g, ctx->k, X, Wt, NULL, G, NULL, gx ? (float *)gx->data : NULL, gw ? (float *)gw->data : NULL,
                    gb ? (float *)gb->data : NULL};
        if (gx) nc__parallel_for(ctx, g.N * g.C, dw_grain(&g), dw_dx_range, &j);
        if (gw || gb) nc__parallel_for(ctx, g.C, 1, dw_dw_range, &j);
        return;
    }
    float *col = (float *)nc__arena_alloc(ctx, (size_t)(CKK * L) * sizeof(float));
    if (!col) return;
    for (int64_t s = 0; s < g.N; s++) {
        for (int64_t gi = 0; gi < g.G; gi++) {
            const float *gs = G + (s * g.OC + gi * OCg) * L;
            const float *xs = X + (s * g.C + gi * Cg) * g.H * g.W;
            if (gw) {
                /* dW_g [OCg, CKK] += dY_g [OCg, L] @ col^T */
                im2col(&g, Cg, xs, col);
                nc__gemm(ctx, 0, 1, OCg, CKK, L, 1.0f, gs, L, col, L, 1.0f, (float *)gw->data + gi * OCg * CKK, CKK);
            }
            if (gx) {
                /* dcol [CKK, L] = W_g^T @ dY_g, then scatter back into dx */
                nc__gemm(ctx, 1, 0, CKK, L, OCg, 1.0f, Wt + gi * OCg * CKK, CKK, gs, L, 0.0f, col, L);
                col2im(&g, Cg, col, (float *)gx->data + (s * g.C + gi * Cg) * g.H * g.W);
            }
        }
        if (gb)
            for (int64_t o = 0; o < g.OC; o++)
                ((float *)gb->data)[o] += ctx->k->sum_f32(L, G + (s * g.OC + o) * L);
    }
}

const nc_op_def nc_op_conv2d = {"conv2d", 0, 0, conv_infer, NULL, conv_fwd, conv_bwd};

nc_tensor *nc_conv2d_ex(nc_tensor *x, nc_tensor *weight, nc_tensor *bias, int stride_h, int stride_w, int pad_h,
                        int pad_w, int dilation_h, int dilation_w, int groups) {
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = stride_h, p.i[1] = stride_w, p.i[2] = pad_h, p.i[3] = pad_w;
    p.i[4] = dilation_h, p.i[5] = dilation_w, p.i[6] = groups;
    nc_tensor *in[3] = {x, weight, bias};
    return nc__apply(&nc_op_conv2d, in, bias ? 3 : 2, &p);
}

nc_tensor *nc_conv2d(nc_tensor *x, nc_tensor *weight, nc_tensor *bias, int stride_h, int stride_w, int pad_h,
                     int pad_w) {
    return nc_conv2d_ex(x, weight, bias, stride_h, stride_w, pad_h, pad_w, 1, 1, 1);
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
