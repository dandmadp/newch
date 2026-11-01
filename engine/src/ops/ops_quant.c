/* int8 dynamic quantization (inference only).
 *
 * Weights: symmetric per-output-row int8, w ~= q * scale[row]. 4x smaller.
 * Activations: quantized per row at runtime, so no calibration data is
 * needed. Products accumulate exactly in int32 before one rescale. */
#include "../core/nc_internal.h"

#include <math.h>
#include <stdio.h>

static void quantize_row(int64_t n, const float *x, int8_t *q, float *scale) {
    float amax = 0.0f;
    for (int64_t i = 0; i < n; i++) amax = fmaxf(amax, fabsf(x[i]));
    float s = amax > 0.0f ? amax / 127.0f : 1.0f;
    float inv = 1.0f / s;
    for (int64_t i = 0; i < n; i++) {
        float v = nearbyintf(x[i] * inv);
        q[i] = (int8_t)(v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
    }
    *scale = s;
}

nc_tensor *nc_quantize_rows(nc_tensor *w, nc_tensor **scale) {
    if (!w) return NULL;
    nc_ctx *ctx = w->ctx;
    if (w->dtype != NC_F32 || w->ndim != 2) {
        nc__error(ctx, NC_ERR_SHAPE, "quantize_rows: expected a 2-D f32 tensor");
        return NULL;
    }
    const float *W = nc__contig_f32(w);
    nc_tensor *q = nc_empty(ctx, NC_I8, 2, w->shape);
    nc_tensor *s = nc_empty(ctx, NC_F32, 1, w->shape);
    if (!W || !q || !s) return NULL;
    int64_t rows = w->shape[0], cols = w->shape[1];
    for (int64_t r = 0; r < rows; r++)
        quantize_row(cols, W + r * cols, (int8_t *)q->data + r * cols, (float *)s->data + r);
    if (scale) *scale = s;
    return q;
}

static nc_status qlinear_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0], *wq = n->in[1], *ws = n->in[2], *b = n->n_in > 3 ? n->in[3] : NULL;
    if (x->dtype != NC_F32 || wq->dtype != NC_I8 || ws->dtype != NC_F32 || (b && b->dtype != NC_F32)) {
        nc__error(x->ctx, NC_ERR_DTYPE, "linear_int8: expected x f32, weight i8, scale f32, bias f32");
        return NC_ERR_DTYPE;
    }
    if (x->ndim < 1 || wq->ndim != 2 || x->shape[x->ndim - 1] != wq->shape[1] || ws->ndim != 1 ||
        ws->shape[0] != wq->shape[0] || (b && (b->ndim != 1 || b->shape[0] != wq->shape[0]))) {
        char sx[96], sw[96];
        nc__error(x->ctx, NC_ERR_SHAPE, "linear_int8: expected x [..., in], weight [out, in], scale/bias [out]; got %s, %s",
                  nc__shape_str(x->ndim, x->shape, sx, sizeof sx), nc__shape_str(wq->ndim, wq->shape, sw, sizeof sw));
        return NC_ERR_SHAPE;
    }
    *dt = NC_F32;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    shape[x->ndim - 1] = wq->shape[0];
    return NC_OK;
}

typedef struct {
    const nc_kernels *k;
    const float *X, *ws, *b;
    const int8_t *Wq;
    int8_t *Xq;
    float *xs, *Y;
    int32_t *acc;
    int64_t in, out;
} qjob;

static void qlinear_rows(void *arg, int64_t r0, int64_t r1) {
    const qjob *j = (const qjob *)arg;
    for (int64_t r = r0; r < r1; r++) quantize_row(j->in, j->X + r * j->in, j->Xq + r * j->in, j->xs + r);
    j->k->gemm_s8_nt(r1 - r0, j->out, j->in, j->Xq + r0 * j->in, j->in, j->Wq, j->in, j->acc + r0 * j->out, j->out);
    for (int64_t r = r0; r < r1; r++) {
        const int32_t *a = j->acc + r * j->out;
        float *y = j->Y + r * j->out;
        for (int64_t o = 0; o < j->out; o++) y[o] = (float)a[o] * j->xs[r] * j->ws[o] + (j->b ? j->b[o] : 0.0f);
    }
}

static nc_status qlinear_fwd(nc_node *n) {
    nc_tensor *x = n->in[0], *wq = n->in[1];
    nc_ctx *ctx = x->ctx;
    qjob j;
    j.k = ctx->k;
    j.in = wq->shape[1];
    j.out = wq->shape[0];
    int64_t M = j.in ? nc_numel(x) / j.in : 0;
    j.X = nc__contig_f32(x);
    j.ws = nc__contig_f32(n->in[2]);
    j.b = n->n_in > 3 ? nc__contig_f32(n->in[3]) : NULL;
    j.Wq = (const int8_t *)wq->data;
    if (!nc_is_contiguous(wq)) {
        nc__error(ctx, NC_ERR_ARG, "linear_int8: weight must be contiguous");
        return NC_ERR_ARG;
    }
    j.Xq = (int8_t *)nc__arena_alloc(ctx, (size_t)(M * j.in) + 1);
    j.xs = (float *)nc__arena_alloc(ctx, (size_t)(M ? M : 1) * sizeof(float));
    j.acc = (int32_t *)nc__arena_alloc(ctx, (size_t)(M * j.out + 1) * sizeof(int32_t));
    j.Y = (float *)n->out->data;
    if (!j.X || !j.ws || (n->n_in > 3 && !j.b) || !j.Xq || !j.xs || !j.acc) return NC_ERR_OOM;
    int64_t grain = (int64_t)(65536 / (double)((j.in ? j.in : 1) * (j.out ? j.out : 1))) + 1;
    nc__parallel_for(ctx, M, grain, qlinear_rows, &j);
    return NC_OK;
}

const nc_op_def nc_op_linear_int8 = {"linear_int8", NC_OPF_NONDIFF, 0, qlinear_infer, NULL, qlinear_fwd, NULL};

nc_tensor *nc_linear_int8(nc_tensor *x, nc_tensor *wq, nc_tensor *w_scale, nc_tensor *bias) {
    nc_tensor *in[4] = {x, wq, w_scale, bias};
    return nc__apply(&nc_op_linear_int8, in, bias ? 4 : 3, NULL);
}
