/* Neural-network functional ops: softmax family and losses. */
#include "../core/nc_internal.h"

#include <math.h>

enum { SM_LOG, SM_PLAIN };

static nc_status softmax_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0];
    if (!nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    if (x->ndim < 1) {
        nc__error(x->ctx, NC_ERR_SHAPE, "%s: input must have at least 1 dimension", n->op->name);
        return NC_ERR_SHAPE;
    }
    *dt = NC_F32;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    return NC_OK;
}

/* Numerically stable softmax / log-softmax of one row (y may not alias x). */
static void softmax_row(const nc_kernels *k, int64_t C, const float *x, float *y, int log_form) {
    float m = x[0];
    for (int64_t c = 1; c < C; c++)
        if (x[c] > m) m = x[c];
    for (int64_t c = 0; c < C; c++) y[c] = x[c] - m;
    k->exp_f32(C, y, y);
    float s = k->sum_f32(C, y);
    if (log_form) {
        float ls = logf(s);
        for (int64_t c = 0; c < C; c++) y[c] = x[c] - m - ls;
    } else {
        k->scale_f32(C, 1.0f / s, y, y);
    }
}

static nc_status softmax_fwd(nc_node *n) {
    const float *x = nc__contig_f32(n->in[0]);
    if (!x) return NC_ERR_OOM;
    float *y = (float *)n->out->data;
    int64_t C = n->out->shape[n->out->ndim - 1];
    int64_t rows = C ? nc_numel(n->out) / C : 0;
    const nc_kernels *k = n->out->ctx->k;
    for (int64_t r = 0; r < rows; r++) softmax_row(k, C, x + r * C, y + r * C, n->op->kind == SM_LOG);
    return NC_OK;
}

static void softmax_bwd(nc_node *n) {
    nc_tensor *gt = nc__grad_buf(n->in[0]);
    if (!gt) return;
    const float *y = (const float *)n->out->data;
    const float *g = (const float *)n->out->grad->data;
    float *gi = (float *)gt->data;
    int64_t C = n->out->shape[n->out->ndim - 1];
    int64_t rows = C ? nc_numel(n->out) / C : 0;
    for (int64_t r = 0; r < rows; r++) {
        const float *yr = y + r * C, *gr = g + r * C;
        float *o = gi + r * C;
        if (n->op->kind == SM_LOG) {
            /* d logsoftmax: g - softmax * sum(g) */
            float sg = 0.0f;
            for (int64_t c = 0; c < C; c++) sg += gr[c];
            for (int64_t c = 0; c < C; c++) o[c] += gr[c] - expf(yr[c]) * sg;
        } else {
            /* d softmax: y * (g - <g, y>) */
            float dot = 0.0f;
            for (int64_t c = 0; c < C; c++) dot += gr[c] * yr[c];
            for (int64_t c = 0; c < C; c++) o[c] += yr[c] * (gr[c] - dot);
        }
    }
}

const nc_op_def nc_op_log_softmax = {"log_softmax", 0, SM_LOG, softmax_infer, NULL, softmax_fwd, softmax_bwd};
const nc_op_def nc_op_softmax = {"softmax", 0, SM_PLAIN, softmax_infer, NULL, softmax_fwd, softmax_bwd};

nc_tensor *nc_log_softmax(nc_tensor *x) { return nc__apply(&nc_op_log_softmax, &x, 1, NULL); }
nc_tensor *nc_softmax(nc_tensor *x) { return nc__apply(&nc_op_softmax, &x, 1, NULL); }

/* ------------------------------------------------------------------------ */
/* cross entropy (fused log-softmax + NLL, mean over the batch)              */
/* ------------------------------------------------------------------------ */

static nc_status ce_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0], *y = n->in[1];
    (void)shape;
    if (!nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    if (y->dtype != NC_I32) {
        nc__error(x->ctx, NC_ERR_DTYPE, "cross_entropy: labels must be i32");
        return NC_ERR_DTYPE;
    }
    if (x->ndim != 2 || y->ndim != 1 || y->shape[0] != x->shape[0] || x->shape[1] < 1) {
        char sa[96], sb[96];
        nc__error(x->ctx, NC_ERR_SHAPE, "cross_entropy: expected logits [N, C] and labels [N], got %s and %s",
                  nc__shape_str(x->ndim, x->shape, sa, sizeof sa), nc__shape_str(y->ndim, y->shape, sb, sizeof sb));
        return NC_ERR_SHAPE;
    }
    *dt = NC_F32;
    *nd = 0;
    return NC_OK;
}

typedef struct {
    float *prob; /* softmax, [N, C] */
    int32_t *lab;
} ce_saved;

static nc_status ce_fwd(nc_node *n) {
    nc_ctx *ctx = n->out->ctx;
    const float *x = nc__contig_f32(n->in[0]);
    int64_t N = n->in[0]->shape[0], C = n->in[0]->shape[1];
    ce_saved *sv = (ce_saved *)nc__arena_alloc(ctx, sizeof(ce_saved));
    int32_t *lab = (int32_t *)nc__arena_alloc(ctx, (size_t)N * sizeof(int32_t));
    float *prob = (float *)nc__arena_alloc(ctx, (size_t)(N * C) * sizeof(float));
    if (!x || !sv || !lab || !prob) return NC_ERR_OOM;
    nc__copy_strided(n->in[1], lab);

    double loss = 0.0;
    for (int64_t i = 0; i < N; i++) {
        if (lab[i] < 0 || lab[i] >= C) {
            nc__error(ctx, NC_ERR_ARG, "cross_entropy: label %d at row %lld outside [0, %lld)", lab[i],
                      (long long)i, (long long)C);
            return NC_ERR_ARG;
        }
        float *p = prob + i * C;
        softmax_row(ctx->k, C, x + i * C, p, 1);
        loss -= p[lab[i]];
        ctx->k->exp_f32(C, p, p);
    }
    *(float *)n->out->data = N ? (float)(loss / (double)N) : NAN; /* mean over no rows: 0/0, as in torch */
    sv->prob = prob;
    sv->lab = lab;
    n->saved = sv;
    return NC_OK;
}

static void ce_bwd(nc_node *n) {
    nc_tensor *gt = nc__grad_buf(n->in[0]);
    if (!gt) return;
    int64_t N = n->in[0]->shape[0], C = n->in[0]->shape[1];
    const ce_saved *sv = (const ce_saved *)n->saved;
    const float *prob = sv->prob;
    const int32_t *lab = sv->lab;
    float s = *(const float *)n->out->grad->data / (float)(N ? N : 1);
    float *gi = (float *)gt->data;
    for (int64_t i = 0; i < N; i++) {
        for (int64_t c = 0; c < C; c++) gi[i * C + c] += s * prob[i * C + c];
        gi[i * C + lab[i]] -= s;
    }
}

const nc_op_def nc_op_cross_entropy = {"cross_entropy", 0, 0, ce_infer, NULL, ce_fwd, ce_bwd};

nc_tensor *nc_cross_entropy(nc_tensor *logits, nc_tensor *labels) {
    nc_tensor *in[2] = {logits, labels};
    return nc__apply(&nc_op_cross_entropy, in, 2, NULL);
}

/* Composite: differentiable through the ops it is built from. */
nc_tensor *nc_mse_loss(nc_tensor *pred, nc_tensor *target) {
    nc_tensor *d = nc_sub(pred, target);
    return nc_mean(nc_mul(d, d), NC_ALL_AXES, 0);
}

/* ------------------------------------------------------------------------ */
/* dropout (inverted)                                                        */
/* ------------------------------------------------------------------------ */

static nc_status dropout_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0];
    if (!nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    float p = n->p.f[0];
    if (!(p >= 0.0f && p <= 1.0f)) {
        nc__error(x->ctx, NC_ERR_ARG, "dropout: p must be in [0, 1], got %g", p);
        return NC_ERR_ARG;
    }
    *dt = NC_F32;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    return NC_OK;
}

static nc_status dropout_fwd(nc_node *n) {
    nc_ctx *ctx = n->out->ctx;
    const float *x = nc__contig_f32(n->in[0]);
    int64_t N = nc_numel(n->out);
    float *mask = (float *)nc__arena_alloc(ctx, (size_t)(N ? N : 1) * sizeof(float));
    if (!x || !mask) return NC_ERR_OOM;
    float p = n->p.f[0], keep = p < 1.0f ? 1.0f / (1.0f - p) : 0.0f;
    float *y = (float *)n->out->data;
    for (int64_t i = 0; i < N; i++) {
        mask[i] = nc__rand_uniform01(ctx) >= p ? keep : 0.0f;
        y[i] = x[i] * mask[i];
    }
    n->saved = mask;
    return NC_OK;
}

static void dropout_bwd(nc_node *n) {
    nc_tensor *gt = nc__grad_buf(n->in[0]);
    if (!gt) return;
    const float *mask = (const float *)n->saved, *g = (const float *)n->out->grad->data;
    float *gi = (float *)gt->data;
    for (int64_t i = 0, N = nc_numel(n->out); i < N; i++) gi[i] += g[i] * mask[i];
}

const nc_op_def nc_op_dropout = {"dropout", NC_OPF_ELEMENTWISE, 0, dropout_infer, NULL, dropout_fwd, dropout_bwd};

nc_tensor *nc_dropout(nc_tensor *x, float p) {
    nc_op_params prm;
    memset(&prm, 0, sizeof prm);
    prm.f[0] = p;
    return nc__apply(&nc_op_dropout, &x, 1, &prm);
}

/* ------------------------------------------------------------------------ */
/* batch_norm: inputs x, weight, bias, running_mean, running_var             */
/* params: i[0] = training, f[0] = momentum, f[1] = eps                      */
/* ------------------------------------------------------------------------ */

typedef struct {
    float *mean, *invstd;
} bn_saved;

static void bn_dims(const nc_tensor *x, int64_t *N, int64_t *C, int64_t *S) {
    *N = x->shape[0];
    *C = x->shape[1];
    *S = 1;
    for (int i = 2; i < x->ndim; i++) *S *= x->shape[i];
}

static nc_status bn_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0];
    for (int i = 0; i < 5; i++)
        if (!nc__expect_f32(n, i)) return NC_ERR_DTYPE;
    if (x->ndim < 2) {
        nc__error(x->ctx, NC_ERR_SHAPE, "batch_norm: expected input [N, C, ...]");
        return NC_ERR_SHAPE;
    }
    int64_t N, C, S;
    bn_dims(x, &N, &C, &S);
    for (int i = 1; i < 5; i++) {
        nc_tensor *t = n->in[i];
        if (t->ndim != 1 || t->shape[0] != C) {
            nc__error(x->ctx, NC_ERR_SHAPE, "batch_norm: weight, bias and running stats must be [%lld]", (long long)C);
            return NC_ERR_SHAPE;
        }
    }
    if (n->p.i[0] && N * S <= 1) {
        nc__error(x->ctx, NC_ERR_SHAPE, "batch_norm: training needs more than 1 value per channel");
        return NC_ERR_SHAPE;
    }
    if (!nc_data_f32(n->in[3]) || !nc_data_f32(n->in[4])) {
        nc__error(x->ctx, NC_ERR_ARG, "batch_norm: running stats must be contiguous");
        return NC_ERR_ARG;
    }
    *dt = NC_F32;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    return NC_OK;
}

static nc_status bn_fwd(nc_node *n) {
    nc_tensor *x = n->in[0];
    nc_ctx *ctx = x->ctx;
    int64_t N, C, S;
    bn_dims(x, &N, &C, &S);
    int64_t M = N * S;
    const float *X = nc__contig_f32(x), *w = nc__contig_f32(n->in[1]), *b = nc__contig_f32(n->in[2]);
    float *rm = nc_data_f32(n->in[3]), *rv = nc_data_f32(n->in[4]);
    bn_saved *sv = (bn_saved *)nc__arena_alloc(ctx, sizeof(bn_saved));
    float *mean = (float *)nc__arena_alloc(ctx, (size_t)C * sizeof(float));
    float *invstd = (float *)nc__arena_alloc(ctx, (size_t)C * sizeof(float));
    if (!X || !w || !b || !sv || !mean || !invstd) return NC_ERR_OOM;
    float mom = n->p.f[0], eps = n->p.f[1];
    int training = (int)n->p.i[0];
    for (int64_t c = 0; c < C; c++) {
        if (training) {
            double s = 0.0, ss = 0.0;
            for (int64_t i = 0; i < N; i++) {
                const float *p = X + (i * C + c) * S;
                for (int64_t j = 0; j < S; j++) s += p[j];
            }
            double mu = s / (double)M;
            for (int64_t i = 0; i < N; i++) {
                const float *p = X + (i * C + c) * S;
                for (int64_t j = 0; j < S; j++) ss += (p[j] - mu) * (p[j] - mu);
            }
            mean[c] = (float)mu;
            invstd[c] = (float)(1.0 / sqrt(ss / (double)M + eps));
            /* running variance is unbiased, as in PyTorch */
            rm[c] = (1.0f - mom) * rm[c] + mom * (float)mu;
            rv[c] = (1.0f - mom) * rv[c] + mom * (float)(ss / (double)(M - 1));
        } else {
            mean[c] = rm[c];
            invstd[c] = 1.0f / sqrtf(rv[c] + eps);
        }
    }
    float *Y = (float *)n->out->data;
    for (int64_t i = 0; i < N; i++)
        for (int64_t c = 0; c < C; c++) {
            const float *p = X + (i * C + c) * S;
            float *y = Y + (i * C + c) * S;
            float a = w[c] * invstd[c], d = b[c] - mean[c] * a;
            for (int64_t j = 0; j < S; j++) y[j] = p[j] * a + d;
        }
    sv->mean = mean;
    sv->invstd = invstd;
    n->saved = sv;
    return NC_OK;
}

static void bn_bwd(nc_node *n) {
    nc_tensor *x = n->in[0];
    int64_t N, C, S;
    bn_dims(x, &N, &C, &S);
    int64_t M = N * S;
    const bn_saved *sv = (const bn_saved *)n->saved;
    const float *X = nc__contig_f32(x), *w = nc__contig_f32(n->in[1]);
    const float *G = (const float *)n->out->grad->data;
    if (!X || !w) return;
    nc_tensor *gx = nc__needs_grad(x) ? nc__grad_buf(x) : NULL;
    nc_tensor *gw = nc__needs_grad(n->in[1]) ? nc__grad_buf(n->in[1]) : NULL;
    nc_tensor *gb = nc__needs_grad(n->in[2]) ? nc__grad_buf(n->in[2]) : NULL;
    int training = (int)n->p.i[0];
    for (int64_t c = 0; c < C; c++) {
        float mu = sv->mean[c], is = sv->invstd[c];
        double sg = 0.0, sgx = 0.0; /* sum(g), sum(g * xhat) */
        for (int64_t i = 0; i < N; i++) {
            const float *p = X + (i * C + c) * S, *g = G + (i * C + c) * S;
            for (int64_t j = 0; j < S; j++) {
                sg += g[j];
                sgx += g[j] * (p[j] - mu) * is;
            }
        }
        if (gw) ((float *)gw->data)[c] += (float)sgx;
        if (gb) ((float *)gb->data)[c] += (float)sg;
        if (!gx) continue;
        float k = w[c] * is;
        float mg = (float)(sg / (double)M), mgx = (float)(sgx / (double)M);
        for (int64_t i = 0; i < N; i++) {
            const float *p = X + (i * C + c) * S, *g = G + (i * C + c) * S;
            float *o = (float *)gx->data + (i * C + c) * S;
            if (training)
                for (int64_t j = 0; j < S; j++) o[j] += k * (g[j] - mg - (p[j] - mu) * is * mgx);
            else
                for (int64_t j = 0; j < S; j++) o[j] += k * g[j];
        }
    }
}

const nc_op_def nc_op_batch_norm = {"batch_norm", 0, 0, bn_infer, NULL, bn_fwd, bn_bwd};

nc_tensor *nc_batch_norm(nc_tensor *x, nc_tensor *weight, nc_tensor *bias, nc_tensor *running_mean,
                         nc_tensor *running_var, int training, float momentum, float eps) {
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = training ? 1 : 0;
    p.f[0] = momentum;
    p.f[1] = eps;
    nc_tensor *in[5] = {x, weight, bias, running_mean, running_var};
    return nc__apply(&nc_op_batch_norm, in, 5, &p);
}

/* ------------------------------------------------------------------------ */
/* nll_loss: -logp[i, label_i], mean (p.i[0] = 0) or sum (1) over rows       */
/* ------------------------------------------------------------------------ */

static nc_status nll_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0], *y = n->in[1];
    (void)shape;
    if (!nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    if (y->dtype != NC_I32) {
        nc__error(x->ctx, NC_ERR_DTYPE, "nll_loss: labels must be i32");
        return NC_ERR_DTYPE;
    }
    if (x->ndim != 2 || y->ndim != 1 || y->shape[0] != x->shape[0]) {
        char sa[96], sb[96];
        nc__error(x->ctx, NC_ERR_SHAPE, "nll_loss: expected log-probabilities [N, C] and labels [N], got %s and %s",
                  nc__shape_str(x->ndim, x->shape, sa, sizeof sa), nc__shape_str(y->ndim, y->shape, sb, sizeof sb));
        return NC_ERR_SHAPE;
    }
    *dt = NC_F32;
    *nd = 0;
    return NC_OK;
}

static nc_status nll_fwd(nc_node *n) {
    nc_ctx *ctx = n->out->ctx;
    const float *x = nc__contig_f32(n->in[0]);
    int64_t N = n->in[0]->shape[0], C = n->in[0]->shape[1];
    int32_t *lab = (int32_t *)nc__arena_alloc(ctx, (size_t)(N ? N : 1) * sizeof(int32_t));
    if (!x || !lab) return NC_ERR_OOM;
    nc__copy_strided(n->in[1], lab);
    double loss = 0.0;
    for (int64_t i = 0; i < N; i++) {
        if (lab[i] < 0 || lab[i] >= C) {
            nc__error(ctx, NC_ERR_ARG, "nll_loss: label %d at row %lld outside [0, %lld)", lab[i], (long long)i,
                      (long long)C);
            return NC_ERR_ARG;
        }
        loss -= x[i * C + lab[i]];
    }
    *(float *)n->out->data = n->p.i[0] ? (float)loss : N ? (float)(loss / (double)N) : NAN;
    n->saved = lab;
    return NC_OK;
}

static void nll_bwd(nc_node *n) {
    nc_tensor *gt = nc__grad_buf(n->in[0]);
    if (!gt) return;
    int64_t N = n->in[0]->shape[0], C = n->in[0]->shape[1];
    const int32_t *lab = (const int32_t *)n->saved;
    float g = *(const float *)n->out->grad->data;
    float s = n->p.i[0] ? g : g / (float)(N ? N : 1);
    for (int64_t i = 0; i < N; i++) ((float *)gt->data)[i * C + lab[i]] -= s;
}

const nc_op_def nc_op_nll_loss = {"nll_loss", 0, 0, nll_infer, NULL, nll_fwd, nll_bwd};

nc_tensor *nc_nll_loss(nc_tensor *logp, nc_tensor *labels, int sum) {
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = sum ? 1 : 0;
    nc_tensor *in[2] = {logp, labels};
    return nc__apply(&nc_op_nll_loss, in, 2, &p);
}

/* ------------------------------------------------------------------------ */
/* layer_norm over the last D elements of each row (torch layer_norm).        */
/* Inputs: x, then weight and bias [D] if present (p.i[1], p.i[2]).           */
/* Saves mean and 1/std per row for backward.                                 */
/* ------------------------------------------------------------------------ */

static nc_status layer_norm_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0];
    int k = (int)n->p.i[0];
    for (int i = 0; i < n->n_in; i++)
        if (!nc__expect_f32(n, i)) return NC_ERR_DTYPE;
    if (k < 1 || k > x->ndim) {
        nc__error(x->ctx, NC_ERR_SHAPE, "layer_norm: normalized_shape has %d dims, input only %d", k, x->ndim);
        return NC_ERR_SHAPE;
    }
    int64_t D = nc__shape_numel(k, x->shape + x->ndim - k);
    for (int i = 1; i < n->n_in; i++) {
        if (nc_numel(n->in[i]) != D) {
            nc__error(x->ctx, NC_ERR_SHAPE, "layer_norm: weight/bias need %lld elements, got %lld", (long long)D,
                      (long long)nc_numel(n->in[i]));
            return NC_ERR_SHAPE;
        }
    }
    *dt = NC_F32;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    return NC_OK;
}

static int64_t ln_dim(const nc_node *n) {
    const nc_tensor *x = n->in[0];
    int k = (int)n->p.i[0];
    return nc__shape_numel(k, x->shape + x->ndim - k);
}

static nc_status layer_norm_fwd(nc_node *n) {
    nc_tensor *x = n->in[0];
    int has_w = (int)n->p.i[1], has_b = (int)n->p.i[2];
    const float *w = has_w ? nc__contig_f32(n->in[1]) : NULL;
    const float *b = has_b ? nc__contig_f32(n->in[1 + has_w]) : NULL;
    const float *X = nc__contig_f32(x);
    int64_t D = ln_dim(n), R = D ? nc_numel(x) / D : 0;
    float *stats = (float *)nc__arena_alloc(x->ctx, (size_t)(2 * R + 1) * sizeof(float)); /* mean, rstd */
    if (!X || !stats || (has_w && !w) || (has_b && !b)) return NC_ERR_OOM;
    float *Y = (float *)n->out->data, eps = n->p.f[0];
    for (int64_t r = 0; r < R; r++) {
        const float *xr = X + r * D;
        double s = 0.0, ss = 0.0;
        for (int64_t i = 0; i < D; i++) s += xr[i];
        double mean = s / (double)D;
        for (int64_t i = 0; i < D; i++) ss += (xr[i] - mean) * (xr[i] - mean);
        float rstd = (float)(1.0 / sqrt(ss / (double)D + eps)); /* biased variance, as torch */
        stats[2 * r] = (float)mean;
        stats[2 * r + 1] = rstd;
        float *yr = Y + r * D;
        for (int64_t i = 0; i < D; i++) {
            float v = (xr[i] - (float)mean) * rstd;
            yr[i] = (w ? v * w[i] : v) + (b ? b[i] : 0.0f);
        }
    }
    n->saved = stats;
    return NC_OK;
}

static void layer_norm_bwd(nc_node *n) {
    nc_tensor *x = n->in[0];
    int has_w = (int)n->p.i[1], has_b = (int)n->p.i[2];
    nc_tensor *wt = has_w ? n->in[1] : NULL, *bt = has_b ? n->in[1 + has_w] : NULL;
    const float *X = nc__contig_f32(x), *w = wt ? nc__contig_f32(wt) : NULL, *stats = (const float *)n->saved;
    const float *g = (const float *)n->out->grad->data;
    int64_t D = ln_dim(n), R = D ? nc_numel(x) / D : 0;
    if (!X || (wt && !w)) return;
    nc_tensor *gx = nc__needs_grad(x) ? nc__grad_buf(x) : NULL;
    nc_tensor *gw = wt && nc__needs_grad(wt) ? nc__grad_buf(wt) : NULL;
    nc_tensor *gb = bt && nc__needs_grad(bt) ? nc__grad_buf(bt) : NULL;
    for (int64_t r = 0; r < R; r++) {
        const float *xr = X + r * D, *gr = g + r * D;
        float mean = stats[2 * r], rstd = stats[2 * r + 1];
        double sg = 0.0, sgx = 0.0; /* sums of g*w and g*w*xhat */
        for (int64_t i = 0; i < D; i++) {
            float xhat = (xr[i] - mean) * rstd, gi = w ? gr[i] * w[i] : gr[i];
            sg += gi;
            sgx += gi * xhat;
            if (gw) ((float *)gw->data)[i] += gr[i] * xhat;
            if (gb) ((float *)gb->data)[i] += gr[i];
        }
        if (gx) {
            float *dx = (float *)gx->data + r * D, mg = (float)(sg / (double)D), mgx = (float)(sgx / (double)D);
            for (int64_t i = 0; i < D; i++) {
                float xhat = (xr[i] - mean) * rstd, gi = w ? gr[i] * w[i] : gr[i];
                dx[i] += rstd * (gi - mg - xhat * mgx);
            }
        }
    }
}

const nc_op_def nc_op_layer_norm = {"layer_norm", 0, 0, layer_norm_infer, NULL, layer_norm_fwd, layer_norm_bwd};

nc_tensor *nc_layer_norm(nc_tensor *x, int normalized_ndim, nc_tensor *weight, nc_tensor *bias, float eps) {
    nc_tensor *in[3] = {x, NULL, NULL};
    int k = 1;
    if (weight) in[k++] = weight;
    if (bias) in[k++] = bias;
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = normalized_ndim;
    p.i[1] = weight != NULL;
    p.i[2] = bias != NULL;
    p.f[0] = eps;
    return nc__apply(&nc_op_layer_norm, in, k, &p);
}

/* ------------------------------------------------------------------------ */
/* masked_fill: out = mask ? value : x, mask (i32 0/1) broadcast onto x.      */
/* The value may be -inf (attention masks), which arithmetic cannot express.  */
/* ------------------------------------------------------------------------ */

static nc_status masked_fill_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0], *m = n->in[1];
    if (!nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    int ok = m->dtype == NC_I32 && m->ndim <= x->ndim;
    for (int i = 0; ok && i < m->ndim; i++) {
        int64_t dm = m->shape[i], dx = x->shape[x->ndim - m->ndim + i];
        ok = dm == dx || dm == 1;
    }
    if (!ok) {
        char sx[96], sm[96];
        nc__error(x->ctx, NC_ERR_SHAPE, "masked_fill: mask %s %s does not broadcast to input %s",
                  nc__dtype_name(m->dtype), nc__shape_str(m->ndim, m->shape, sm, sizeof sm),
                  nc__shape_str(x->ndim, x->shape, sx, sizeof sx));
        return NC_ERR_SHAPE;
    }
    *dt = NC_F32;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    return NC_OK;
}

static void row_masked_fill(int64_t n, float *const *p, const int64_t *s, const void *u) {
    float v = *(const float *)u;
    for (int64_t i = 0; i < n; i++) {
        int32_t m = *(const int32_t *)(const void *)(p[1] + i * s[1]);
        p[2][i * s[2]] = m ? v : p[0][i * s[0]];
    }
}

static void row_masked_grad(int64_t n, float *const *p, const int64_t *s, const void *u) {
    (void)u;
    for (int64_t i = 0; i < n; i++) {
        int32_t m = *(const int32_t *)(const void *)(p[1] + i * s[1]);
        if (!m) p[2][i * s[2]] += p[0][i * s[0]];
    }
}

static nc_status masked_fill_fwd(nc_node *n) {
    nc_tensor *x = n->in[0], *m = n->in[1], *o = n->out;
    int64_t st[3][NC_MAX_DIMS];
    memcpy(st[0], x->stride, sizeof st[0]);
    nc__bcast_strides(m, o->ndim, o->shape, st[1]);
    memcpy(st[2], o->stride, sizeof st[2]);
    float *base[3] = {(float *)x->data, (float *)m->data, (float *)o->data};
    nc__iterate(3, base, (const int64_t(*)[NC_MAX_DIMS])st, o->ndim, o->shape, row_masked_fill, &n->p.f[0]);
    return NC_OK;
}

static void masked_fill_bwd(nc_node *n) {
    nc_tensor *x = n->in[0], *m = n->in[1], *o = n->out;
    nc_tensor *gx = nc__grad_buf(x);
    if (!gx) return;
    int64_t st[3][NC_MAX_DIMS];
    nc__contig_strides(o->ndim, o->shape, st[0]);
    nc__bcast_strides(m, o->ndim, o->shape, st[1]);
    nc__contig_strides(o->ndim, o->shape, st[2]);
    float *base[3] = {(float *)o->grad->data, (float *)m->data, (float *)gx->data};
    nc__iterate(3, base, (const int64_t(*)[NC_MAX_DIMS])st, o->ndim, o->shape, row_masked_grad, NULL);
}

const nc_op_def nc_op_masked_fill = {"masked_fill", 0, 0, masked_fill_infer, NULL, masked_fill_fwd,
                                     masked_fill_bwd};

nc_tensor *nc_masked_fill(nc_tensor *x, nc_tensor *mask, float value) {
    if (mask && mask->dtype != NC_I32) mask = nc_to_dtype(mask, NC_I32); /* u8 / i8 masks */
    nc_tensor *in[2] = {x, mask};
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.f[0] = value;
    return nc__apply(&nc_op_masked_fill, in, 2, &p);
}
