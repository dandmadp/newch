/* Optimizers. State lives in persistent memory next to the parameters.
 * Memory per parameter: SGD 0x (no momentum) or 1x, RMSprop 1x or 2x, Adam/AdamW 2x.
 *
 * Gradient accumulation (Keras' gradient_accumulation_steps): with n > 1,
 * only every n-th nc_optim_step() updates the weights, using the average of
 * the gradients summed since the last update; nc_optim_zero_grad() keeps the
 * sum until then. A small batch then trains like an n times larger one at
 * the activation memory of the small batch. */
#include "../core/nc_internal.h"

#include <math.h>

enum { OPT_SGD, OPT_ADAM, OPT_ADAMW, OPT_RMSPROP };

struct nc_optim {
    nc_ctx *ctx;
    int kind;
    int n;
    nc_tensor **params;
    float **s1; /* momentum / Adam m */
    float **s2; /* Adam v */
    nc_optim_opts o;
    int64_t t;
    int accum_steps; /* 1 = update on every step */
    int accum_count; /* steps summed since the last update */
    int eps_inside_sqrt; /* RMSprop: Keras form */
};

static nc_optim *create(nc_ctx *ctx, int kind, nc_tensor **params, int n, nc_optim_opts o) {
    for (int i = 0; i < n; i++) {
        nc_tensor *p = params[i];
        if (!p) return NULL;
        if (!(p->flags & NC_TF_PERSISTENT) || !nc__needs_grad(p) || !nc_data_f32(p)) {
            nc__error(ctx, NC_ERR_ARG, "optimizer: param %d must be a contiguous f32 nc_parameter", i);
            return NULL;
        }
    }
    nc_optim *opt = (nc_optim *)nc__persistent_alloc(ctx, sizeof(nc_optim));
    if (!opt) return NULL;
    opt->ctx = ctx;
    opt->kind = kind;
    opt->n = n;
    opt->o = o;
    opt->accum_steps = 1;
    size_t ptrs = sizeof(void *) * (size_t)(n ? n : 1);
    opt->params = (nc_tensor **)nc__persistent_alloc(ctx, ptrs);
    if (!opt->params) return NULL;
    memcpy(opt->params, params, sizeof(nc_tensor *) * (size_t)n);

    int adam = kind == OPT_ADAM || kind == OPT_ADAMW;
    int need1 = adam || o.momentum != 0.0f, need2 = adam || kind == OPT_RMSPROP;
    if (need1 && !(opt->s1 = (float **)nc__persistent_alloc(ctx, ptrs))) return NULL;
    if (need2 && !(opt->s2 = (float **)nc__persistent_alloc(ctx, ptrs))) return NULL;
    for (int i = 0; i < n; i++) {
        size_t bytes = (size_t)nc_numel(params[i]) * sizeof(float);
        if (need1 && !(opt->s1[i] = (float *)nc__persistent_alloc(ctx, bytes))) return NULL;
        if (need2 && !(opt->s2[i] = (float *)nc__persistent_alloc(ctx, bytes))) return NULL;
    }
    return opt;
}

nc_optim *nc_sgd(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts) {
    return create(ctx, OPT_SGD, params, n, opts);
}

static nc_optim_opts adam_defaults(nc_optim_opts opts) {
    if (opts.beta1 == 0.0f) opts.beta1 = 0.9f;
    if (opts.beta2 == 0.0f) opts.beta2 = 0.999f;
    if (opts.eps == 0.0f) opts.eps = 1e-8f;
    return opts;
}

nc_optim *nc_adam(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts) {
    return create(ctx, OPT_ADAM, params, n, adam_defaults(opts));
}

nc_optim *nc_adamw(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts) {
    return create(ctx, OPT_ADAMW, params, n, adam_defaults(opts));
}

nc_optim *nc_rmsprop(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts, int eps_inside_sqrt) {
    if (opts.beta2 == 0.0f) opts.beta2 = 0.99f;
    if (opts.eps == 0.0f) opts.eps = 1e-8f;
    nc_optim *o = create(ctx, OPT_RMSPROP, params, n, opts);
    if (o) o->eps_inside_sqrt = eps_inside_sqrt ? 1 : 0;
    return o;
}

void nc_optim_set_accumulation(nc_optim *o, int steps) {
    if (!o) return;
    if (steps < 1) {
        nc__error(o->ctx, NC_ERR_ARG, "gradient accumulation steps must be >= 1, got %d", steps);
        return;
    }
    o->accum_steps = steps;
    o->accum_count = 0;
}

void nc_optim_zero_grad(nc_optim *o) {
    if (!o || o->accum_count != 0) return; /* still summing toward an update */
    for (int i = 0; i < o->n; i++) nc_zero_grad(o->params[i]);
}

void nc_optim_clear_grad(nc_optim *o) {
    if (!o || o->accum_count != 0) return;
    for (int i = 0; i < o->n; i++) nc_clear_grad(o->params[i]);
}

void nc_optim_set_lr(nc_optim *o, float lr) {
    if (o) o->o.lr = lr;
}

void nc_optim_step(nc_optim *o) {
    if (!o || !nc_ok(o->ctx)) return;
    if (o->accum_steps > 1 && ++o->accum_count < o->accum_steps) return;
    o->accum_count = 0;
    const nc_optim_opts *c = &o->o;
    const float gs = 1.0f / (float)o->accum_steps; /* average the summed grads */
    o->t++;
    float bc1 = 1.0f, bc2 = 1.0f;
    if (o->kind == OPT_ADAM || o->kind == OPT_ADAMW) {
        bc1 = 1.0f - powf(c->beta1, (float)o->t);
        bc2 = 1.0f - powf(c->beta2, (float)o->t);
    }
    for (int i = 0; i < o->n; i++) {
        nc_tensor *p = o->params[i];
        nc_tensor *gt = nc_grad(p);
        if (!gt) continue; /* no gradient (never reached, or cleared) */
        float *w = (float *)p->data;
        const float *g = (const float *)gt->data;
        int64_t N = nc_numel(p);
        if (o->kind == OPT_SGD) {
            float *v = o->s1 ? o->s1[i] : NULL;
            for (int64_t j = 0; j < N; j++) {
                float gj = gs * g[j] + c->weight_decay * w[j];
                if (v) gj = v[j] = c->momentum * v[j] + gj;
                w[j] -= c->lr * gj;
            }
        } else if (o->kind == OPT_RMSPROP) {
            float *buf = o->s1 ? o->s1[i] : NULL, *v = o->s2[i];
            float a = c->beta2;
            for (int64_t j = 0; j < N; j++) {
                float gj = gs * g[j] + c->weight_decay * w[j];
                v[j] = a * v[j] + (1.0f - a) * gj * gj;
                float den = o->eps_inside_sqrt ? sqrtf(v[j] + c->eps) : sqrtf(v[j]) + c->eps;
                float step = gj / den;
                if (buf) step = buf[j] = c->momentum * buf[j] + step;
                w[j] -= c->lr * step;
            }
        } else {
            float *m = o->s1[i], *v = o->s2[i];
            /* Adam: L2 term added to the gradient. AdamW: decoupled decay
             * applied to the weights first (as torch.optim.AdamW). */
            float l2 = o->kind == OPT_ADAM ? c->weight_decay : 0.0f;
            float decay = o->kind == OPT_ADAMW ? 1.0f - c->lr * c->weight_decay : 1.0f;
            for (int64_t j = 0; j < N; j++) {
                w[j] *= decay;
                float gj = gs * g[j] + l2 * w[j];
                m[j] = c->beta1 * m[j] + (1.0f - c->beta1) * gj;
                v[j] = c->beta2 * v[j] + (1.0f - c->beta2) * gj * gj;
                w[j] -= c->lr * (m[j] / bc1) / (sqrtf(v[j] / bc2) + c->eps);
            }
        }
    }
}

/* torch.nn.utils.clip_grad_norm_ (2-norm over all grads together). */
float nc_clip_grad_norm(nc_tensor **params, int n, float max_norm) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        nc_tensor *g = params[i] ? nc_grad(params[i]) : NULL;
        if (!g) continue;
        const float *d = (const float *)g->data;
        for (int64_t j = 0, N = nc_numel(g); j < N; j++) ss += (double)d[j] * d[j];
    }
    float total = (float)sqrt(ss);
    float coef = max_norm / (total + 1e-6f);
    if (coef < 1.0f) {
        for (int i = 0; i < n; i++) {
            nc_tensor *g = params[i] ? nc_grad(params[i]) : NULL;
            if (g) g->ctx->k->scale_f32(nc_numel(g), coef, (const float *)g->data, (float *)g->data);
        }
    }
    return total;
}
