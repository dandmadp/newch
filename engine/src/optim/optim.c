/* Optimizers. State lives in persistent memory next to the parameters.
 * Memory per parameter: SGD 0x (no momentum) or 1x, Adam 2x. */
#include "../core/nc_internal.h"

#include <math.h>

enum { OPT_SGD, OPT_ADAM };

struct nc_optim {
    nc_ctx *ctx;
    int kind;
    int n;
    nc_tensor **params;
    float **s1; /* momentum / Adam m */
    float **s2; /* Adam v */
    nc_optim_opts o;
    int64_t t;
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
    size_t ptrs = sizeof(void *) * (size_t)(n ? n : 1);
    opt->params = (nc_tensor **)nc__persistent_alloc(ctx, ptrs);
    if (!opt->params) return NULL;
    memcpy(opt->params, params, sizeof(nc_tensor *) * (size_t)n);

    int need1 = kind == OPT_ADAM || o.momentum != 0.0f, need2 = kind == OPT_ADAM;
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

nc_optim *nc_adam(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts) {
    if (opts.beta1 == 0.0f) opts.beta1 = 0.9f;
    if (opts.beta2 == 0.0f) opts.beta2 = 0.999f;
    if (opts.eps == 0.0f) opts.eps = 1e-8f;
    return create(ctx, OPT_ADAM, params, n, opts);
}

void nc_optim_zero_grad(nc_optim *o) {
    if (!o) return;
    for (int i = 0; i < o->n; i++) nc_zero_grad(o->params[i]);
}

void nc_optim_set_lr(nc_optim *o, float lr) {
    if (o) o->o.lr = lr;
}

void nc_optim_step(nc_optim *o) {
    if (!o || !nc_ok(o->ctx)) return;
    const nc_optim_opts *c = &o->o;
    o->t++;
    float bc1 = 1.0f, bc2 = 1.0f;
    if (o->kind == OPT_ADAM) {
        bc1 = 1.0f - powf(c->beta1, (float)o->t);
        bc2 = 1.0f - powf(c->beta2, (float)o->t);
    }
    for (int i = 0; i < o->n; i++) {
        nc_tensor *p = o->params[i];
        if (!p->grad) continue; /* not reached by backward this step */
        float *w = (float *)p->data;
        const float *g = (const float *)p->grad->data;
        int64_t N = nc_numel(p);
        if (o->kind == OPT_SGD) {
            float *v = o->s1 ? o->s1[i] : NULL;
            for (int64_t j = 0; j < N; j++) {
                float gj = g[j] + c->weight_decay * w[j];
                if (v) gj = v[j] = c->momentum * v[j] + gj;
                w[j] -= c->lr * gj;
            }
        } else {
            float *m = o->s1[i], *v = o->s2[i];
            for (int64_t j = 0; j < N; j++) {
                float gj = g[j] + c->weight_decay * w[j];
                m[j] = c->beta1 * m[j] + (1.0f - c->beta1) * gj;
                v[j] = c->beta2 * v[j] + (1.0f - c->beta2) * gj * gj;
                w[j] -= c->lr * (m[j] / bc1) / (sqrtf(v[j] / bc2) + c->eps);
            }
        }
    }
}
