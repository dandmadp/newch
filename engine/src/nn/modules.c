/* NN modules: a tiny vtable so containers (Sequential now, Keras-style
 * models later) can treat every layer the same way. Modules are allocated in
 * persistent memory and freed with the context. */
#include "../core/nc_internal.h"

#include <math.h>

typedef struct nc_module_vt {
    nc_tensor *(*forward)(nc_module *m, nc_tensor *x);
    int (*params)(nc_module *m, nc_tensor **out, int max);
    void (*set_training)(nc_module *m, int training); /* NULL = flag only */
} nc_module_vt;

struct nc_module {
    const nc_module_vt *vt;
    nc_ctx *ctx;
    const char *name;
    int training;
};

static int emit(nc_tensor **out, int max, int count, nc_tensor *t) {
    if (t && out && count < max) out[count] = t;
    return t ? count + 1 : count;
}

/* ------------------------------------------------------------------------ */
/* Linear: y = x @ W^T + b, W stored [out, in] like PyTorch                  */
/* ------------------------------------------------------------------------ */

typedef struct {
    nc_module base;
    nc_tensor *weight;
    nc_tensor *bias;
} linear_m;

static nc_tensor *linear_forward(nc_module *m, nc_tensor *x) {
    linear_m *l = (linear_m *)m;
    return nc_linear_fn(x, l->weight, l->bias);
}

static int linear_params(nc_module *m, nc_tensor **out, int max) {
    linear_m *l = (linear_m *)m;
    int c = emit(out, max, 0, l->weight);
    return emit(out, max, c, l->bias);
}

static const nc_module_vt linear_vt = {linear_forward, linear_params, NULL};

/* PyTorch's default init: U(-1/sqrt(in), 1/sqrt(in)) for weight and bias. */
static void fill_uniform(nc_ctx *ctx, nc_tensor *t, float bound) {
    float *d = (float *)t->data;
    for (int64_t i = 0, n = nc_numel(t); i < n; i++) d[i] = bound * (2.0f * nc__rand_uniform01(ctx) - 1.0f);
}

nc_module *nc_linear(nc_ctx *ctx, int64_t in_features, int64_t out_features, int bias) {
    if (in_features <= 0 || out_features <= 0) {
        nc__error(ctx, NC_ERR_ARG, "linear: features must be positive");
        return NULL;
    }
    linear_m *l = (linear_m *)nc__persistent_alloc(ctx, sizeof(linear_m));
    if (!l) return NULL;
    l->base = (nc_module){&linear_vt, ctx, "Linear", 1};
    int64_t ws[2] = {out_features, in_features};
    float bound = 1.0f / sqrtf((float)in_features);
    l->weight = nc_parameter(ctx, NULL, 2, ws);
    if (!l->weight) return NULL;
    nc_set_name(l->weight, "weight");
    fill_uniform(ctx, l->weight, bound);
    if (bias) {
        l->bias = nc_parameter(ctx, NULL, 1, &out_features);
        if (!l->bias) return NULL;
        nc_set_name(l->bias, "bias");
        fill_uniform(ctx, l->bias, bound);
    }
    return &l->base;
}

/* ------------------------------------------------------------------------ */
/* Activation                                                                */
/* ------------------------------------------------------------------------ */

typedef struct {
    nc_module base;
    nc_activation kind;
} act_m;

static nc_tensor *act_forward(nc_module *m, nc_tensor *x) {
    switch (((act_m *)m)->kind) {
    case NC_ACT_RELU: return nc_relu(x);
    case NC_ACT_SIGMOID: return nc_sigmoid(x);
    case NC_ACT_TANH: return nc_tanh(x);
    }
    return NULL;
}

static int no_params(nc_module *m, nc_tensor **out, int max) {
    (void)m, (void)out, (void)max;
    return 0;
}

static const nc_module_vt act_vt = {act_forward, no_params, NULL};

nc_module *nc_act(nc_ctx *ctx, nc_activation kind) {
    static const char *names[] = {"ReLU", "Sigmoid", "Tanh"};
    act_m *a = (act_m *)nc__persistent_alloc(ctx, sizeof(act_m));
    if (!a) return NULL;
    a->base = (nc_module){&act_vt, ctx, names[kind], 1};
    a->kind = kind;
    return &a->base;
}

/* ------------------------------------------------------------------------ */
/* Sequential                                                                */
/* ------------------------------------------------------------------------ */

typedef struct {
    nc_module base;
    int n;
    nc_module **mods;
} seq_m;

static nc_tensor *seq_forward(nc_module *m, nc_tensor *x) {
    seq_m *s = (seq_m *)m;
    for (int i = 0; i < s->n; i++) x = nc_forward(s->mods[i], x);
    return x;
}

static int seq_params(nc_module *m, nc_tensor **out, int max) {
    seq_m *s = (seq_m *)m;
    int count = 0;
    for (int i = 0; i < s->n; i++) {
        int room = count < max ? max - count : 0;
        count += nc_parameters(s->mods[i], room ? out + count : NULL, room);
    }
    return count;
}

static void seq_set_training(nc_module *m, int training) {
    seq_m *s = (seq_m *)m;
    for (int i = 0; i < s->n; i++) nc_set_training(s->mods[i], training);
}

static const nc_module_vt seq_vt = {seq_forward, seq_params, seq_set_training};

nc_module *nc_sequential(nc_ctx *ctx, int n, nc_module **mods) {
    for (int i = 0; i < n; i++)
        if (!mods[i]) return NULL; /* a layer failed to build; error already set */
    seq_m *s = (seq_m *)nc__persistent_alloc(ctx, sizeof(seq_m));
    nc_module **copy = (nc_module **)nc__persistent_alloc(ctx, sizeof(nc_module *) * (size_t)(n ? n : 1));
    if (!s || !copy) return NULL;
    memcpy(copy, mods, sizeof(nc_module *) * (size_t)n);
    s->base = (nc_module){&seq_vt, ctx, "Sequential", 1};
    s->n = n;
    s->mods = copy;
    return &s->base;
}

/* ------------------------------------------------------------------------ */
/* Generic                                                                   */
/* ------------------------------------------------------------------------ */

nc_tensor *nc_forward(nc_module *m, nc_tensor *x) { return (m && x) ? m->vt->forward(m, x) : NULL; }

int nc_parameters(nc_module *m, nc_tensor **out, int max) { return m ? m->vt->params(m, out, max) : 0; }

const char *nc_module_name(const nc_module *m) { return m ? m->name : ""; }

void nc_set_training(nc_module *m, int training) {
    if (!m) return;
    m->training = training ? 1 : 0;
    if (m->vt->set_training) m->vt->set_training(m, training);
}
