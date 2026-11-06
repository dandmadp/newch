/* Reductions over one axis (or all). The input is viewed as
 * [outer, R, inner] where R is the reduced extent. */
#include "../core/nc_internal.h"

#include <math.h>

enum { RED_SUM, RED_MEAN, RED_MAX, RED_ARGMAX };

typedef struct {
    int64_t outer, R, inner;
} red_dims;

static red_dims get_dims(const nc_node *n) {
    const nc_tensor *x = n->in[0];
    red_dims d = {1, 1, 1};
    int axis = (int)n->p.i[0];
    if (axis == NC_ALL_AXES) {
        d.R = nc_numel(x);
        return d;
    }
    axis = nc__norm_axis(axis, x->ndim);
    for (int i = 0; i < axis; i++) d.outer *= x->shape[i];
    d.R = x->shape[axis];
    for (int i = axis + 1; i < x->ndim; i++) d.inner *= x->shape[i];
    return d;
}

static nc_status reduce_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0];
    int int_sum = n->op->kind == RED_SUM && x->dtype == NC_I32; /* counting, e.g. sum of eq() */
    if (!int_sum && !nc__expect_f32(n, 0)) return NC_ERR_DTYPE;
    int axis = (int)n->p.i[0], keep = (int)n->p.i[1];
    *dt = (n->op->kind == RED_ARGMAX || int_sum) ? NC_I32 : NC_F32;
    if (axis == NC_ALL_AXES) {
        if (nc_numel(x) == 0 && n->op->kind >= RED_MAX) {
            nc__error(x->ctx, NC_ERR_SHAPE, "%s: reduction of an empty tensor", n->op->name);
            return NC_ERR_SHAPE;
        }
        *nd = keep ? x->ndim : 0;
        for (int i = 0; i < *nd; i++) shape[i] = 1;
        return NC_OK;
    }
    int a = nc__norm_axis(axis, x->ndim);
    if (a < 0) {
        nc__error(x->ctx, NC_ERR_ARG, "%s: axis %d out of range for %d-d tensor", n->op->name, axis, x->ndim);
        return NC_ERR_ARG;
    }
    if (x->shape[a] == 0 && n->op->kind >= RED_MAX) {
        nc__error(x->ctx, NC_ERR_SHAPE, "%s: reduction over an empty axis", n->op->name);
        return NC_ERR_SHAPE;
    }
    *nd = 0;
    for (int i = 0; i < x->ndim; i++) {
        if (i != a) shape[(*nd)++] = x->shape[i];
        else if (keep) shape[(*nd)++] = 1;
    }
    return NC_OK;
}

static nc_status reduce_fwd(nc_node *n) {
    const nc_kernels *k = n->out->ctx->k;
    red_dims d = get_dims(n);
    if (n->in[0]->dtype == NC_I32) { /* integer sum, accumulated in 64 bits */
        int32_t *buf = (int32_t *)nc__arena_alloc(n->out->ctx, (size_t)(nc_numel(n->in[0]) + 1) * sizeof(int32_t));
        if (!buf) return NC_ERR_OOM;
        nc__copy_strided(n->in[0], buf);
        int32_t *y = (int32_t *)n->out->data;
        for (int64_t o = 0; o < d.outer; o++)
            for (int64_t i = 0; i < d.inner; i++) {
                int64_t s = 0;
                for (int64_t r = 0; r < d.R; r++) s += buf[(o * d.R + r) * d.inner + i];
                y[o * d.inner + i] = (int32_t)s;
            }
        return NC_OK;
    }
    const float *x = nc__contig_f32(n->in[0]);
    if (!x) return NC_ERR_OOM;
    int kind = n->op->kind;

    if (kind == RED_SUM || kind == RED_MEAN) {
        float *y = (float *)n->out->data;
        float scale = kind == RED_MEAN ? 1.0f / (float)(d.R ? d.R : 1) : 1.0f;
        if (kind == RED_MEAN && d.R == 0) { /* mean of nothing is 0/0, as in torch and numpy */
            for (int64_t i = 0; i < d.outer * d.inner; i++) y[i] = NAN;
            return NC_OK;
        }
        if (d.inner == 1) {
            for (int64_t o = 0; o < d.outer; o++) y[o] = scale * k->sum_f32(d.R, x + o * d.R);
            return NC_OK;
        }
        memset(y, 0, (size_t)(d.outer * d.inner) * sizeof(float));
        for (int64_t o = 0; o < d.outer; o++)
            for (int64_t r = 0; r < d.R; r++)
                k->axpy_f32(d.inner, scale, x + (o * d.R + r) * d.inner, y + o * d.inner);
        return NC_OK;
    }

    /* max / argmax */
    int32_t *arg = kind == RED_ARGMAX
                       ? (int32_t *)n->out->data
                       : (int32_t *)nc__arena_alloc(n->out->ctx, (size_t)(d.outer * d.inner) * sizeof(int32_t));
    if (!arg) return NC_ERR_OOM;
    for (int64_t o = 0; o < d.outer; o++) {
        for (int64_t i = 0; i < d.inner; i++) {
            const float *p = x + o * d.R * d.inner + i;
            int32_t best = 0;
            float bv = p[0];
            for (int64_t r = 1; r < d.R; r++)
                if (p[r * d.inner] > bv) bv = p[r * d.inner], best = (int32_t)r;
            arg[o * d.inner + i] = best;
            if (kind == RED_MAX) ((float *)n->out->data)[o * d.inner + i] = bv;
        }
    }
    if (kind == RED_MAX) n->saved = arg;
    return NC_OK;
}

static void reduce_bwd(nc_node *n) {
    nc_tensor *gt = nc__grad_buf(n->in[0]);
    if (!gt) return;
    const nc_kernels *k = gt->ctx->k;
    red_dims d = get_dims(n);
    const float *g = (const float *)n->out->grad->data;
    float *gi = (float *)gt->data;
    if (n->op->kind == RED_MAX) {
        const int32_t *arg = (const int32_t *)n->saved;
        for (int64_t o = 0; o < d.outer; o++)
            for (int64_t i = 0; i < d.inner; i++)
                gi[(o * d.R + arg[o * d.inner + i]) * d.inner + i] += g[o * d.inner + i];
        return;
    }
    float scale = n->op->kind == RED_MEAN ? 1.0f / (float)(d.R ? d.R : 1) : 1.0f;
    for (int64_t o = 0; o < d.outer; o++)
        for (int64_t r = 0; r < d.R; r++)
            k->axpy_f32(d.inner, scale, g + o * d.inner, gi + (o * d.R + r) * d.inner);
}

const nc_op_def nc_op_sum = {"sum", NC_OPF_REDUCTION, RED_SUM, reduce_infer, NULL, reduce_fwd, reduce_bwd};
const nc_op_def nc_op_mean = {"mean", NC_OPF_REDUCTION, RED_MEAN, reduce_infer, NULL, reduce_fwd, reduce_bwd};
const nc_op_def nc_op_max = {"max", NC_OPF_REDUCTION, RED_MAX, reduce_infer, NULL, reduce_fwd, reduce_bwd};
const nc_op_def nc_op_argmax = {"argmax", NC_OPF_REDUCTION | NC_OPF_NONDIFF, RED_ARGMAX, reduce_infer, NULL,
                                reduce_fwd, NULL};

static nc_tensor *reduce(const nc_op_def *op, nc_tensor *x, int axis, int keepdim) {
    nc_op_params p = {{0}, {0}};
    p.i[0] = axis;
    p.i[1] = keepdim ? 1 : 0;
    return nc__apply(op, &x, 1, &p);
}

nc_tensor *nc_sum(nc_tensor *x, int axis, int keepdim) { return reduce(&nc_op_sum, x, axis, keepdim); }
nc_tensor *nc_mean(nc_tensor *x, int axis, int keepdim) { return reduce(&nc_op_mean, x, axis, keepdim); }
nc_tensor *nc_max(nc_tensor *x, int axis, int keepdim) { return reduce(&nc_op_max, x, axis, keepdim); }
nc_tensor *nc_argmax(nc_tensor *x, int axis) { return reduce(&nc_op_argmax, x, axis, 0); }
