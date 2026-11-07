/* Shape ops. These alias their input when they can (NC_OPF_VIEW) and only
 * copy when the input layout or the persistence rules require it. */
#include "../core/nc_internal.h"

#include <stdio.h>

static void row_acc1(int64_t n, float *const *p, const int64_t *s, const void *u) {
    (void)u;
    for (int64_t i = 0; i < n; i++) p[1][i * s[1]] += p[0][i * s[0]];
}

static nc_tensor *alias_with(nc_node *n, int ndim, const int64_t *shape, const int64_t *stride) {
    nc_tensor *src = n->in[0];
    /* nc__apply never calls view() in persistent mode, so views are always
     * temporaries (persistent tensors own their data) */
    nc_tensor *t = nc__tensor_alias(src, 0);
    if (!t) return NULL;
    t->ndim = ndim;
    memcpy(t->shape, shape, sizeof(int64_t) * (size_t)ndim);
    memcpy(t->stride, stride, sizeof(int64_t) * (size_t)ndim);
    n->out = t;
    return t;
}

static nc_status same_shape_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    *dt = n->in[0]->dtype;
    *nd = n->in[0]->ndim;
    memcpy(shape, n->in[0]->shape, sizeof(int64_t) * (size_t)*nd);
    return NC_OK;
}

static nc_status copy_fwd(nc_node *n) {
    nc__copy_strided(n->in[0], n->out->data);
    return NC_OK;
}

/* grad layout equals the input's logical layout, so flat accumulate works */
static void flat_bwd(nc_node *n) {
    nc_tensor *gt = nc__grad_buf(n->in[0]);
    if (gt) gt->ctx->k->axpy_f32(nc_numel(gt), 1.0f, (const float *)n->out->grad->data, (float *)gt->data);
}

/* ------------------------------------------------------------------------ */
/* reshape                                                                   */
/* ------------------------------------------------------------------------ */

static nc_status reshape_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0];
    int r = (int)n->p.i[NC_MAX_DIMS];
    int infer_at = -1;
    int64_t known = 1;
    for (int i = 0; i < r; i++) {
        shape[i] = n->p.i[i];
        if (shape[i] == -1 && infer_at < 0) infer_at = i;
        else if (shape[i] < 0) {
            nc__error(x->ctx, NC_ERR_SHAPE, "reshape: invalid dimension %lld", (long long)shape[i]);
            return NC_ERR_SHAPE;
        } else known *= shape[i];
    }
    int64_t total = nc_numel(x);
    if (infer_at >= 0) shape[infer_at] = known ? total / known : 0;
    if (nc__shape_numel(r, shape) != total) {
        char sa[96], sb[96];
        nc__error(x->ctx, NC_ERR_SHAPE, "reshape: cannot view %s as %s",
                  nc__shape_str(x->ndim, x->shape, sa, sizeof sa), nc__shape_str(r, shape, sb, sizeof sb));
        return NC_ERR_SHAPE;
    }
    *dt = x->dtype;
    *nd = r;
    return NC_OK;
}

/* Strides that show x's elements in row-major order as `shape`, if they exist (ATen's computeStride):
 * dims of x that are laid out back to back form chunks; each chunk must split into whole new dims. */
static int view_strides(const nc_tensor *x, int ndim, const int64_t *shape, int64_t *st) {
    if (nc_is_contiguous(x) || nc_numel(x) == 0) {
        nc__contig_strides(ndim, shape, st);
        return 1;
    }
    int on = x->ndim;
    const int64_t *os = x->shape, *ost = x->stride;
    int vd = ndim - 1;
    int64_t base = ost[on - 1], tn = 1, vn = 1;
    for (int td = on - 1; td >= 0; td--) {
        tn *= os[td];
        if (td == 0 || (os[td - 1] != 1 && ost[td - 1] != tn * base)) {
            while (vd >= 0 && (vn < tn || shape[vd] == 1)) {
                st[vd] = vn * base;
                vn *= shape[vd];
                vd--;
            }
            if (vn != tn) return 0;
            if (td > 0) {
                base = ost[td - 1];
                tn = vn = 1;
            }
        }
    }
    return vd == -1;
}

static int reshape_view(nc_node *n, int ndim, const int64_t *shape) {
    int64_t st[NC_MAX_DIMS];
    if (!view_strides(n->in[0], ndim, shape, st)) return 0;
    return alias_with(n, ndim, shape, st) != NULL;
}

const nc_op_def nc_op_reshape = {"reshape", NC_OPF_VIEW, 0, reshape_infer, reshape_view, copy_fwd, flat_bwd};

nc_tensor *nc_reshape(nc_tensor *x, int ndim, const int64_t *shape) {
    if (x && (ndim < 0 || ndim > NC_MAX_DIMS)) {
        nc__error(x->ctx, NC_ERR_SHAPE, "reshape: ndim %d out of range", ndim);
        return NULL;
    }
    nc_op_params p;
    memset(&p, 0, sizeof p);
    for (int i = 0; i < ndim; i++) p.i[i] = shape[i];
    p.i[NC_MAX_DIMS] = ndim;
    return nc__apply(&nc_op_reshape, &x, 1, &p);
}

/* ------------------------------------------------------------------------ */
/* transpose                                                                 */
/* ------------------------------------------------------------------------ */

static nc_status transpose_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0];
    int a0 = nc__norm_axis((int)n->p.i[0], x->ndim), a1 = nc__norm_axis((int)n->p.i[1], x->ndim);
    if (a0 < 0 || a1 < 0) {
        nc__error(x->ctx, NC_ERR_ARG, "transpose: axes (%d, %d) out of range for %d-d tensor",
                  (int)n->p.i[0], (int)n->p.i[1], x->ndim);
        return NC_ERR_ARG;
    }
    n->p.i[0] = a0;
    n->p.i[1] = a1;
    *dt = x->dtype;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    shape[a0] = x->shape[a1];
    shape[a1] = x->shape[a0];
    return NC_OK;
}

static void swapped_strides(const nc_node *n, int64_t *st) {
    const nc_tensor *x = n->in[0];
    memcpy(st, x->stride, sizeof(int64_t) * (size_t)x->ndim);
    int a0 = (int)n->p.i[0], a1 = (int)n->p.i[1];
    st[a0] = x->stride[a1];
    st[a1] = x->stride[a0];
}

static int transpose_view(nc_node *n, int ndim, const int64_t *shape) {
    int64_t st[NC_MAX_DIMS];
    swapped_strides(n, st);
    return alias_with(n, ndim, shape, st) != NULL;
}

static nc_status transpose_fwd(nc_node *n) {
    nc_tensor tmp = *n->in[0];
    tmp.ndim = n->out->ndim;
    memcpy(tmp.shape, n->out->shape, sizeof tmp.shape);
    swapped_strides(n, tmp.stride);
    nc__copy_strided(&tmp, n->out->data);
    return NC_OK;
}

static void transpose_bwd(nc_node *n) {
    nc_tensor *x = n->in[0], *o = n->out;
    nc_tensor *gt = nc__grad_buf(x);
    if (!gt) return;
    /* walk the (contiguous) out grad; write through swapped input strides */
    int64_t st[2][NC_MAX_DIMS];
    nc__contig_strides(o->ndim, o->shape, st[0]);
    nc__contig_strides(x->ndim, x->shape, st[1]);
    int a0 = (int)n->p.i[0], a1 = (int)n->p.i[1];
    int64_t t = st[1][a0];
    st[1][a0] = st[1][a1];
    st[1][a1] = t;
    float *base[2] = {(float *)o->grad->data, (float *)gt->data};
    nc__iterate(2, base, (const int64_t(*)[NC_MAX_DIMS])st, o->ndim, o->shape, row_acc1, NULL);
}

const nc_op_def nc_op_transpose = {"transpose", NC_OPF_VIEW, 0, transpose_infer, transpose_view, transpose_fwd,
                                   transpose_bwd};

nc_tensor *nc_transpose(nc_tensor *x, int axis0, int axis1) {
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = axis0;
    p.i[1] = axis1;
    return nc__apply(&nc_op_transpose, &x, 1, &p);
}

/* ------------------------------------------------------------------------ */
/* slice: x[..., start:start+len*step:step, ...] along one axis (a view)     */
/* p.i = {axis, start, len, step}                                            */
/* ------------------------------------------------------------------------ */

static nc_status slice_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0];
    int a = nc__norm_axis((int)n->p.i[0], x->ndim);
    int64_t start = n->p.i[1], len = n->p.i[2], step = n->p.i[3];
    if (a < 0) {
        nc__error(x->ctx, NC_ERR_ARG, "slice: axis %d out of range for %d-d tensor", (int)n->p.i[0], x->ndim);
        return NC_ERR_ARG;
    }
    if (step < 1 || len < 0 || start < 0 || start > x->shape[a] || (len > 0 && start + (len - 1) * step >= x->shape[a])) {
        nc__error(x->ctx, NC_ERR_ARG, "slice: start %lld, length %lld, step %lld do not fit axis %d of size %lld",
                  (long long)start, (long long)len, (long long)step, a, (long long)x->shape[a]);
        return NC_ERR_ARG;
    }
    n->p.i[0] = a;
    *dt = x->dtype;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    shape[a] = len;
    return NC_OK;
}

/* x seen through the slice: same memory, shifted start, scaled stride */
static void sliced(const nc_node *n, nc_tensor *t) {
    const nc_tensor *x = n->in[0];
    int a = (int)n->p.i[0];
    *t = *x;
    t->shape[a] = n->p.i[2];
    if (t->shape[a] > 0)
        t->data = (char *)x->data + (size_t)(n->p.i[1] * x->stride[a]) * nc__dtype_size(x->dtype);
    t->stride[a] = x->stride[a] * n->p.i[3];
}

static int slice_view(nc_node *n, int ndim, const int64_t *shape) {
    nc_tensor t;
    sliced(n, &t);
    nc_tensor *out = alias_with(n, ndim, shape, t.stride);
    if (!out) return 0;
    out->data = t.data;
    return 1;
}

static nc_status slice_fwd(nc_node *n) {
    nc_tensor t;
    sliced(n, &t);
    nc__copy_strided(&t, n->out->data);
    return NC_OK;
}

static void slice_bwd(nc_node *n) {
    nc_tensor *x = n->in[0], *o = n->out;
    nc_tensor *gt = nc__grad_buf(x);
    if (!gt) return;
    int a = (int)n->p.i[0];
    if (o->shape[a] == 0) return;
    int64_t st[2][NC_MAX_DIMS];
    nc__contig_strides(o->ndim, o->shape, st[0]);
    nc__contig_strides(x->ndim, x->shape, st[1]);
    float *base[2] = {(float *)o->grad->data, (float *)gt->data + n->p.i[1] * st[1][a]};
    st[1][a] *= n->p.i[3];
    nc__iterate(2, base, (const int64_t(*)[NC_MAX_DIMS])st, o->ndim, o->shape, row_acc1, NULL);
}

const nc_op_def nc_op_slice = {"slice", NC_OPF_VIEW, 0, slice_infer, slice_view, slice_fwd, slice_bwd};

nc_tensor *nc_slice(nc_tensor *x, int axis, int64_t start, int64_t length, int64_t step) {
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = axis;
    p.i[1] = start;
    p.i[2] = length;
    p.i[3] = step;
    return nc__apply(&nc_op_slice, &x, 1, &p);
}

/* ------------------------------------------------------------------------ */
/* contiguous / detach                                                       */
/* ------------------------------------------------------------------------ */

static int contiguous_view(nc_node *n, int ndim, const int64_t *shape) {
    if (!nc_is_contiguous(n->in[0])) return 0;
    return alias_with(n, ndim, shape, n->in[0]->stride) != NULL;
}

const nc_op_def nc_op_contiguous = {"contiguous", NC_OPF_VIEW, 0, same_shape_infer, contiguous_view, copy_fwd,
                                    flat_bwd};

nc_tensor *nc_contiguous(nc_tensor *x) { return nc__apply(&nc_op_contiguous, &x, 1, NULL); }

static int detach_view(nc_node *n, int ndim, const int64_t *shape) {
    return alias_with(n, ndim, shape, n->in[0]->stride) != NULL;
}

const nc_op_def nc_op_detach = {"detach", NC_OPF_VIEW | NC_OPF_NONDIFF, 0, same_shape_infer, detach_view, copy_fwd,
                                NULL};

nc_tensor *nc_detach(nc_tensor *x) { return nc__apply(&nc_op_detach, &x, 1, NULL); }

/* ------------------------------------------------------------------------ */
/* index_select (axis 0)                                                     */
/* ------------------------------------------------------------------------ */

static nc_status index_select_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x = n->in[0], *idx = n->in[1];
    if (x->ndim < 1 || idx->ndim != 1 || idx->dtype != NC_I32) {
        char sa[96], sb[96];
        nc__error(x->ctx, NC_ERR_SHAPE, "index_select: expected x [N, ...] and i32 indices [B], got %s and %s %s",
                  nc__shape_str(x->ndim, x->shape, sa, sizeof sa), nc__dtype_name(idx->dtype),
                  nc__shape_str(idx->ndim, idx->shape, sb, sizeof sb));
        return NC_ERR_SHAPE;
    }
    *dt = x->dtype;
    *nd = x->ndim;
    memcpy(shape, x->shape, sizeof(int64_t) * (size_t)x->ndim);
    shape[0] = idx->shape[0];
    return NC_OK;
}

static nc_status index_select_fwd(nc_node *n) {
    nc_tensor *x = n->in[0], *o = n->out;
    nc_ctx *ctx = o->ctx;
    int64_t B = n->in[1]->shape[0], N = x->shape[0];
    int32_t *idx = (int32_t *)nc__arena_alloc(ctx, (size_t)(B ? B : 1) * sizeof(int32_t));
    const unsigned char *src = (const unsigned char *)x->data;
    if (!idx) return NC_ERR_OOM;
    nc__copy_strided(n->in[1], idx);
    if (!nc_is_contiguous(x)) {
        unsigned char *buf = (unsigned char *)nc__arena_alloc(ctx, (size_t)nc_numel(x) * nc__dtype_size(x->dtype));
        if (!buf) return NC_ERR_OOM;
        nc__copy_strided(x, buf);
        src = buf;
    }
    size_t row = (size_t)(N ? nc_numel(x) / N : 0) * nc__dtype_size(x->dtype);
    for (int64_t b = 0; b < B; b++) {
        if (idx[b] < 0 || idx[b] >= N) {
            nc__error(ctx, NC_ERR_ARG, "index_select: index %d at position %lld outside [0, %lld)", idx[b],
                      (long long)b, (long long)N);
            return NC_ERR_ARG;
        }
        memcpy((unsigned char *)o->data + (size_t)b * row, src + (size_t)idx[b] * row, row);
    }
    n->saved = idx;
    return NC_OK;
}

static void index_select_bwd(nc_node *n) {
    nc_tensor *x = n->in[0];
    nc_tensor *gt = nc__grad_buf(x);
    if (!gt) return;
    int64_t B = n->in[1]->shape[0], N = x->shape[0];
    int64_t row = N ? nc_numel(x) / N : 0;
    const int32_t *idx = (const int32_t *)n->saved;
    const float *g = (const float *)n->out->grad->data;
    for (int64_t b = 0; b < B; b++)
        x->ctx->k->axpy_f32(row, 1.0f, g + b * row, (float *)gt->data + idx[b] * row);
}

const nc_op_def nc_op_index_select = {"index_select", 0, 0, index_select_infer, NULL, index_select_fwd,
                                      index_select_bwd};

nc_tensor *nc_index_select(nc_tensor *x, nc_tensor *idx) {
    nc_tensor *in[2] = {x, idx};
    return nc__apply(&nc_op_index_select, in, 2, NULL);
}

/* ------------------------------------------------------------------------ */
/* cat: inputs viewed as [outer, len_k * inner]; output rows interleave them  */
/* ------------------------------------------------------------------------ */

static nc_status cat_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_tensor *x0 = n->in[0];
    int axis = nc__norm_axis((int)n->p.i[0], x0->ndim);
    if (axis < 0) {
        nc__error(x0->ctx, NC_ERR_ARG, "cat: dim %d out of range for %d-d tensors", (int)n->p.i[0], x0->ndim);
        return NC_ERR_ARG;
    }
    n->p.i[0] = axis;
    memcpy(shape, x0->shape, sizeof(int64_t) * (size_t)x0->ndim);
    for (int k = 1; k < n->n_in; k++) {
        nc_tensor *x = n->in[k];
        int same = x->dtype == x0->dtype && x->ndim == x0->ndim;
        for (int i = 0; same && i < x0->ndim; i++) same = i == axis || x->shape[i] == x0->shape[i];
        if (!same) {
            char sa[96], sb[96];
            nc__error(x0->ctx, NC_ERR_SHAPE, "cat: %s %s and %s %s differ outside dim %d", nc__dtype_name(x0->dtype),
                      nc__shape_str(x0->ndim, x0->shape, sa, sizeof sa), nc__dtype_name(x->dtype),
                      nc__shape_str(x->ndim, x->shape, sb, sizeof sb), axis);
            return NC_ERR_SHAPE;
        }
        shape[axis] += x->shape[axis];
    }
    *dt = x0->dtype;
    *nd = x0->ndim;
    return NC_OK;
}

static void cat_dims(const nc_node *n, int64_t *outer, int64_t *inner) {
    const nc_tensor *x0 = n->in[0];
    int axis = (int)n->p.i[0];
    *outer = 1, *inner = 1;
    for (int i = 0; i < axis; i++) *outer *= x0->shape[i];
    for (int i = axis + 1; i < x0->ndim; i++) *inner *= x0->shape[i];
}

static nc_status cat_fwd(nc_node *n) {
    nc_ctx *ctx = n->out->ctx;
    int axis = (int)n->p.i[0];
    size_t es = nc__dtype_size(n->out->dtype);
    int64_t outer, inner;
    cat_dims(n, &outer, &inner);
    int64_t row_out = n->out->shape[axis] * inner, off = 0;
    for (int k = 0; k < n->n_in; k++) {
        nc_tensor *x = n->in[k];
        const unsigned char *src = (const unsigned char *)x->data;
        if (!nc_is_contiguous(x)) {
            unsigned char *buf = (unsigned char *)nc__arena_alloc(ctx, (size_t)nc_numel(x) * es + 1);
            if (!buf) return NC_ERR_OOM;
            nc__copy_strided(x, buf);
            src = buf;
        }
        int64_t row = x->shape[axis] * inner;
        for (int64_t o = 0; o < outer; o++)
            memcpy((unsigned char *)n->out->data + (size_t)(o * row_out + off) * es, src + (size_t)(o * row) * es,
                   (size_t)row * es);
        off += row;
    }
    return NC_OK;
}

static void cat_bwd(nc_node *n) {
    int axis = (int)n->p.i[0];
    int64_t outer, inner;
    cat_dims(n, &outer, &inner);
    int64_t row_out = n->out->shape[axis] * inner, off = 0;
    const float *g = (const float *)n->out->grad->data;
    for (int k = 0; k < n->n_in; k++) {
        nc_tensor *x = n->in[k];
        int64_t row = x->shape[axis] * inner;
        nc_tensor *gx = nc__needs_grad(x) ? nc__grad_buf(x) : NULL;
        if (gx)
            for (int64_t o = 0; o < outer; o++)
                x->ctx->k->axpy_f32(row, 1.0f, g + o * row_out + off, (float *)gx->data + o * row);
        off += row;
    }
}

const nc_op_def nc_op_cat = {"cat", 0, 0, cat_infer, NULL, cat_fwd, cat_bwd};

nc_tensor *nc_cat(int n, nc_tensor *const *xs, int axis) {
    if (n < 1 || n > NC_MAX_INPUTS) {
        if (n > 0 && xs[0]) nc__error(xs[0]->ctx, NC_ERR_ARG, "cat: takes 1 to %d tensors, got %d", NC_MAX_INPUTS, n);
        return NULL;
    }
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = axis;
    nc_tensor *in[NC_MAX_INPUTS];
    for (int k = 0; k < n; k++) in[k] = xs[k];
    return nc__apply(&nc_op_cat, in, n, &p);
}

/* ------------------------------------------------------------------------ */
/* to_dtype                                                                  */
/* ------------------------------------------------------------------------ */

static nc_status to_dtype_infer(nc_node *n, nc_dtype *dt, int *nd, int64_t *shape) {
    nc_dtype to = (nc_dtype)n->p.i[0], from = n->in[0]->dtype;
    if (to == NC_F16 || from == NC_F16 || nc__dtype_size(to) == 0) {
        nc__error(n->in[0]->ctx, NC_ERR_DTYPE, "to_dtype: %s -> %s is not supported", nc__dtype_name(from),
                  nc__dtype_name(to));
        return NC_ERR_DTYPE;
    }
    *dt = to;
    return same_shape_infer(n, &from, nd, shape);
}

static double load_as_double(const void *p, nc_dtype dt, int64_t i) {
    switch (dt) {
    case NC_F32: return (double)((const float *)p)[i];
    case NC_I32: return (double)((const int32_t *)p)[i];
    case NC_I8: return (double)((const int8_t *)p)[i];
    default: return (double)((const uint8_t *)p)[i];
    }
}

/* Float to int truncates toward zero. Out-of-range values saturate and NaN
 * becomes 0: a plain C cast is undefined there and differs between x86 and
 * ARM, which would break cross-platform reproducibility. */
static double saturate(double v, double lo, double hi) {
    if (v != v) return 0.0;
    return v < lo ? lo : v > hi ? hi : v;
}

static nc_status to_dtype_fwd(nc_node *n) {
    nc_tensor *x = n->in[0], *o = n->out;
    void *src = nc__arena_alloc(o->ctx, (size_t)nc_numel(x) * nc__dtype_size(x->dtype) + 1);
    if (!src) return NC_ERR_OOM;
    nc__copy_strided(x, src);
    for (int64_t i = 0, N = nc_numel(x); i < N; i++) {
        double v = load_as_double(src, x->dtype, i);
        if (o->dtype == NC_F32) ((float *)o->data)[i] = (float)v;
        else if (o->dtype == NC_I32) ((int32_t *)o->data)[i] = (int32_t)saturate(v, -2147483648.0, 2147483647.0);
        else if (o->dtype == NC_I8) ((int8_t *)o->data)[i] = (int8_t)saturate(v, -128.0, 127.0);
        else ((uint8_t *)o->data)[i] = (uint8_t)saturate(v, 0.0, 255.0);
    }
    return NC_OK;
}

/* only f32 -> f32 can be on the tape (integer results are detached first) */
const nc_op_def nc_op_to_dtype = {"to_dtype", 0, 0, to_dtype_infer, NULL, to_dtype_fwd, flat_bwd};

nc_tensor *nc_to_dtype(nc_tensor *x, nc_dtype dt) {
    if (x && dt != NC_F32) x = nc_detach(x);
    nc_op_params p;
    memset(&p, 0, sizeof p);
    p.i[0] = dt;
    return nc__apply(&nc_op_to_dtype, &x, 1, &p);
}
