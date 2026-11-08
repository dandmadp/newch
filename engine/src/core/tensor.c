/* Tensor creation, accessors, strided iteration and the op registry. */
#include "nc_internal.h"

#include <math.h>
#include <stdio.h>

/* ------------------------------------------------------------------------ */
/* Shape helpers                                                             */
/* ------------------------------------------------------------------------ */

size_t nc__dtype_size(nc_dtype dt) {
    switch (dt) {
    case NC_F32: return 4;
    case NC_I32: return 4;
    case NC_F16: return 2;
    case NC_I8: return 1;
    case NC_U8: return 1;
    default: return 0;
    }
}

const char *nc__dtype_name(nc_dtype dt) {
    switch (dt) {
    case NC_F32: return "f32";
    case NC_I32: return "i32";
    case NC_F16: return "f16";
    case NC_I8: return "i8";
    case NC_U8: return "u8";
    default: return "?";
    }
}

int64_t nc__shape_numel(int ndim, const int64_t *shape) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

void nc__contig_strides(int ndim, const int64_t *shape, int64_t *stride) {
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        stride[i] = s;
        s *= shape[i];
    }
}

int nc__norm_axis(int axis, int ndim) {
    if (axis < 0) axis += ndim;
    return (axis >= 0 && axis < ndim) ? axis : -1;
}

int nc__same_shape(const nc_tensor *a, const nc_tensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

const char *nc__shape_str(int ndim, const int64_t *shape, char *buf, size_t size) {
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, size - pos, "[");
    for (int i = 0; i < ndim && pos < size; i++)
        pos += (size_t)snprintf(buf + pos, size - pos, i ? ", %lld" : "%lld", (long long)shape[i]);
    if (pos < size) snprintf(buf + pos, size - pos, "]");
    return buf;
}

int nc__expect_f32(const nc_node *n, int i) {
    if (n->in[i]->dtype == NC_F32) return 1;
    nc__error(n->in[i]->ctx, NC_ERR_DTYPE, "%s: input %d must be f32, got %s", n->op->name, i,
              nc__dtype_name(n->in[i]->dtype));
    return 0;
}

void nc__bcast_strides_shape(int tn, const int64_t *tshape, int ndim, const int64_t *shape,
                             int64_t *out) {
    int64_t ts[NC_MAX_DIMS];
    nc__contig_strides(tn, tshape, ts);
    for (int i = 0; i < ndim; i++) {
        int j = i - (ndim - tn);
        out[i] = (j < 0 || (tshape[j] == 1 && shape[i] != 1)) ? 0 : ts[j];
    }
}

void nc__bcast_strides(const nc_tensor *t, int ndim, const int64_t *shape, int64_t *out) {
    for (int i = 0; i < ndim; i++) {
        int j = i - (ndim - t->ndim);
        out[i] = (j < 0 || (t->shape[j] == 1 && shape[i] != 1)) ? 0 : t->stride[j];
    }
}

/* ------------------------------------------------------------------------ */
/* Creation                                                                  */
/* ------------------------------------------------------------------------ */

static int check_shape(nc_ctx *ctx, int ndim, const int64_t *shape) {
    if (ndim < 0 || ndim > NC_MAX_DIMS) {
        nc__error(ctx, NC_ERR_SHAPE, "ndim %d out of range [0, %d]", ndim, NC_MAX_DIMS);
        return 0;
    }
    /* the element count times the largest dtype size must fit in int64 / size_t: a product that
     * wrapped around would allocate a tiny buffer for a huge shape */
    const int64_t limit = INT64_MAX / 8;
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) {
            nc__error(ctx, NC_ERR_SHAPE, "negative dimension %lld", (long long)shape[i]);
            return 0;
        }
        if (shape[i] && n > limit / shape[i]) {
            char sb[160];
            nc__error(ctx, NC_ERR_SHAPE, "shape %s has too many elements", nc__shape_str(ndim, shape, sb, sizeof sb));
            return 0;
        }
        n *= shape[i];
    }
    return 1;
}

nc_tensor *nc__tensor_new(nc_ctx *ctx, nc_dtype dt, int ndim, const int64_t *shape, int persistent) {
    if (!ctx || !check_shape(ctx, ndim, shape)) return NULL;
    if (nc__dtype_size(dt) == 0) {
        nc__error(ctx, NC_ERR_DTYPE, "unknown dtype %d", (int)dt);
        return NULL;
    }
    nc_tensor *t = (nc_tensor *)nc__alloc(ctx, sizeof(nc_tensor), persistent);
    if (!t) return NULL;
    memset(t, 0, sizeof *t);
    t->ctx = ctx;
    t->dtype = dt;
    t->ndim = ndim;
    if (ndim) memcpy(t->shape, shape, (size_t)ndim * sizeof(int64_t));
    nc__contig_strides(ndim, t->shape, t->stride);
    t->data = nc__alloc(ctx, (size_t)nc__shape_numel(ndim, shape) * nc__dtype_size(dt), persistent);
    if (!t->data) return NULL;
    t->flags |= NC_TF_OWNS_DATA;
    if (persistent) t->flags |= NC_TF_PERSISTENT;
    return t;
}

nc_tensor *nc__tensor_alias(const nc_tensor *src, int persistent) {
    nc_tensor *t = (nc_tensor *)nc__alloc(src->ctx, sizeof(nc_tensor), persistent);
    if (!t) return NULL;
    *t = *src;
    t->grad = NULL;
    t->node = NULL;
    t->name = NULL;
    t->owner = src->owner ? src->owner : (nc_tensor *)src;
    t->flags = persistent ? NC_TF_PERSISTENT : 0;
    return t;
}

nc_tensor *nc_empty(nc_ctx *ctx, nc_dtype dt, int ndim, const int64_t *shape) {
    return nc__tensor_new(ctx, dt, ndim, shape, nc__is_persistent_mode(ctx));
}

nc_tensor *nc_zeros(nc_ctx *ctx, nc_dtype dt, int ndim, const int64_t *shape) {
    nc_tensor *t = nc_empty(ctx, dt, ndim, shape);
    if (t) memset(t->data, 0, (size_t)nc_numel(t) * nc__dtype_size(dt));
    return t;
}

nc_tensor *nc_full(nc_ctx *ctx, float value, int ndim, const int64_t *shape) {
    nc_tensor *t = nc_empty(ctx, NC_F32, ndim, shape);
    if (!t) return NULL;
    float *d = (float *)t->data;
    for (int64_t i = 0, n = nc_numel(t); i < n; i++) d[i] = value;
    return t;
}

nc_tensor *nc_randn(nc_ctx *ctx, float stddev, int ndim, const int64_t *shape) {
    nc_tensor *t = nc_empty(ctx, NC_F32, ndim, shape);
    if (!t) return NULL;
    float *d = (float *)t->data;
    for (int64_t i = 0, n = nc_numel(t); i < n; i++) d[i] = stddev * nc__rand_normal(ctx);
    return t;
}

nc_tensor *nc_rand_uniform(nc_ctx *ctx, float lo, float hi, int ndim, const int64_t *shape) {
    nc_tensor *t = nc_empty(ctx, NC_F32, ndim, shape);
    if (!t) return NULL;
    float *d = (float *)t->data;
    for (int64_t i = 0, n = nc_numel(t); i < n; i++) d[i] = lo + (hi - lo) * nc__rand_uniform01(ctx);
    return t;
}

nc_tensor *nc_from_data(nc_ctx *ctx, nc_dtype dt, const void *data, int ndim, const int64_t *shape) {
    nc_tensor *t = nc_empty(ctx, dt, ndim, shape);
    if (t && data) memcpy(t->data, data, (size_t)nc_numel(t) * nc__dtype_size(dt));
    return t;
}

nc_tensor *nc_scalar(nc_ctx *ctx, float value) { return nc_full(ctx, value, 0, NULL); }

nc_tensor *nc_parameter(nc_ctx *ctx, const nc_tensor *init, int ndim, const int64_t *shape) {
    nc_tensor *t = nc__tensor_new(ctx, NC_F32, ndim, shape, 1);
    if (!t) return NULL;
    if (init) {
        if (init->dtype != NC_F32 || nc_numel(init) != nc_numel(t)) {
            nc__error(ctx, NC_ERR_SHAPE, "parameter init has %lld elements, expected %lld",
                      (long long)nc_numel(init), (long long)nc_numel(t));
            return NULL;
        }
        nc__copy_strided(init, t->data);
    }
    t->flags |= NC_TF_REQUIRES_GRAD;
    return t;
}

/* ------------------------------------------------------------------------ */
/* Accessors                                                                 */
/* ------------------------------------------------------------------------ */

int nc_ndim(const nc_tensor *t) { return t ? t->ndim : 0; }
const int64_t *nc_shape(const nc_tensor *t) { return t ? t->shape : NULL; }
int64_t nc_numel(const nc_tensor *t) { return t ? nc__shape_numel(t->ndim, t->shape) : 0; }
nc_dtype nc_get_dtype(const nc_tensor *t) { return t ? t->dtype : NC_F32; }
int nc_requires_grad(const nc_tensor *t) { return t ? nc__needs_grad(t) : 0; }
nc_tensor *nc_grad(const nc_tensor *t) {
    return (t && !(t->flags & NC_TF_GRAD_CLEARED)) ? t->grad : NULL;
}
int nc_is_persistent(const nc_tensor *t) { return t && (t->flags & NC_TF_PERSISTENT) != 0; }

void nc_release(nc_tensor *t) {
    if (!t || !(t->flags & NC_TF_PERSISTENT)) return;
    nc_ctx *ctx = t->ctx;
    if (t->grad) nc_release(t->grad);
    if (t->flags & NC_TF_OWNS_DATA) nc__persistent_free(ctx, t->data);
    nc__persistent_free(ctx, t);
}
void nc_set_name(nc_tensor *t, const char *name) {
    if (t) t->name = name;
}
const char *nc_name(const nc_tensor *t) { return t && t->name ? t->name : ""; }

int64_t nc_dim(const nc_tensor *t, int axis) {
    if (!t) return 0;
    int a = nc__norm_axis(axis, t->ndim);
    return a < 0 ? 0 : t->shape[a];
}

void nc_set_requires_grad(nc_tensor *t, int on) {
    if (!t) return;
    if (on && t->dtype != NC_F32) {
        nc__error(t->ctx, NC_ERR_DTYPE, "only f32 tensors can require grad");
        return;
    }
    if (on) t->flags |= NC_TF_REQUIRES_GRAD;
    else t->flags &= ~NC_TF_REQUIRES_GRAD;
}

int nc_is_contiguous(const nc_tensor *t) {
    if (!t) return 0;
    int64_t s = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->shape[i] != 1 && t->stride[i] != s) return 0;
        s *= t->shape[i];
    }
    return 1;
}

float *nc_data_f32(nc_tensor *t) {
    return (t && t->dtype == NC_F32 && nc_is_contiguous(t)) ? (float *)t->data : NULL;
}
int32_t *nc_data_i32(nc_tensor *t) {
    return (t && t->dtype == NC_I32 && nc_is_contiguous(t)) ? (int32_t *)t->data : NULL;
}

float nc_get(const nc_tensor *t, const int64_t *index) {
    if (!t) return NAN;
    int64_t off = 0;
    for (int i = 0; i < t->ndim; i++) off += index[i] * t->stride[i];
    if (t->dtype == NC_F32) return ((const float *)t->data)[off];
    if (t->dtype == NC_I32) return (float)((const int32_t *)t->data)[off];
    if (t->dtype == NC_I8) return (float)((const int8_t *)t->data)[off];
    if (t->dtype == NC_U8) return (float)((const uint8_t *)t->data)[off];
    return NAN;
}

float nc_item(const nc_tensor *t) {
    if (!t || nc_numel(t) != 1) return NAN;
    int64_t zero[NC_MAX_DIMS] = {0};
    return nc_get(t, zero);
}

int nc_copy_to(const nc_tensor *t, void *dst, size_t dst_bytes) {
    if (!t || !dst) return 0;
    size_t need = (size_t)nc_numel(t) * nc__dtype_size(t->dtype);
    if (dst_bytes < need) {
        nc__error(t->ctx, NC_ERR_ARG, "nc_copy_to: buffer has %zu bytes, need %zu", dst_bytes, need);
        return 0;
    }
    nc__copy_strided(t, dst);
    return 1;
}

int nc_copy_from(nc_tensor *t, const void *src, size_t src_bytes) {
    if (!t || !src) return 0;
    size_t need = (size_t)nc_numel(t) * nc__dtype_size(t->dtype);
    if (!nc_is_contiguous(t) || src_bytes != need) {
        nc__error(t->ctx, NC_ERR_ARG, "nc_copy_from: need a contiguous tensor and exactly %zu bytes (got %zu)", need,
                  src_bytes);
        return 0;
    }
    memcpy(t->data, src, need);
    return 1;
}

/* ------------------------------------------------------------------------ */
/* Strided iteration                                                         */
/* ------------------------------------------------------------------------ */

void nc__iterate(int nops, float *const *base, const int64_t (*st)[NC_MAX_DIMS], int nd,
                 const int64_t *shape, nc_row_fn fn, const void *user) {
    float *p[4];
    int64_t s[4];
    if (nd == 0) {
        for (int k = 0; k < nops; k++) {
            p[k] = base[k];
            s[k] = 0;
        }
        fn(1, p, s, user);
        return;
    }
    for (int d = 0; d < nd; d++)
        if (shape[d] == 0) return;

    int64_t idx[NC_MAX_DIMS] = {0};
    for (int k = 0; k < nops; k++) s[k] = st[k][nd - 1];
    for (;;) {
        for (int k = 0; k < nops; k++) {
            int64_t off = 0;
            for (int d = 0; d < nd - 1; d++) off += idx[d] * st[k][d];
            p[k] = base[k] + off;
        }
        fn(shape[nd - 1], p, s, user);
        int d = nd - 2;
        while (d >= 0) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void nc__copy_strided(const nc_tensor *src, void *dst) {
    size_t es = nc__dtype_size(src->dtype);
    int64_t n = nc_numel(src);
    if (nc_is_contiguous(src)) {
        memcpy(dst, src->data, (size_t)n * es);
        return;
    }
    const unsigned char *s = (const unsigned char *)src->data;
    unsigned char *d = (unsigned char *)dst;
    int64_t idx[NC_MAX_DIMS] = {0};
    for (int64_t i = 0; i < n; i++) {
        int64_t off = 0;
        for (int k = 0; k < src->ndim; k++) off += idx[k] * src->stride[k];
        memcpy(d + (size_t)i * es, s + (size_t)off * es, es);
        for (int k = src->ndim - 1; k >= 0; k--) {
            if (++idx[k] < src->shape[k]) break;
            idx[k] = 0;
        }
    }
}

const float *nc__contig_f32(nc_tensor *t) {
    if (nc_is_contiguous(t)) return (const float *)t->data;
    float *buf = (float *)nc__arena_alloc(t->ctx, (size_t)nc_numel(t) * sizeof(float));
    if (buf) nc__copy_strided(t, buf);
    return buf;
}

nc_tensor *nc__grad_buf(nc_tensor *t) {
    if (t->grad && (t->flags & NC_TF_GRAD_CLEARED)) {
        memset(t->grad->data, 0, (size_t)nc_numel(t) * sizeof(float));
        t->flags &= ~NC_TF_GRAD_CLEARED;
    }
    if (!t->grad) {
        int persistent = (t->flags & NC_TF_PERSISTENT) != 0;
        t->ctx->grad_alloc = 1; /* not a backward temporary: it outlives the node */
        t->grad = nc__tensor_new(t->ctx, NC_F32, t->ndim, t->shape, persistent);
        t->ctx->grad_alloc = 0;
        if (t->grad) memset(t->grad->data, 0, (size_t)nc_numel(t) * sizeof(float));
    }
    return t->grad;
}

/* ------------------------------------------------------------------------ */
/* Printing                                                                  */
/* ------------------------------------------------------------------------ */

void nc_print(const nc_tensor *t) {
    if (!t) {
        printf("tensor(NULL)\n");
        return;
    }
    printf("tensor(%s%s%s, %s, shape=[", t->name ? "\"" : "", t->name ? t->name : "",
           t->name ? "\"" : "", nc__dtype_name(t->dtype));
    for (int i = 0; i < t->ndim; i++) printf(i ? ", %lld" : "%lld", (long long)t->shape[i]);
    printf("]%s)\n  ", nc__needs_grad(t) ? ", grad" : "");
    int64_t n = nc_numel(t), shown = n < 32 ? n : 32;
    int64_t idx[NC_MAX_DIMS] = {0};
    for (int64_t i = 0; i < shown; i++) {
        printf("%.5g ", nc_get(t, idx));
        for (int k = t->ndim - 1; k >= 0; k--) {
            if (++idx[k] < t->shape[k]) break;
            idx[k] = 0;
        }
    }
    printf(n > shown ? "... (%lld total)\n" : "\n", (long long)n);
}

/* ------------------------------------------------------------------------ */
/* Op registry                                                               */
/* ------------------------------------------------------------------------ */

#define NC_OP_PTR(name) &nc_op_##name,
static const nc_op_def *const op_table[] = {NC_OP_LIST(NC_OP_PTR)};
#undef NC_OP_PTR

int nc__op_count(void) { return (int)(sizeof op_table / sizeof op_table[0]); }
const nc_op_def *nc__op_at(int i) { return (i >= 0 && i < nc__op_count()) ? op_table[i] : NULL; }
const nc_op_def *nc__op_find(const char *name) {
    for (int i = 0; i < nc__op_count(); i++)
        if (strcmp(op_table[i]->name, name) == 0) return op_table[i];
    return NULL;
}
