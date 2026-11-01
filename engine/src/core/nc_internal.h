/* Internal definitions shared by every engine source file. Not installed. */
#ifndef NC_INTERNAL_H
#define NC_INTERNAL_H

#include "nc.h"
#include "../kernels/kernels.h"

#include <stdarg.h>
#include <string.h>

#define NC_ALIGN 64
#define NC_MAX_INPUTS 6
#define NC_DEFAULT_CHUNK (1u << 20)

/* ------------------------------------------------------------------------ */
/* Tensor                                                                    */
/* ------------------------------------------------------------------------ */

enum {
    NC_TF_REQUIRES_GRAD = 1u << 0,
    NC_TF_PERSISTENT = 1u << 1,
    NC_TF_OWNS_DATA = 1u << 2, /* false for views */
    /* grad buffer is kept for reuse but counts as absent (nc_clear_grad):
     * nc_grad() returns NULL and optimizers skip the tensor until backward
     * writes a fresh gradient into it */
    NC_TF_GRAD_CLEARED = 1u << 3
};

typedef struct nc_node nc_node;

struct nc_tensor {
    nc_ctx *ctx;
    nc_dtype dtype;
    int ndim;
    uint32_t flags;
    int64_t shape[NC_MAX_DIMS];
    int64_t stride[NC_MAX_DIMS]; /* in elements */
    void *data;                  /* points at logical element 0 */
    nc_tensor *grad;             /* contiguous, same logical shape */
    nc_node *node;               /* op that produced this tensor (NULL = leaf) */
    const char *name;
};

/* ------------------------------------------------------------------------ */
/* Op registry                                                               */
/* ------------------------------------------------------------------------ */

enum {
    NC_OPF_ELEMENTWISE = 1u << 0, /* candidate for fusion */
    NC_OPF_REDUCTION = 1u << 1,
    NC_OPF_VIEW = 1u << 2,        /* may alias its input */
    NC_OPF_NONDIFF = 1u << 3
};

typedef struct {
    int64_t i[NC_MAX_DIMS + 2];
    float f[4];
} nc_op_params;

typedef struct nc_op_def {
    const char *name;
    uint32_t flags;
    int kind; /* sub-kind shared by ops with a common implementation */
    /* Computes output dtype/shape from inputs and params. Must not read data,
     * so a future graph compiler can plan memory ahead of execution.
     * Returns NC_OK or an error code after calling nc__error(). */
    nc_status (*infer)(nc_node *n, nc_dtype *dt, int *ndim, int64_t *shape);
    /* Optional: make n->out alias an input instead of allocating and running
     * forward. Receives the inferred shape. Returns 1 if it did. */
    int (*view)(nc_node *n, int ndim, const int64_t *shape);
    nc_status (*forward)(nc_node *n);
    /* Accumulates into the grads of inputs that require grad. NULL = none. */
    void (*backward)(nc_node *n);
} nc_op_def;

struct nc_node {
    const nc_op_def *op;
    nc_tensor *in[NC_MAX_INPUTS];
    int n_in;
    nc_tensor *out;
    nc_op_params p;
    void *saved; /* op-private data, arena-allocated */
    nc_node *prev; /* tape: previously recorded node */
    uint32_t mark;
};

/* Every op, in one list. Used to declare op defs and build the registry. */
#define NC_OP_LIST(X)                                                          \
    X(add) X(sub) X(mul) X(div)                                                \
    X(neg) X(exp) X(log) X(sqrt) X(relu) X(sigmoid) X(tanh)                    \
    X(scale) X(add_scalar) X(pow_scalar)                                       \
    X(leaky_relu) X(relu6) X(hardsigmoid) X(hardswish) X(silu) X(gelu)         \
    X(gelu_tanh)                                                               \
    X(matmul) X(linear)                                                        \
    X(sum) X(mean) X(max) X(argmax)                                            \
    X(reshape) X(transpose) X(contiguous) X(detach) X(index_select)            \
    X(log_softmax) X(softmax) X(cross_entropy)                                 \
    X(conv2d) X(max_pool2d) X(avg_pool2d) X(dropout) X(batch_norm)             \
    X(linear_int8) X(nll_loss) X(compare) X(cat) X(to_dtype)

#define NC_DECLARE_OP(name) extern const nc_op_def nc_op_##name;
NC_OP_LIST(NC_DECLARE_OP)
#undef NC_DECLARE_OP

int nc__op_count(void);
const nc_op_def *nc__op_at(int i);
const nc_op_def *nc__op_find(const char *name);

/* Runs an op eagerly: validates, infers, allocates, executes and (if needed)
 * records it on the tape. The single entry point every op goes through. */
nc_tensor *nc__apply(const nc_op_def *op, nc_tensor **in, int n_in, const nc_op_params *p);

/* ------------------------------------------------------------------------ */
/* Context                                                                   */
/* ------------------------------------------------------------------------ */

typedef struct nc_chunk {
    struct nc_chunk *next;
    size_t cap;
    size_t used;
    unsigned char *base; /* aligned start of usable memory */
} nc_chunk;

typedef void (*nc_trace_fn)(nc_ctx *ctx, nc_node *n, void *user);

struct nc_ctx {
    /* step arena */
    nc_chunk *chunks;
    nc_chunk *cur;
    size_t chunk_bytes;
    size_t arena_in_use;
    size_t arena_reserved;
    size_t arena_peak;

    /* persistent allocations (freed with the context) */
    void *persistent_list;
    size_t persistent_bytes;

    int persistent_depth;
    int grad_enabled;
    nc_node *tape;
    uint32_t mark_epoch;

    nc_status status;
    char err[256];

    uint64_t rng[4];
    nc_cpu_info cpu;
    const nc_kernels *k;

    struct nc_pool *pool; /* worker threads, created on first parallel use */
    int n_threads;

    /* Hook for a future graph compiler: called after every executed op. */
    nc_trace_fn trace;
    void *trace_user;
};

void *nc__arena_alloc(nc_ctx *ctx, size_t bytes);
void *nc__persistent_alloc(nc_ctx *ctx, size_t bytes);
void nc__persistent_free(nc_ctx *ctx, void *p);
/* Allocates in the arena or persistently, following ctx->persistent_depth. */
void *nc__alloc(nc_ctx *ctx, size_t bytes, int persistent);

void nc__error(nc_ctx *ctx, nc_status code, const char *fmt, ...);

float nc__rand_uniform01(nc_ctx *ctx);
float nc__rand_normal(nc_ctx *ctx);

void nc__detect_cpu(nc_cpu_info *info);

/* ------------------------------------------------------------------------ */
/* Threads                                                                   */
/* ------------------------------------------------------------------------ */

/* Runs fn over [0, n) split into chunks of at least `grain` items, on the
 * context's pool plus the calling thread. Blocks until all chunks finish. */
typedef void (*nc_range_fn)(void *arg, int64_t begin, int64_t end);
void nc__parallel_for(nc_ctx *ctx, int64_t n, int64_t grain, nc_range_fn fn, void *arg);
void nc__pool_destroy(struct nc_pool *pool);

/* GEMM entry point for ops: same contract as nc_kernels.gemm_f32, split
 * across threads when the problem is large enough. Results do not depend on
 * the thread count (each output element is computed by one thread). */
void nc__gemm(nc_ctx *ctx, int ta, int tb, int64_t M, int64_t N, int64_t K, float alpha, const float *A,
              int64_t lda, const float *B, int64_t ldb, float beta, float *C, int64_t ldc);

/* ------------------------------------------------------------------------ */
/* Tensor helpers                                                            */
/* ------------------------------------------------------------------------ */

size_t nc__dtype_size(nc_dtype dt);
const char *nc__dtype_name(nc_dtype dt);
int64_t nc__shape_numel(int ndim, const int64_t *shape);
void nc__contig_strides(int ndim, const int64_t *shape, int64_t *stride);
int nc__norm_axis(int axis, int ndim); /* -1 if out of range */
int nc__same_shape(const nc_tensor *a, const nc_tensor *b);
/* Formats "[2, 3]" into buf; returns buf. */
const char *nc__shape_str(int ndim, const int64_t *shape, char *buf, size_t size);
/* Shared infer checks. Return 0 after setting an error. */
int nc__expect_f32(const nc_node *n, int i);

nc_tensor *nc__tensor_new(nc_ctx *ctx, nc_dtype dt, int ndim, const int64_t *shape, int persistent);
/* Shallow copy of the header: shares data, no grad, no node. */
nc_tensor *nc__tensor_alias(const nc_tensor *src, int persistent);
int nc__is_persistent_mode(const nc_ctx *ctx);

/* Gradient buffer of t, created zero-filled on first use. */
nc_tensor *nc__grad_buf(nc_tensor *t);
static inline int nc__needs_grad(const nc_tensor *t) {
    return (t->flags & NC_TF_REQUIRES_GRAD) != 0;
}

/* Returns a pointer to contiguous f32 data of t, copying into the arena if
 * t is a strided view. */
const float *nc__contig_f32(nc_tensor *t);

/* Strides of `t` broadcast onto an `ndim`-d output shape (0 = broadcast). */
void nc__bcast_strides(const nc_tensor *t, int ndim, const int64_t *shape, int64_t *out);
/* Same, for a contiguous buffer of shape (tn, tshape). */
void nc__bcast_strides_shape(int tn, const int64_t *tshape, int ndim, const int64_t *shape,
                             int64_t *out);

/* N-d strided iteration over up to 4 f32 operands. `fn` is called once per
 * innermost row with pointers and strides for that row. */
typedef void (*nc_row_fn)(int64_t n, float *const *p, const int64_t *s, const void *user);
void nc__iterate(int nops, float *const *base, const int64_t (*strides)[NC_MAX_DIMS], int ndim,
                 const int64_t *shape, nc_row_fn fn, const void *user);

/* Copies a (possibly strided) tensor into contiguous dst, any dtype. */
void nc__copy_strided(const nc_tensor *src, void *dst);

#endif /* NC_INTERNAL_H */
