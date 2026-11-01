/*
 * newch (nc) - a small, portable machine-learning engine in C11.
 *
 * Layers (bottom to top):
 *   L1 kernels   : portable C reference + per-ISA kernels, chosen at runtime
 *   L3 eager     : tensors, op registry, tape-based autograd
 *   L4 library   : nn modules, optimizers
 *   (L2 graph compiler, L5 high-level fit/predict, ml estimators: planned)
 *
 * Memory model:
 *   - persistent tensors (parameters, optimizer state) live until nc_free_ctx()
 *   - temporary tensors (activations, intermediate grads) live in the step
 *     arena and are all released at once by nc_step_end()
 *
 * Error model:
 *   Ops never abort. On error they record a message in the context and return
 *   NULL; every op accepts NULL inputs and propagates NULL, so a whole forward
 *   pass can be checked once at the end with nc_ok() / nc_error().
 */
#ifndef NC_H
#define NC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Only functions marked NC_API are exported from the shared library. */
#if defined(_WIN32) && defined(NC_BUILD_SHARED)
#define NC_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define NC_API __attribute__((visibility("default")))
#else
#define NC_API
#endif

#define NC_VERSION_MAJOR 0
#define NC_VERSION_MINOR 3
#define NC_VERSION_PATCH 0

#define NC_MAX_DIMS 6
#define NC_ALL_AXES (-128) /* reduce over every axis */

typedef enum {
    NC_F32 = 0,
    NC_I32 = 1,
    NC_F16 = 2, /* reserved: storage format for later fp16 kernels */
    NC_I8 = 3,  /* quantized weights */
    NC_DTYPE_COUNT
} nc_dtype;

typedef enum {
    NC_DEVICE_CPU = 0
    /* NC_DEVICE_VULKAN, NC_DEVICE_NPU: planned backends */
} nc_device;

typedef enum {
    NC_OK = 0,
    NC_ERR_SHAPE = 1,
    NC_ERR_DTYPE = 2,
    NC_ERR_OOM = 3,
    NC_ERR_ARG = 4,
    NC_ERR_STATE = 5
} nc_status;

typedef struct nc_ctx nc_ctx;
typedef struct nc_tensor nc_tensor;
typedef struct nc_module nc_module;
typedef struct nc_optim nc_optim;

/* ------------------------------------------------------------------------ */
/* Context                                                                   */
/* ------------------------------------------------------------------------ */

typedef struct {
    size_t arena_chunk_bytes; /* step-arena growth unit; 0 = 1 MiB */
    uint64_t seed;            /* RNG seed; 0 = fixed default */
    int force_reference;      /* 1 = ignore SIMD kernels (debugging) */
    int threads;              /* worker threads incl. caller; 0 = all cores, 1 = single */
} nc_config;

NC_API nc_ctx *nc_init(const nc_config *cfg); /* cfg may be NULL */
NC_API void nc_free_ctx(nc_ctx *ctx);
NC_API void nc_manual_seed(nc_ctx *ctx, uint64_t seed); /* reseeds the RNG only */

/* Releases every temporary tensor and the autograd tape. O(1). */
NC_API void nc_step_end(nc_ctx *ctx);
/* Returns unused arena chunks to the OS (call after nc_step_end, e.g. after
 * an unusually large step). The next step re-grows the arena as needed. */
NC_API void nc_arena_trim(nc_ctx *ctx);
NC_API int nc_num_threads(const nc_ctx *ctx);

/* Ops executed between begin/end produce persistent tensors and are never
 * recorded for autograd (use for weight init, preprocessing constants). */
NC_API void nc_persistent_begin(nc_ctx *ctx);
NC_API void nc_persistent_end(nc_ctx *ctx);

/* Autograd recording switch (like torch.no_grad). Returns previous value. */
NC_API int nc_set_grad_enabled(nc_ctx *ctx, int enabled);

NC_API int nc_ok(const nc_ctx *ctx);
NC_API nc_status nc_status_code(const nc_ctx *ctx);
NC_API const char *nc_error(const nc_ctx *ctx);
NC_API void nc_clear_error(nc_ctx *ctx);

typedef struct {
    size_t persistent_bytes;
    size_t arena_bytes_in_use;
    size_t arena_bytes_reserved;
    size_t arena_peak_bytes; /* high-water mark since nc_init / reset */
} nc_mem_stats;

NC_API nc_mem_stats nc_get_mem_stats(const nc_ctx *ctx);
NC_API void nc_reset_peak(nc_ctx *ctx);

/* ------------------------------------------------------------------------ */
/* Platform                                                                  */
/* ------------------------------------------------------------------------ */

typedef struct {
    const char *arch; /* "aarch64", "armv7", "x86_64", "riscv64", ... */
    int has_neon;
    int has_dotprod;
    int has_fp16;
    int has_avx2;
    int has_fma;
    int has_rvv;
    int n_cores;
    int n_big_cores; /* cores at max capacity (big.LITTLE); == n_cores if uniform */
} nc_cpu_info;

NC_API const nc_cpu_info *nc_cpu(const nc_ctx *ctx);
NC_API const char *nc_kernel_backend(const nc_ctx *ctx); /* e.g. "ref", "neon" */

/* ------------------------------------------------------------------------ */
/* Tensors                                                                   */
/* ------------------------------------------------------------------------ */

NC_API nc_tensor *nc_empty(nc_ctx *ctx, nc_dtype dt, int ndim, const int64_t *shape);
NC_API nc_tensor *nc_zeros(nc_ctx *ctx, nc_dtype dt, int ndim, const int64_t *shape);
NC_API nc_tensor *nc_full(nc_ctx *ctx, float value, int ndim, const int64_t *shape);
NC_API nc_tensor *nc_randn(nc_ctx *ctx, float stddev, int ndim, const int64_t *shape);
NC_API nc_tensor *nc_rand_uniform(nc_ctx *ctx, float lo, float hi, int ndim, const int64_t *shape);
/* Copies `data` (contiguous, row-major) into a new tensor. */
NC_API nc_tensor *nc_from_data(nc_ctx *ctx, nc_dtype dt, const void *data, int ndim, const int64_t *shape);
NC_API nc_tensor *nc_scalar(nc_ctx *ctx, float value);

/* Trainable leaf: persistent, requires_grad = 1. Takes values from `init`
 * (any tensor, copied) or zeros if init == NULL. */
NC_API nc_tensor *nc_parameter(nc_ctx *ctx, const nc_tensor *init, int ndim, const int64_t *shape);

NC_API int nc_ndim(const nc_tensor *t);
NC_API int64_t nc_dim(const nc_tensor *t, int axis); /* negative axis allowed */
NC_API const int64_t *nc_shape(const nc_tensor *t);
NC_API int64_t nc_numel(const nc_tensor *t);
NC_API nc_dtype nc_get_dtype(const nc_tensor *t);
NC_API int nc_is_contiguous(const nc_tensor *t);
NC_API int nc_requires_grad(const nc_tensor *t);
NC_API int nc_is_persistent(const nc_tensor *t);
/* Frees a persistent tensor and its grad now instead of at nc_free_ctx().
 * Nothing (views, modules, optimizers) may still reference it. */
NC_API void nc_release(nc_tensor *t);
NC_API void nc_set_requires_grad(nc_tensor *t, int on);
NC_API nc_tensor *nc_grad(const nc_tensor *t); /* NULL until backward reaches it */
NC_API void nc_set_name(nc_tensor *t, const char *name);
NC_API const char *nc_name(const nc_tensor *t);

/* Raw data access. Only valid for contiguous tensors (else NULL). */
NC_API float *nc_data_f32(nc_tensor *t);
NC_API int32_t *nc_data_i32(nc_tensor *t);
NC_API float nc_item(const nc_tensor *t); /* value of a 1-element f32 tensor */
NC_API float nc_get(const nc_tensor *t, const int64_t *index);
NC_API int nc_copy_to(const nc_tensor *t, void *dst, size_t dst_bytes);
/* Overwrites a contiguous tensor's values (e.g. loading weights). */
NC_API int nc_copy_from(nc_tensor *t, const void *src, size_t src_bytes);

NC_API void nc_print(const nc_tensor *t);

/* ------------------------------------------------------------------------ */
/* Ops (all differentiable unless noted)                                     */
/* ------------------------------------------------------------------------ */

/* Elementwise binary, numpy-style broadcasting. */
NC_API nc_tensor *nc_add(nc_tensor *a, nc_tensor *b);
NC_API nc_tensor *nc_sub(nc_tensor *a, nc_tensor *b);
NC_API nc_tensor *nc_mul(nc_tensor *a, nc_tensor *b);
NC_API nc_tensor *nc_div(nc_tensor *a, nc_tensor *b);

/* Elementwise unary. */
NC_API nc_tensor *nc_neg(nc_tensor *x);
NC_API nc_tensor *nc_exp(nc_tensor *x);
NC_API nc_tensor *nc_log(nc_tensor *x);
NC_API nc_tensor *nc_sqrt(nc_tensor *x);
NC_API nc_tensor *nc_relu(nc_tensor *x);
NC_API nc_tensor *nc_sigmoid(nc_tensor *x);
NC_API nc_tensor *nc_tanh(nc_tensor *x);
NC_API nc_tensor *nc_scale(nc_tensor *x, float s);      /* x * s */
NC_API nc_tensor *nc_add_scalar(nc_tensor *x, float s); /* x + s */
NC_API nc_tensor *nc_pow_scalar(nc_tensor *x, float p); /* x ^ p */

/* Activations (PyTorch definitions; gradients at kinks follow PyTorch). */
NC_API nc_tensor *nc_leaky_relu(nc_tensor *x, float negative_slope); /* x > 0 ? x : slope * x */
NC_API nc_tensor *nc_relu6(nc_tensor *x);       /* min(max(x, 0), 6) */
NC_API nc_tensor *nc_hardsigmoid(nc_tensor *x); /* min(max(x / 6 + 1/2, 0), 1) */
NC_API nc_tensor *nc_hardswish(nc_tensor *x);   /* x * relu6(x + 3) / 6 */
NC_API nc_tensor *nc_silu(nc_tensor *x);        /* x * sigmoid(x), a.k.a. swish */
/* x * Phi(x) with the exact erf form, or the tanh approximation when
 * approximate_tanh != 0 */
NC_API nc_tensor *nc_gelu(nc_tensor *x, int approximate_tanh);

/* Linear algebra. a: [..., M, K] (2-D for now), b: [K, N]. */
NC_API nc_tensor *nc_matmul(nc_tensor *a, nc_tensor *b);
/* y = x @ weight^T + bias. x [..., in], weight [out, in] (PyTorch layout),
 * bias [out] or NULL. Fused: one GEMM per pass, no transposed copies. */
NC_API nc_tensor *nc_linear_fn(nc_tensor *x, nc_tensor *weight, nc_tensor *bias);

/* Reductions. axis may be NC_ALL_AXES. */
NC_API nc_tensor *nc_sum(nc_tensor *x, int axis, int keepdim);
NC_API nc_tensor *nc_mean(nc_tensor *x, int axis, int keepdim);
NC_API nc_tensor *nc_max(nc_tensor *x, int axis, int keepdim); /* grad to first argmax */

/* Shape ops. reshape/transpose return views when possible. */
NC_API nc_tensor *nc_reshape(nc_tensor *x, int ndim, const int64_t *shape); /* one dim may be -1 */
NC_API nc_tensor *nc_transpose(nc_tensor *x, int axis0, int axis1);
NC_API nc_tensor *nc_contiguous(nc_tensor *x);
/* Rows of x along axis 0: x [N, ...], idx i32 [B] -> [B, ...].
 * Batch gathering and embedding lookup; grads scatter-add back into x. */
NC_API nc_tensor *nc_index_select(nc_tensor *x, nc_tensor *idx);

/* Convolution / pooling on NCHW tensors. stride and padding are per axis
 * (h, w); padding is zero for conv/avg and -inf for max. Avg pooling counts
 * padded cells (PyTorch's count_include_pad=True). */
NC_API nc_tensor *nc_conv2d(nc_tensor *x, nc_tensor *weight, nc_tensor *bias, int stride_h, int stride_w,
                     int pad_h, int pad_w);
NC_API nc_tensor *nc_max_pool2d(nc_tensor *x, int k_h, int k_w, int stride_h, int stride_w, int pad_h, int pad_w);
NC_API nc_tensor *nc_avg_pool2d(nc_tensor *x, int k_h, int k_w, int stride_h, int stride_w, int pad_h, int pad_w);

/* Inverted dropout: zeroes with probability p and scales the rest by
 * 1/(1-p). Always active; callers skip it in eval mode. */
NC_API nc_tensor *nc_dropout(nc_tensor *x, float p);

/* x [N, C, ...]. training=1 normalizes with batch statistics and updates the
 * running buffers in place (unbiased variance, like PyTorch); training=0 uses
 * the running buffers. */
NC_API nc_tensor *nc_batch_norm(nc_tensor *x, nc_tensor *weight, nc_tensor *bias, nc_tensor *running_mean,
                         nc_tensor *running_var, int training, float momentum, float eps);

/* NN functional. */
NC_API nc_tensor *nc_log_softmax(nc_tensor *x); /* over the last axis */
NC_API nc_tensor *nc_softmax(nc_tensor *x);     /* over the last axis */
/* logits: [N, C] f32, labels: [N] i32. Returns mean loss, shape []. */
NC_API nc_tensor *nc_cross_entropy(nc_tensor *logits, nc_tensor *labels);
NC_API nc_tensor *nc_mse_loss(nc_tensor *pred, nc_tensor *target);

/* Non-differentiable helpers. */
NC_API nc_tensor *nc_argmax(nc_tensor *x, int axis); /* i32, axis removed */
NC_API nc_tensor *nc_detach(nc_tensor *x);           /* shares data, no grad */

/* ------------------------------------------------------------------------ */
/* Quantization (inference only)                                             */
/* ------------------------------------------------------------------------ */

/* Symmetric per-row int8 quantization of w [rows, cols]: returns the i8
 * tensor and writes the f32 [rows] scales to *scale (w ~= q * scale). */
NC_API nc_tensor *nc_quantize_rows(nc_tensor *w, nc_tensor **scale);
/* y = x @ dequant(wq)^T + bias with x quantized per row at runtime (dynamic
 * quantization). x [..., in] f32, wq [out, in] i8, w_scale [out] f32. */
NC_API nc_tensor *nc_linear_int8(nc_tensor *x, nc_tensor *wq, nc_tensor *w_scale, nc_tensor *bias);

/* ------------------------------------------------------------------------ */
/* Serialization                                                             */
/* ------------------------------------------------------------------------ */

/* Named-tensor file ("NCW1"): little-endian, see docs/FORMAT.md. */
NC_API nc_status nc_save(nc_ctx *ctx, const char *path, int n, const char *const *names, nc_tensor *const *tensors);
/* Fills existing tensors by name; shapes and dtypes must match the file. */
NC_API nc_status nc_load(nc_ctx *ctx, const char *path, int n, const char *const *names, nc_tensor *const *tensors);

/* ------------------------------------------------------------------------ */
/* Autograd                                                                  */
/* ------------------------------------------------------------------------ */

/* Backpropagates from a scalar (1-element) tensor. Gradients accumulate into
 * nc_grad() of every reachable tensor with requires_grad. */
NC_API nc_status nc_backward(nc_tensor *loss);
NC_API void nc_zero_grad(nc_tensor *t);
/* Marks the grad as absent (torch's zero_grad(set_to_none=True)): nc_grad()
 * returns NULL and optimizers skip t until a backward reaches it again. The
 * buffer is kept and reused, so there is no allocation per step. */
NC_API void nc_clear_grad(nc_tensor *t);

/* Numerical gradient check. `fn` builds a scalar from `inputs`; every input
 * with requires_grad is checked. Returns max relative error found. */
typedef nc_tensor *(*nc_scalar_fn)(nc_ctx *ctx, nc_tensor **inputs, void *user);
NC_API float nc_gradcheck(nc_ctx *ctx, nc_scalar_fn fn, nc_tensor **inputs, int n_inputs,
                   void *user, float eps);

/* ------------------------------------------------------------------------ */
/* NN modules                                                                */
/* ------------------------------------------------------------------------ */

typedef enum { NC_ACT_RELU, NC_ACT_SIGMOID, NC_ACT_TANH } nc_activation;

/* weight [out, in], bias [out]; init U(-1/sqrt(in), 1/sqrt(in)) as in PyTorch */
NC_API nc_module *nc_linear(nc_ctx *ctx, int64_t in_features, int64_t out_features, int bias);
NC_API nc_module *nc_act(nc_ctx *ctx, nc_activation kind);
/* Takes ownership of `mods`. */
NC_API nc_module *nc_sequential(nc_ctx *ctx, int n, nc_module **mods);

NC_API nc_tensor *nc_forward(nc_module *m, nc_tensor *x);
/* Writes up to `max` parameter pointers to `out`; returns total count. */
NC_API int nc_parameters(nc_module *m, nc_tensor **out, int max);
NC_API const char *nc_module_name(const nc_module *m);
NC_API void nc_set_training(nc_module *m, int training);

/* ------------------------------------------------------------------------ */
/* Optimizers                                                                */
/* ------------------------------------------------------------------------ */

typedef struct {
    float lr;
    float momentum;     /* SGD only */
    float weight_decay; /* SGD/Adam: L2 added to grad; AdamW: decoupled */
    float beta1, beta2, eps; /* Adam: 0 = defaults 0.9/0.999/1e-8; RMSprop uses beta2, eps */
} nc_optim_opts;

NC_API nc_optim *nc_sgd(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts);
NC_API nc_optim *nc_adam(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts);
/* Adam with decoupled weight decay (torch.optim.AdamW): w *= 1 - lr * wd. */
NC_API nc_optim *nc_adamw(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts);
/* RMSprop. opts.beta2 is the smoothing constant (0 = 0.99), opts.eps the
 * stabilizer (0 = 1e-8). eps_inside_sqrt = 0: w -= lr * g / (sqrt(v) + eps)
 * (PyTorch); 1: w -= lr * g / sqrt(v + eps) (Keras). */
NC_API nc_optim *nc_rmsprop(nc_ctx *ctx, nc_tensor **params, int n, nc_optim_opts opts, int eps_inside_sqrt);
NC_API void nc_optim_zero_grad(nc_optim *o);
/* Like nc_optim_zero_grad but with nc_clear_grad (set_to_none semantics). */
NC_API void nc_optim_clear_grad(nc_optim *o);
NC_API void nc_optim_step(nc_optim *o);
NC_API void nc_optim_set_lr(nc_optim *o, float lr);
/* Update every `steps` calls of nc_optim_step with the averaged gradient
 * (Keras' gradient_accumulation_steps). zero_grad keeps the running sum
 * until the update. Large effective batch at small-batch memory. */
NC_API void nc_optim_set_accumulation(nc_optim *o, int steps);
/* Scales all grads so their joint 2-norm is at most max_norm
 * (torch.nn.utils.clip_grad_norm_). Returns the norm before clipping. */
NC_API float nc_clip_grad_norm(nc_tensor **params, int n, float max_norm);

#ifdef __cplusplus
}
#endif

#endif /* NC_H */
