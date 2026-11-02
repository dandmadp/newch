/* Engine test suite. Uses internals (registry, kernel tables) on purpose. */
#include "../src/core/nc_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_checks, g_fails;

#define CHECK(cond)                                                        \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(cond)) {                                                     \
            g_fails++;                                                     \
            printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                    \
    do {                                                                                         \
        double a_ = (a), b_ = (b);                                                               \
        g_checks++;                                                                              \
        if (!(fabs(a_ - b_) <= (tol))) {                                                         \
            g_fails++;                                                                           \
            printf("    FAIL %s:%d: %s = %g, expected %g\n", __FILE__, __LINE__, #a, a_, b_);    \
        }                                                                                        \
    } while (0)

#define SHAPE(...) (int64_t[]){__VA_ARGS__}
#define NDIM(...) ((int)(sizeof((int64_t[]){__VA_ARGS__}) / sizeof(int64_t)))
#define T(ctx, data, ...) nc_from_data(ctx, NC_F32, data, NDIM(__VA_ARGS__), SHAPE(__VA_ARGS__))

static nc_ctx *new_ctx(void) {
    nc_config cfg = {0};
    cfg.seed = 1234;
    return nc_init(&cfg);
}

/* ------------------------------------------------------------------------ */

static void test_tensor_basics(void) {
    nc_ctx *ctx = new_ctx();
    float d[6] = {1, 2, 3, 4, 5, 6};
    nc_tensor *t = T(ctx, d, 2, 3);
    CHECK(t && nc_ndim(t) == 2 && nc_numel(t) == 6);
    CHECK(nc_dim(t, -1) == 3 && nc_is_contiguous(t));
    CHECK_NEAR(nc_get(t, SHAPE(1, 2)), 6, 0);
    nc_tensor *s = nc_scalar(ctx, 2.5f);
    CHECK(nc_ndim(s) == 0 && nc_numel(s) == 1);
    CHECK_NEAR(nc_item(s), 2.5, 0);
    nc_tensor *z = nc_zeros(ctx, NC_I32, 1, SHAPE(4));
    CHECK(nc_data_i32(z) && nc_data_i32(z)[3] == 0 && !nc_data_f32(z));
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_views(void) {
    nc_ctx *ctx = new_ctx();
    float d[6] = {1, 2, 3, 4, 5, 6};
    nc_tensor *t = T(ctx, d, 2, 3);
    nc_tensor *r = nc_reshape(t, 2, SHAPE(3, -1));
    CHECK(r && nc_dim(r, 0) == 3 && nc_dim(r, 1) == 2);
    CHECK(nc_data_f32(r) == nc_data_f32(t)); /* view, no copy */

    nc_tensor *tt = nc_transpose(t, 0, 1);
    CHECK(tt && !nc_is_contiguous(tt) && nc_dim(tt, 0) == 3);
    CHECK_NEAR(nc_get(tt, SHAPE(2, 1)), 6, 0);
    nc_tensor *c = nc_contiguous(tt);
    float want[6] = {1, 4, 2, 5, 3, 6};
    float *cd = nc_data_f32(c);
    CHECK(cd != NULL);
    for (int i = 0; cd && i < 6; i++) CHECK_NEAR(cd[i], want[i], 0);

    /* reshape of a non-contiguous view must copy */
    nc_tensor *r2 = nc_reshape(tt, 1, SHAPE(6));
    CHECK(r2 && nc_is_contiguous(r2) && nc_data_f32(r2) != nc_data_f32(t));
    CHECK_NEAR(nc_get(r2, SHAPE(1)), 4, 0);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_broadcast(void) {
    nc_ctx *ctx = new_ctx();
    float a[6] = {1, 2, 3, 4, 5, 6}, b[3] = {10, 20, 30}, c[2] = {2, 3};
    nc_tensor *y = nc_add(T(ctx, a, 2, 3), T(ctx, b, 3));
    CHECK(y && nc_dim(y, 0) == 2 && nc_dim(y, 1) == 3);
    CHECK_NEAR(nc_get(y, SHAPE(1, 2)), 36, 0);
    nc_tensor *z = nc_mul(T(ctx, c, 2, 1), T(ctx, b, 1, 3)); /* outer product */
    CHECK(z && nc_dim(z, 0) == 2 && nc_dim(z, 1) == 3);
    CHECK_NEAR(nc_get(z, SHAPE(1, 1)), 60, 0);
    nc_tensor *w = nc_sub(nc_scalar(ctx, 1.0f), T(ctx, b, 3));
    CHECK_NEAR(nc_get(w, SHAPE(2)), -29, 0);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_matmul_values(void) {
    nc_ctx *ctx = new_ctx();
    float a[6] = {1, 2, 3, 4, 5, 6}, b[6] = {7, 8, 9, 10, 11, 12};
    nc_tensor *A = T(ctx, a, 2, 3), *B = T(ctx, b, 3, 2);
    nc_tensor *C = nc_matmul(A, B);
    CHECK(C && nc_dim(C, 0) == 2 && nc_dim(C, 1) == 2);
    CHECK_NEAR(nc_get(C, SHAPE(0, 0)), 58, 1e-5);
    CHECK_NEAR(nc_get(C, SHAPE(1, 1)), 154, 1e-5);
    /* A^T (3x2) @ A (2x3) through a transposed view */
    nc_tensor *G = nc_matmul(nc_transpose(A, 0, 1), A);
    CHECK(G && nc_dim(G, 0) == 3);
    CHECK_NEAR(nc_get(G, SHAPE(0, 0)), 17, 1e-5);
    CHECK_NEAR(nc_get(G, SHAPE(2, 1)), 3 * 2 + 6 * 5, 1e-5);
    /* batched a: [2, 1, 3] @ [3, 2] */
    nc_tensor *Cb = nc_matmul(nc_reshape(A, 3, SHAPE(2, 1, 3)), B);
    CHECK(Cb && nc_ndim(Cb) == 3 && nc_dim(Cb, 2) == 2);
    CHECK_NEAR(nc_get(Cb, SHAPE(1, 0, 1)), 154, 1e-5);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_reduce_values(void) {
    nc_ctx *ctx = new_ctx();
    float a[6] = {1, 5, 3, 4, 2, 6};
    nc_tensor *A = T(ctx, a, 2, 3);
    nc_tensor *s0 = nc_sum(A, 0, 0), *s1 = nc_sum(A, 1, 1), *m = nc_mean(A, NC_ALL_AXES, 0);
    CHECK(nc_ndim(s0) == 1 && nc_dim(s0, 0) == 3);
    CHECK_NEAR(nc_get(s0, SHAPE(1)), 7, 0);
    CHECK(nc_ndim(s1) == 2 && nc_dim(s1, 1) == 1);
    CHECK_NEAR(nc_get(s1, SHAPE(1, 0)), 12, 0);
    CHECK_NEAR(nc_item(m), 3.5, 1e-6);
    nc_tensor *mx = nc_max(A, 1, 0), *am = nc_argmax(A, 1);
    CHECK_NEAR(nc_get(mx, SHAPE(0)), 5, 0);
    CHECK(nc_get_dtype(am) == NC_I32);
    CHECK_NEAR(nc_get(am, SHAPE(1)), 2, 0);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_errors(void) {
    nc_ctx *ctx = new_ctx();
    nc_tensor *a = nc_zeros(ctx, NC_F32, 2, SHAPE(2, 3)), *b = nc_zeros(ctx, NC_F32, 2, SHAPE(4, 3));
    nc_tensor *y = nc_add(a, b);
    CHECK(y == NULL && !nc_ok(ctx) && nc_status_code(ctx) == NC_ERR_SHAPE);
    CHECK(strstr(nc_error(ctx), "[2, 3]") != NULL);
    /* NULL propagates through a whole chain without crashing */
    nc_tensor *z = nc_relu(nc_matmul(y, a));
    CHECK(z == NULL);
    nc_clear_error(ctx);
    CHECK(nc_ok(ctx));
    CHECK(nc_matmul(a, a) == NULL && nc_status_code(ctx) == NC_ERR_SHAPE);
    nc_clear_error(ctx);
    int32_t bad[2] = {0, 7};
    nc_tensor *lab = nc_from_data(ctx, NC_I32, bad, 1, SHAPE(2));
    CHECK(nc_cross_entropy(a, lab) == NULL && nc_status_code(ctx) == NC_ERR_ARG);
    nc_free_ctx(ctx);
}

/* ------------------------------------------------------------------------ */
/* Gradient checks: loss = sum(op(x)^2) (or op(x) itself if scalar)          */
/* ------------------------------------------------------------------------ */

typedef nc_tensor *(*op_fn)(nc_tensor **in, nc_tensor *aux);

typedef struct {
    op_fn f;
    nc_tensor *aux; /* non-differentiable extra input (labels, ...) */
} gc_case;

static nc_tensor *gc_loss(nc_ctx *ctx, nc_tensor **in, void *user) {
    (void)ctx;
    gc_case *c = (gc_case *)user;
    nc_tensor *y = c->f(in, c->aux);
    if (y && nc_numel(y) == 1) return nc_reshape(y, 0, NULL);
    return nc_sum(nc_mul(y, y), NC_ALL_AXES, 0);
}

static nc_tensor *f_add(nc_tensor **x, nc_tensor *a) { (void)a; return nc_add(x[0], x[1]); }
static nc_tensor *f_sub(nc_tensor **x, nc_tensor *a) { (void)a; return nc_sub(x[0], x[1]); }
static nc_tensor *f_mul(nc_tensor **x, nc_tensor *a) { (void)a; return nc_mul(x[0], x[1]); }
static nc_tensor *f_div(nc_tensor **x, nc_tensor *a) { (void)a; return nc_div(x[0], x[1]); }
static nc_tensor *f_square(nc_tensor **x, nc_tensor *a) { (void)a; return nc_mul(x[0], x[0]); }
static nc_tensor *f_neg(nc_tensor **x, nc_tensor *a) { (void)a; return nc_neg(x[0]); }
static nc_tensor *f_exp(nc_tensor **x, nc_tensor *a) { (void)a; return nc_exp(x[0]); }
static nc_tensor *f_log(nc_tensor **x, nc_tensor *a) { (void)a; return nc_log(x[0]); }
static nc_tensor *f_sqrt(nc_tensor **x, nc_tensor *a) { (void)a; return nc_sqrt(x[0]); }
static nc_tensor *f_relu(nc_tensor **x, nc_tensor *a) { (void)a; return nc_relu(x[0]); }
static nc_tensor *f_sigmoid(nc_tensor **x, nc_tensor *a) { (void)a; return nc_sigmoid(x[0]); }
static nc_tensor *f_tanh(nc_tensor **x, nc_tensor *a) { (void)a; return nc_tanh(x[0]); }
static nc_tensor *f_scale(nc_tensor **x, nc_tensor *a) { (void)a; return nc_scale(x[0], -1.5f); }
static nc_tensor *f_adds(nc_tensor **x, nc_tensor *a) { (void)a; return nc_add_scalar(x[0], 0.7f); }
static nc_tensor *f_pow(nc_tensor **x, nc_tensor *a) { (void)a; return nc_pow_scalar(x[0], 1.5f); }
static nc_tensor *f_leaky(nc_tensor **x, nc_tensor *a) { (void)a; return nc_leaky_relu(x[0], 0.1f); }
static nc_tensor *f_relu6(nc_tensor **x, nc_tensor *a) { (void)a; return nc_relu6(x[0]); }
static nc_tensor *f_hsig(nc_tensor **x, nc_tensor *a) { (void)a; return nc_hardsigmoid(x[0]); }
static nc_tensor *f_hswish(nc_tensor **x, nc_tensor *a) { (void)a; return nc_hardswish(x[0]); }
static nc_tensor *f_silu(nc_tensor **x, nc_tensor *a) { (void)a; return nc_silu(x[0]); }
static nc_tensor *f_gelu(nc_tensor **x, nc_tensor *a) { (void)a; return nc_gelu(x[0], 0); }
static nc_tensor *f_gelu_t(nc_tensor **x, nc_tensor *a) { (void)a; return nc_gelu(x[0], 1); }
static nc_tensor *f_matmul(nc_tensor **x, nc_tensor *a) { (void)a; return nc_matmul(x[0], x[1]); }
static nc_tensor *f_matmul_tt(nc_tensor **x, nc_tensor *a) {
    (void)a; /* both operands as transposed views */
    return nc_matmul(nc_transpose(x[0], 0, 1), nc_transpose(x[1], 0, 1));
}
static nc_tensor *f_matmul_3d(nc_tensor **x, nc_tensor *a) {
    (void)a;
    return nc_matmul(nc_reshape(x[0], 3, SHAPE(2, 2, 3)), x[1]);
}
static nc_tensor *f_linear(nc_tensor **x, nc_tensor *a) { (void)a; return nc_linear_fn(x[0], x[1], x[2]); }
static nc_tensor *f_linear_nb(nc_tensor **x, nc_tensor *a) { (void)a; return nc_linear_fn(x[0], x[1], NULL); }
static nc_tensor *f_sum0(nc_tensor **x, nc_tensor *a) { (void)a; return nc_sum(x[0], 0, 0); }
static nc_tensor *f_sum1k(nc_tensor **x, nc_tensor *a) { (void)a; return nc_sum(x[0], -1, 1); }
static nc_tensor *f_mean(nc_tensor **x, nc_tensor *a) { (void)a; return nc_mean(x[0], NC_ALL_AXES, 0); }
static nc_tensor *f_max(nc_tensor **x, nc_tensor *a) { (void)a; return nc_max(x[0], 1, 0); }
static nc_tensor *f_reshape(nc_tensor **x, nc_tensor *a) {
    (void)a;
    return nc_mul(nc_reshape(x[0], 2, SHAPE(3, 4)), nc_reshape(x[0], 2, SHAPE(3, 4)));
}
static nc_tensor *f_transpose(nc_tensor **x, nc_tensor *a) {
    (void)a;
    return nc_add(nc_transpose(x[0], 0, 1), x[1]);
}
static nc_tensor *f_contig(nc_tensor **x, nc_tensor *a) {
    (void)a;
    return nc_reshape(nc_contiguous(nc_transpose(x[0], 0, 1)), 1, SHAPE(-1));
}
static nc_tensor *f_index(nc_tensor **x, nc_tensor *a) { return nc_index_select(x[0], a); }
static nc_tensor *f_logsm(nc_tensor **x, nc_tensor *a) { (void)a; return nc_log_softmax(x[0]); }
static nc_tensor *f_sm(nc_tensor **x, nc_tensor *a) { (void)a; return nc_softmax(x[0]); }
static nc_tensor *f_ce(nc_tensor **x, nc_tensor *a) { return nc_cross_entropy(x[0], a); }
static nc_tensor *f_nll(nc_tensor **x, nc_tensor *a) { return nc_nll_loss(nc_log_softmax(x[0]), a, 0); }
static nc_tensor *f_nll_sum(nc_tensor **x, nc_tensor *a) { return nc_nll_loss(nc_log_softmax(x[0]), a, 1); }
static nc_tensor *f_cat(nc_tensor **x, nc_tensor *a) {
    (void)a;
    nc_tensor *parts[3] = {x[0], nc_transpose(x[1], 0, 1), x[0]}; /* strided input too */
    return nc_cat(3, parts, 1);
}
static nc_tensor *f_to_f32(nc_tensor **x, nc_tensor *a) { (void)a; return nc_to_dtype(x[0], NC_F32); }
static nc_tensor *f_mse(nc_tensor **x, nc_tensor *a) { (void)a; return nc_mse_loss(x[0], x[1]); }
static nc_tensor *f_mlp(nc_tensor **x, nc_tensor *a) {
    nc_tensor *h = nc_tanh(nc_add(nc_matmul(x[0], x[1]), x[2]));
    return nc_cross_entropy(nc_matmul(h, x[3]), a);
}

/* kind: 0 = randn, 1 = positive (0.5..2), 2 = randn pushed away from 0 */
static nc_tensor *gparam(nc_ctx *ctx, int kind, int ndim, const int64_t *shape) {
    nc_tensor *p = nc_parameter(ctx, NULL, ndim, shape);
    float *d = nc_data_f32(p);
    for (int64_t i = 0; i < nc_numel(p); i++) {
        float v = nc__rand_normal(ctx);
        if (kind == 1) v = 0.5f + 1.5f * nc__rand_uniform01(ctx);
        if (kind == 2) v = v >= 0 ? v + 0.2f : v - 0.2f;
        d[i] = v;
    }
    return p;
}

static void gradcheck_case(nc_ctx *ctx, const char *name, op_fn f, nc_tensor *aux, int n,
                           nc_tensor **in) {
    gc_case c = {f, aux};
    float err = nc_gradcheck(ctx, gc_loss, in, n, &c, 1e-2f);
    g_checks++;
    if (!(err < 2e-2f)) {
        g_fails++;
        printf("    FAIL gradcheck %-12s max err %g %s\n", name, err, nc_error(ctx));
        nc_clear_error(ctx);
    }
    nc_step_end(ctx);
}

#define GC1(name, f, kind, ...)                                                \
    do {                                                                       \
        nc_tensor *in_[1] = {gparam(ctx, kind, NDIM(__VA_ARGS__), SHAPE(__VA_ARGS__))}; \
        gradcheck_case(ctx, name, f, NULL, 1, in_);                            \
    } while (0)

static void test_gradcheck(void) {
    nc_ctx *ctx = new_ctx();
    nc_tensor *in[4];

    in[0] = gparam(ctx, 0, 2, SHAPE(2, 3)), in[1] = gparam(ctx, 0, 1, SHAPE(3));
    gradcheck_case(ctx, "add bcast", f_add, NULL, 2, in);
    in[0] = gparam(ctx, 0, 2, SHAPE(2, 1)), in[1] = gparam(ctx, 0, 2, SHAPE(1, 3));
    gradcheck_case(ctx, "sub bcast", f_sub, NULL, 2, in);
    gradcheck_case(ctx, "mul bcast", f_mul, NULL, 2, in);
    in[0] = gparam(ctx, 0, 2, SHAPE(2, 3)), in[1] = gparam(ctx, 1, 2, SHAPE(2, 3));
    gradcheck_case(ctx, "div", f_div, NULL, 2, in);
    in[1] = gparam(ctx, 1, 1, SHAPE(3));
    gradcheck_case(ctx, "div bcast", f_div, NULL, 2, in);
    GC1("x*x", f_square, 0, 2, 3);
    GC1("neg", f_neg, 0, 2, 3);
    GC1("exp", f_exp, 0, 2, 3);
    GC1("log", f_log, 1, 2, 3);
    GC1("sqrt", f_sqrt, 1, 2, 3);
    GC1("relu", f_relu, 2, 2, 3);
    GC1("sigmoid", f_sigmoid, 0, 2, 3);
    GC1("tanh", f_tanh, 0, 2, 3);
    GC1("scale", f_scale, 0, 2, 3);
    GC1("add_scalar", f_adds, 0, 2, 3);
    GC1("pow", f_pow, 1, 2, 3);
    {
        /* every linear piece of the activations, but no value within 0.05 of a kink */
        static const float act_in[12] = {-7.1f, -4.2f, -3.3f, -2.6f, -1.1f, -0.4f, 0.3f, 1.4f, 2.7f, 3.4f, 5.2f, 7.3f};
        nc_tensor *ai[1] = {nc_parameter(ctx, nc_from_data(ctx, NC_F32, act_in, 1, SHAPE(12)), 1, SHAPE(12))};
        gradcheck_case(ctx, "leaky_relu", f_leaky, NULL, 1, ai);
        gradcheck_case(ctx, "relu6", f_relu6, NULL, 1, ai);
        gradcheck_case(ctx, "hardsigmoid", f_hsig, NULL, 1, ai);
        gradcheck_case(ctx, "hardswish", f_hswish, NULL, 1, ai);
        gradcheck_case(ctx, "silu", f_silu, NULL, 1, ai);
        gradcheck_case(ctx, "gelu", f_gelu, NULL, 1, ai);
        gradcheck_case(ctx, "gelu tanh", f_gelu_t, NULL, 1, ai);
        /* known values */
        float v[3] = {-1.0f, 0.5f, 4.0f};
        nc_tensor *t = nc_from_data(ctx, NC_F32, v, 1, SHAPE(3));
        CHECK_NEAR(nc_get(nc_hardswish(t), SHAPE(1)), 0.5 * 3.5 / 6.0, 1e-6);
        CHECK_NEAR(nc_get(nc_relu6(t), SHAPE(2)), 4.0, 0);
        CHECK_NEAR(nc_get(nc_gelu(t, 0), SHAPE(0)), -0.15865525, 1e-6);
        CHECK_NEAR(nc_get(nc_gelu(t, 1), SHAPE(0)), -0.15880801, 1e-6);
        CHECK_NEAR(nc_get(nc_silu(t), SHAPE(1)), 0.5 / (1.0 + exp(-0.5)), 1e-6);
        nc_step_end(ctx);
    }

    in[0] = gparam(ctx, 0, 2, SHAPE(4, 3)), in[1] = gparam(ctx, 0, 2, SHAPE(3, 5));
    gradcheck_case(ctx, "matmul", f_matmul, NULL, 2, in);
    in[0] = gparam(ctx, 0, 2, SHAPE(3, 4)), in[1] = gparam(ctx, 0, 2, SHAPE(5, 3));
    gradcheck_case(ctx, "matmul T,T", f_matmul_tt, NULL, 2, in);
    in[0] = gparam(ctx, 0, 2, SHAPE(4, 3)), in[1] = gparam(ctx, 0, 2, SHAPE(3, 2));
    gradcheck_case(ctx, "matmul 3d", f_matmul_3d, NULL, 2, in);

    in[0] = gparam(ctx, 0, 2, SHAPE(4, 3)), in[1] = gparam(ctx, 0, 2, SHAPE(5, 3));
    in[2] = gparam(ctx, 0, 1, SHAPE(5));
    gradcheck_case(ctx, "linear", f_linear, NULL, 3, in);
    in[0] = gparam(ctx, 0, 3, SHAPE(2, 3, 3));
    gradcheck_case(ctx, "linear 3d", f_linear, NULL, 3, in);
    gradcheck_case(ctx, "linear nobias", f_linear_nb, NULL, 2, in);

    GC1("sum axis0", f_sum0, 0, 3, 4);
    GC1("sum keepdim", f_sum1k, 0, 2, 3, 4);
    GC1("mean all", f_mean, 0, 3, 4);
    GC1("max", f_max, 0, 3, 4);
    GC1("reshape", f_reshape, 0, 2, 6);
    in[0] = gparam(ctx, 0, 2, SHAPE(2, 3)), in[1] = gparam(ctx, 0, 2, SHAPE(3, 2));
    gradcheck_case(ctx, "transpose", f_transpose, NULL, 2, in);
    GC1("contiguous", f_contig, 0, 3, 4);
    GC1("log_softmax", f_logsm, 0, 3, 5);
    GC1("softmax", f_sm, 0, 3, 5);

    int32_t lab[4] = {0, 4, 2, 1};
    nc_persistent_begin(ctx);
    nc_tensor *labels = nc_from_data(ctx, NC_I32, lab, 1, SHAPE(4));
    nc_persistent_end(ctx);
    in[0] = gparam(ctx, 0, 2, SHAPE(4, 5));
    gradcheck_case(ctx, "cross_entropy", f_ce, labels, 1, in);
    int32_t rows[5] = {2, 0, 2, 3, 2}; /* repeats must accumulate */
    nc_persistent_begin(ctx);
    nc_tensor *ridx = nc_from_data(ctx, NC_I32, rows, 1, SHAPE(5));
    nc_persistent_end(ctx);
    in[0] = gparam(ctx, 0, 2, SHAPE(4, 3));
    gradcheck_case(ctx, "index_select", f_index, ridx, 1, in);
    in[0] = gparam(ctx, 0, 2, SHAPE(3, 2)), in[1] = gparam(ctx, 0, 2, SHAPE(3, 2));
    gradcheck_case(ctx, "mse_loss", f_mse, NULL, 2, in);
    in[0] = gparam(ctx, 0, 2, SHAPE(4, 5));
    gradcheck_case(ctx, "nll_loss", f_nll, labels, 1, in);
    gradcheck_case(ctx, "nll_loss sum", f_nll_sum, labels, 1, in);
    in[0] = gparam(ctx, 0, 2, SHAPE(2, 3)), in[1] = gparam(ctx, 0, 2, SHAPE(4, 2));
    gradcheck_case(ctx, "cat", f_cat, NULL, 2, in);
    gradcheck_case(ctx, "to_dtype f32", f_to_f32, NULL, 1, in);
    {
        /* nll_loss(log_softmax) == cross_entropy */
        nc_tensor *lg = gparam(ctx, 0, 2, SHAPE(4, 5));
        CHECK_NEAR(nc_item(nc_nll_loss(nc_log_softmax(lg), labels, 0)), nc_item(nc_cross_entropy(lg, labels)), 1e-5);
        /* compare: mixed f32 / i32 with broadcasting, then an integer sum */
        float fv[4] = {1.0f, 2.0f, 3.5f, 4.0f};
        int32_t iv[4] = {1, 0, 3, 4};
        nc_tensor *f = nc_from_data(ctx, NC_F32, fv, 1, SHAPE(4)), *i = nc_from_data(ctx, NC_I32, iv, 1, SHAPE(4));
        nc_tensor *eq = nc_compare(f, i, NC_CMP_EQ);
        CHECK(eq && nc_get_dtype(eq) == NC_I32 && nc_data_i32(eq)[0] == 1 && nc_data_i32(eq)[2] == 0);
        nc_tensor *cnt = nc_sum(eq, NC_ALL_AXES, 0);
        CHECK(cnt && nc_get_dtype(cnt) == NC_I32 && nc_data_i32(cnt)[0] == 2);
        nc_tensor *ge = nc_compare(nc_reshape(f, 2, SHAPE(4, 1)), i, NC_CMP_GE); /* [4, 4] */
        CHECK(ge && nc_dim(ge, 0) == 4 && nc_data_i32(nc_sum(ge, NC_ALL_AXES, 0))[0] == 11); /* 2 + 2 + 3 + 4 */
        /* to_dtype: truncation toward zero, int -> float, grad detached for ints */
        float tv[3] = {-1.7f, 2.9f, 0.4f};
        nc_tensor *ti = nc_to_dtype(nc_from_data(ctx, NC_F32, tv, 1, SHAPE(3)), NC_I32);
        CHECK(ti && nc_data_i32(ti)[0] == -1 && nc_data_i32(ti)[1] == 2);
        float big[4] = {3e9f, -3e9f, NAN, 300.0f};
        nc_tensor *bt = nc_from_data(ctx, NC_F32, big, 1, SHAPE(4));
        nc_tensor *b32 = nc_to_dtype(bt, NC_I32), *b8 = nc_to_dtype(bt, NC_I8);
        CHECK(nc_data_i32(b32)[0] == INT32_MAX && nc_data_i32(b32)[1] == INT32_MIN && nc_data_i32(b32)[2] == 0);
        CHECK(b8 && ((int8_t *)b8->data)[3] == 127 && ((int8_t *)b8->data)[1] == -128);
        CHECK(!nc_requires_grad(nc_to_dtype(lg, NC_I32)));
        CHECK_NEAR(nc_data_f32(nc_to_dtype(i, NC_F32))[3], 4.0, 0);
        /* cat of int tensors and the input-count limit */
        nc_tensor *two[2] = {i, i};
        CHECK(nc_dim(nc_cat(2, two, 0), 0) == 8);
        nc_tensor *seven[7] = {i, i, i, i, i, i, i};
        CHECK(!nc_cat(7, seven, 0) && nc_status_code(ctx) == NC_ERR_ARG);
        nc_clear_error(ctx);
        nc_step_end(ctx);
    }

    in[0] = gparam(ctx, 0, 2, SHAPE(4, 3)), in[1] = gparam(ctx, 0, 2, SHAPE(3, 6));
    in[2] = gparam(ctx, 0, 1, SHAPE(6)), in[3] = gparam(ctx, 0, 2, SHAPE(6, 5));
    gradcheck_case(ctx, "mlp chain", f_mlp, labels, 4, in);
    nc_free_ctx(ctx);
}

/* ------------------------------------------------------------------------ */

static void test_autograd_semantics(void) {
    nc_ctx *ctx = new_ctx();
    float wv[2] = {2, 3};
    nc_tensor *w = nc_parameter(ctx, T(ctx, wv, 2), 1, SHAPE(2));

    /* two losses in one step: grads add, the first is not re-propagated */
    nc_tensor *l1 = nc_sum(nc_mul(w, w), NC_ALL_AXES, 0); /* d = 2w */
    nc_tensor *l2 = nc_sum(nc_scale(w, 5.0f), NC_ALL_AXES, 0); /* d = 5 */
    CHECK(nc_backward(l1) == NC_OK);
    CHECK(nc_backward(l2) == NC_OK);
    CHECK_NEAR(nc_get(nc_grad(w), SHAPE(0)), 2 * 2 + 5, 1e-6);
    CHECK_NEAR(nc_get(nc_grad(w), SHAPE(1)), 2 * 3 + 5, 1e-6);
    nc_step_end(ctx);

    /* grads persist across steps until zeroed (accumulation) */
    CHECK(nc_backward(nc_sum(w, NC_ALL_AXES, 0)) == NC_OK);
    CHECK_NEAR(nc_get(nc_grad(w), SHAPE(0)), 10, 1e-6);
    nc_zero_grad(w);
    CHECK_NEAR(nc_get(nc_grad(w), SHAPE(0)), 0, 0);

    /* no_grad */
    int prev = nc_set_grad_enabled(ctx, 0);
    nc_tensor *y = nc_mul(w, w);
    CHECK(prev == 1 && y && !nc_requires_grad(y));
    CHECK(nc_backward(nc_sum(y, NC_ALL_AXES, 0)) == NC_ERR_STATE);
    nc_clear_error(ctx);
    nc_set_grad_enabled(ctx, 1);

    /* detach blocks gradient flow */
    nc_tensor *d = nc_detach(nc_mul(w, w));
    CHECK(!nc_requires_grad(d));
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_memory(void) {
    nc_ctx *ctx = new_ctx();
    nc_tensor *p = nc_parameter(ctx, NULL, 2, SHAPE(64, 64));
    size_t pers = nc_get_mem_stats(ctx).persistent_bytes;
    CHECK(pers >= 64 * 64 * 4);
    for (int step = 0; step < 3; step++) {
        nc_tensor *x = nc_randn(ctx, 1.0f, 2, SHAPE(32, 64));
        nc_tensor *l = nc_sum(nc_relu(nc_matmul(x, p)), NC_ALL_AXES, 0);
        CHECK(nc_backward(l) == NC_OK);
        CHECK(nc_get_mem_stats(ctx).arena_bytes_in_use > 0);
        nc_step_end(ctx);
        CHECK(nc_get_mem_stats(ctx).arena_bytes_in_use == 0);
    }
    nc_mem_stats s = nc_get_mem_stats(ctx);
    CHECK(s.arena_peak_bytes > 0 && s.arena_bytes_reserved >= s.arena_peak_bytes);
    size_t reserved = s.arena_bytes_reserved;
    nc_step_end(ctx);
    CHECK(nc_get_mem_stats(ctx).arena_bytes_reserved == reserved); /* chunks reused */
    CHECK(nc_grad(p) != NULL);                                       /* persistent grad survives */

    /* persistent mode: results survive nc_step_end */
    nc_persistent_begin(ctx);
    nc_tensor *c = nc_add_scalar(nc_full(ctx, 1.0f, 1, SHAPE(4)), 1.0f);
    nc_persistent_end(ctx);
    nc_tensor *junk = nc_full(ctx, 9.0f, 1, SHAPE(1024));
    (void)junk;
    nc_step_end(ctx);
    nc_full(ctx, 7.0f, 1, SHAPE(1024)); /* would overwrite c if it were in the arena */
    CHECK_NEAR(nc_get(c, SHAPE(3)), 2, 0);
    CHECK(nc_get_mem_stats(ctx).persistent_bytes > pers);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_backward_release(void) {
    nc_ctx *ctx = new_ctx();
    nc_tensor *w1 = nc_parameter(ctx, nc_randn(ctx, 0.5f, 2, SHAPE(64, 32)), 2, SHAPE(64, 32));
    nc_tensor *w2 = nc_parameter(ctx, nc_randn(ctx, 0.5f, 2, SHAPE(10, 64)), 2, SHAPE(10, 64));
    int32_t lab[256];
    for (int i = 0; i < 256; i++) lab[i] = i % 10;
    nc_tensor *grads_ref[2] = {NULL, NULL};
    size_t peak[3];
    for (int mode = 0; mode < 3; mode++) { /* retain, default, free graph */
        nc_zero_grad(w1), nc_zero_grad(w2);
        nc_step_end(ctx);
        nc_manual_seed(ctx, 5);
        nc_reset_peak(ctx);
        nc_tensor *x = nc_randn(ctx, 1.0f, 2, SHAPE(256, 32));
        nc_tensor *y = nc_from_data(ctx, NC_I32, lab, 1, SHAPE(256));
        nc_tensor *h = nc_relu(nc_linear_fn(x, w1, NULL));
        nc_tensor *loss = nc_cross_entropy(nc_linear_fn(nc_tanh(h), w2, NULL), y);
        int flags = mode == 0 ? NC_BACKWARD_RETAIN : mode == 1 ? 0 : NC_BACKWARD_FREE_GRAPH;
        CHECK(nc_backward_ex(loss, flags, &h, 1) == NC_OK);
        peak[mode] = nc_get_mem_stats(ctx).arena_peak_bytes;
        if (mode == 0) {
            grads_ref[0] = nc_from_data(ctx, NC_F32, nc_data_f32(nc_grad(w1)), 2, SHAPE(64, 32));
            nc_persistent_begin(ctx);
            grads_ref[0] = nc_contiguous(grads_ref[0]);
            grads_ref[1] = nc_from_data(ctx, NC_F32, nc_data_f32(nc_grad(w2)), 2, SHAPE(10, 64));
            nc_persistent_end(ctx);
            CHECK(nc_grad(h) != NULL); /* retained intermediate grad */
        } else {
            /* same leaf grads whatever is released */
            CHECK(memcmp(nc_data_f32(nc_grad(w1)), nc_data_f32(grads_ref[0]), 64 * 32 * 4) == 0);
            CHECK(memcmp(nc_data_f32(nc_grad(w2)), nc_data_f32(grads_ref[1]), 10 * 64 * 4) == 0);
            CHECK(nc_grad(h) == NULL); /* intermediate grads are released */
            CHECK(nc_data_f32(h) != NULL && isfinite(nc_data_f32(h)[0])); /* kept tensor survives */
            CHECK(isfinite(nc_item(loss)));                                /* so does the loss */
        }
        if (mode == 2) { /* a released graph cannot run backward again */
            CHECK(nc_backward(loss) == NC_ERR_STATE);
            nc_clear_error(ctx);
        }
    }
    printf("    backward peak: retain %.0f KiB, default %.0f KiB, free graph %.0f KiB\n", peak[0] / 1024.0,
           peak[1] / 1024.0, peak[2] / 1024.0);
    CHECK(peak[1] < peak[0] && peak[2] < peak[1]);
    nc_free_ctx(ctx);
}

static void test_inference_release(void) {
    nc_ctx *ctx = new_ctx();
    nc_tensor *w = nc_parameter(ctx, nc_randn(ctx, 0.1f, 4, SHAPE(16, 8, 3, 3)), 4, SHAPE(16, 8, 3, 3));
    size_t peak[2];
    float ref = 0;
    for (int mode = 0; mode < 2; mode++) { /* 0: keep everything (grad on), 1: no_grad + release */
        nc_step_end(ctx);
        nc_manual_seed(ctx, 3);
        nc_tensor *own = nc_randn(ctx, 1.0f, 4, SHAPE(32, 8, 28, 56));
        nc_reset_peak(ctx);
        int prev = nc_set_grad_enabled(ctx, mode == 0);
        for (int layer = 0; layer < 4; layer++) {
            nc_tensor *inp = nc_reshape(own, 4, SHAPE(32, 8, 28, 56)); /* a view of the previous output */
            nc_tensor *c = nc_conv2d(inp, nc_reshape(w, 4, SHAPE(16, 8, 3, 3)), NULL, 1, 2, 1, 1);
            nc_tensor *next = nc_relu(c); /* [32, 16, 28, 28] */
            if (mode == 1) {
                nc_release_temp(c);
                nc_release_temp(own);
            }
            own = next;
        }
        nc_tensor *h = own;
        float v = nc_item(nc_sum(h, NC_ALL_AXES, 0));
        if (mode == 0) ref = v;
        else CHECK_NEAR(v, ref, 1e-3 * fabs(ref)); /* same result */
        nc_set_grad_enabled(ctx, prev);
        peak[mode] = nc_get_mem_stats(ctx).arena_peak_bytes;
    }
    printf("    4-layer forward peak: grad on %.1f MiB, no_grad + release %.1f MiB\n", peak[0] / 1048576.0,
           peak[1] / 1048576.0);
    CHECK(peak[1] * 2 < peak[0]);

    /* never released: tensors on the tape, and views */
    nc_step_end(ctx);
    nc_tensor *p = nc_parameter(ctx, nc_full(ctx, 2.0f, 1, SHAPE(4)), 1, SHAPE(4));
    nc_tensor *a = nc_full(ctx, 3.0f, 1, SHAPE(4));
    nc_tensor *av = nc_reshape(a, 2, SHAPE(2, 2));
    nc_tensor *loss = nc_sum(nc_mul(nc_reshape(av, 1, SHAPE(4)), p), NC_ALL_AXES, 0);
    nc_release_temp(a);  /* a's data is used by the recorded mul (through views): kept */
    nc_release_temp(av); /* a view owns nothing */
    CHECK(nc_data_f32(a) != NULL);
    CHECK(nc_backward(loss) == NC_OK);
    CHECK_NEAR(nc_data_f32(nc_grad(p))[0], 3.0, 0);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

/* A copied optimizer state (buffers + step count) continues bit-identically. */
static void test_optim_state(void) {
    nc_ctx *ctx = new_ctx();
    float wv[4] = {1.0f, -2.0f, 0.5f, 3.0f};
    for (int kind = 0; kind < 3; kind++) { /* adam, sgd + momentum, rmsprop + momentum */
        nc_optim_opts o = {.lr = 0.05f, .momentum = kind ? 0.9f : 0.0f, .weight_decay = 0.01f};
        nc_tensor *a = nc_parameter(ctx, T(ctx, wv, 4), 1, SHAPE(4));
        nc_tensor *b = nc_parameter(ctx, T(ctx, wv, 4), 1, SHAPE(4));
        nc_optim *oa = kind == 0 ? nc_adam(ctx, &a, 1, o) : kind == 1 ? nc_sgd(ctx, &a, 1, o)
                                                                       : nc_rmsprop(ctx, &a, 1, o, 0);
        nc_optim *ob = kind == 0 ? nc_adam(ctx, &b, 1, o) : kind == 1 ? nc_sgd(ctx, &b, 1, o)
                                                                       : nc_rmsprop(ctx, &b, 1, o, 0);
        for (int s = 0; s < 5; s++) { /* train a only */
            nc_optim_zero_grad(oa);
            nc_backward(nc_sum(nc_mul(a, a), NC_ALL_AXES, 0));
            nc_optim_step(oa);
            nc_step_end(ctx);
        }
        CHECK(nc_optim_get_step(oa) == 5);
        memcpy(nc_data_f32(b), nc_data_f32(a), sizeof wv);
        for (int slot = 0; slot < 2; slot++) {
            float *sa = nc_optim_state(oa, 0, slot), *sb = nc_optim_state(ob, 0, slot);
            CHECK((sa == NULL) == (sb == NULL));
            if (sa) memcpy(sb, sa, sizeof wv);
        }
        CHECK(nc_optim_state(oa, 0, 1) != NULL || kind == 1); /* SGD has no slot 1 */
        CHECK(nc_optim_state(oa, 1, 0) == NULL);              /* out of range */
        nc_optim_set_step(ob, nc_optim_get_step(oa));
        for (int s = 0; s < 3; s++) { /* both continue the same way */
            nc_optim *os[2] = {oa, ob};
            nc_tensor *ps[2] = {a, b};
            for (int k = 0; k < 2; k++) {
                nc_optim_zero_grad(os[k]);
                nc_backward(nc_sum(nc_mul(ps[k], ps[k]), NC_ALL_AXES, 0));
                nc_optim_step(os[k]);
            }
            nc_step_end(ctx);
        }
        CHECK(memcmp(nc_data_f32(a), nc_data_f32(b), sizeof wv) == 0);
        if (kind) { /* momentum buffer cannot appear or vanish */
            o.momentum = 0.0f;
            nc_optim_set_opts(oa, o);
            CHECK(nc_status_code(ctx) == NC_ERR_ARG);
            nc_clear_error(ctx);
        }
    }
    nc_optim_set_step(NULL, 1);
    /* nc_optim_free gives all of its state back */
    nc_tensor *w = nc_parameter(ctx, T(ctx, wv, 4), 1, SHAPE(4));
    size_t base = nc_get_mem_stats(ctx).persistent_bytes;
    nc_optim_free(nc_adam(ctx, &w, 1, (nc_optim_opts){.lr = 0.1f}));
    nc_optim_free(nc_sgd(ctx, &w, 1, (nc_optim_opts){.lr = 0.1f, .momentum = 0.9f}));
    CHECK(nc_get_mem_stats(ctx).persistent_bytes == base);
    nc_optim_free(NULL);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

/* u8: storage dtype for raw data, converted with to_dtype */
static void test_uint8(void) {
    nc_ctx *ctx = new_ctx();
    uint8_t px[4] = {0, 1, 128, 255};
    nc_tensor *u = nc_from_data(ctx, NC_U8, px, 2, SHAPE(2, 2));
    CHECK(u && nc_get_dtype(u) == NC_U8);
    CHECK_NEAR(nc_item(nc_reshape(nc_index_select(nc_reshape(u, 1, SHAPE(4)),
                                                  nc_from_data(ctx, NC_I32, (int32_t[]){3}, 1, SHAPE(1))),
                                  0, NULL)),
               255.0, 0);
    nc_tensor *f = nc_to_dtype(nc_transpose(u, 0, 1), NC_F32); /* strided u8 view */
    float want[4] = {0, 128, 1, 255};
    for (int i = 0; i < 4; i++) CHECK_NEAR(nc_data_f32(f)[i], want[i], 0);
    float fv[4] = {-3.0f, 0.4f, 254.6f, 1000.0f};
    nc_tensor *back = nc_to_dtype(T(ctx, fv, 4), NC_U8); /* truncates, saturates to [0, 255] */
    uint8_t got[4];
    CHECK(nc_copy_to(back, got, sizeof got));
    CHECK(got[0] == 0 && got[1] == 0 && got[2] == 254 && got[3] == 255);
    nc_tensor *i8 = nc_to_dtype(u, NC_I8);
    CHECK_NEAR(nc_get(i8, (int64_t[]){1, 1}), 127.0, 0);
    CHECK(nc_empty(ctx, (nc_dtype)99, 1, SHAPE(2)) == NULL);
    CHECK(nc_status_code(ctx) == NC_ERR_DTYPE);
    nc_clear_error(ctx);
    nc_free_ctx(ctx);
}

static void test_release(void) {
    nc_ctx *ctx = new_ctx();
    size_t base = nc_get_mem_stats(ctx).persistent_bytes;
    nc_tensor *p = nc_parameter(ctx, NULL, 2, SHAPE(16, 16));
    nc_backward(nc_sum(p, NC_ALL_AXES, 0)); /* creates a persistent grad */
    nc_step_end(ctx);
    nc_persistent_begin(ctx);
    nc_tensor *v = nc_reshape(p, 1, SHAPE(256)); /* persistent mode: a copy, never a view */
    nc_persistent_end(ctx);
    CHECK(nc_is_persistent(p) && nc_is_persistent(v) && nc_grad(p));
    CHECK(nc_data_f32(v) != nc_data_f32(p));
    nc_release(v); /* frees the copy only */
    CHECK_NEAR(nc_get(p, SHAPE(3, 3)), 0, 0);
    nc_release(p);
    CHECK(nc_get_mem_stats(ctx).persistent_bytes == base);

    float d[4] = {10, 11, 12, 13};
    int32_t ix[3] = {3, 0, 3};
    nc_tensor *sel = nc_index_select(nc_from_data(ctx, NC_F32, d, 2, SHAPE(4, 1)),
                                     nc_from_data(ctx, NC_I32, ix, 1, SHAPE(3)));
    CHECK(sel && nc_dim(sel, 0) == 3);
    CHECK_NEAR(nc_get(sel, SHAPE(2, 0)), 13, 0);
    ix[0] = 4;
    CHECK(!nc_index_select(nc_from_data(ctx, NC_F32, d, 1, SHAPE(4)), nc_from_data(ctx, NC_I32, ix, 1, SHAPE(3))));
    CHECK(nc_status_code(ctx) == NC_ERR_ARG);
    nc_free_ctx(ctx);
}

static void test_kernel_parity(void) {
    nc_ctx *ctx = new_ctx();
    const nc_kernels *ref = nc__kernels_ref(), *k = ctx->k;
    printf("    backend: %s (arch %s, neon %d, dotprod %d, fp16 %d, cores %d/%d big)\n", k->name,
           nc_cpu(ctx)->arch, nc_cpu(ctx)->has_neon, nc_cpu(ctx)->has_dotprod, nc_cpu(ctx)->has_fp16,
           nc_cpu(ctx)->n_cores, nc_cpu(ctx)->n_big_cores);
    enum { M = 13, N = 19, K = 11 };
    float A[M * K], B[K * N], C0[M * N], C1[M * N], x[37], y0[37], y1[37];
    for (int i = 0; i < M * K; i++) A[i] = nc__rand_normal(ctx);
    for (int i = 0; i < K * N; i++) B[i] = nc__rand_normal(ctx);
    for (int ta = 0; ta < 2; ta++) {
        for (int tb = 0; tb < 2; tb++) {
            for (int i = 0; i < M * N; i++) C0[i] = C1[i] = nc__rand_normal(ctx);
            int64_t lda = ta ? M : K, ldb = tb ? K : N;
            ref->gemm_f32(ta, tb, M, N, K, 0.7f, A, lda, B, ldb, 0.3f, C0, N);
            k->gemm_f32(ta, tb, M, N, K, 0.7f, A, lda, B, ldb, 0.3f, C1, N);
            float md = 0;
            for (int i = 0; i < M * N; i++) md = fmaxf(md, fabsf(C0[i] - C1[i]));
            CHECK(md < 1e-4f);
        }
    }
    for (int i = 0; i < 37; i++) x[i] = nc__rand_normal(ctx), y0[i] = y1[i] = nc__rand_normal(ctx);
    ref->axpy_f32(37, 0.5f, x, y0);
    k->axpy_f32(37, 0.5f, x, y1);
    for (int i = 0; i < 37; i++) CHECK_NEAR(y0[i], y1[i], 1e-6);
    CHECK_NEAR(ref->sum_f32(37, x), k->sum_f32(37, x), 1e-4);
    k->relu_f32(37, x, y1);
    for (int i = 0; i < 37; i++) CHECK_NEAR(y1[i], x[i] > 0 ? x[i] : 0, 0);

    /* transcendental kernels over a wide range, including the clamp edges
     * and the small-|x| branch of tanh */
    enum { NT = 203 };
    float tx[NT], r0[NT], r1[NT];
    for (int i = 0; i < NT; i++) tx[i] = -100.0f + 200.0f * (float)i / (NT - 1);
    tx[0] = 0.0f, tx[1] = 1e-6f, tx[2] = -0.2f, tx[3] = 0.249f, tx[4] = 0.251f;
    float worst[3] = {0, 0, 0};
    for (int f = 0; f < 3; f++) {
        void (*rf)(int64_t, const float *, float *) = f == 0 ? ref->exp_f32 : f == 1 ? ref->sigmoid_f32 : ref->tanh_f32;
        void (*kf)(int64_t, const float *, float *) = f == 0 ? k->exp_f32 : f == 1 ? k->sigmoid_f32 : k->tanh_f32;
        rf(NT, tx, r0);
        kf(NT, tx, r1);
        for (int i = 0; i < NT; i++) {
            if (f < 2 && (tx[i] > 88.0f || tx[i] < -87.0f)) continue; /* exp argument outside the float range */
            float err = fabsf(r0[i] - r1[i]) / fmaxf(fabsf(r0[i]), 1e-30f);
            if (!(err <= worst[f])) worst[f] = err;
            CHECK(isfinite(r1[i]));
        }
    }
    printf("    exp/sigmoid/tanh max rel err vs libm: %.2g %.2g %.2g\n", worst[0], worst[1], worst[2]);
    CHECK(worst[0] < 1e-6f && worst[1] < 1e-6f && worst[2] < 1e-5f);
    ref->erf_f32(NT, tx, r0);
    k->erf_f32(NT, tx, r1);
    float erf_abs = 0;
    for (int i = 0; i < NT; i++) erf_abs = fmaxf(erf_abs, fabsf(r0[i] - r1[i]));
    for (int i = 0; i < 40; i++) { /* dense around the curved part */
        float v = -4.0f + 0.2f * (float)i, a, b;
        ref->erf_f32(1, &v, &a);
        k->erf_f32(1, &v, &b);
        erf_abs = fmaxf(erf_abs, fabsf(a - b));
    }
    printf("    erf max abs err vs libm: %.2g\n", erf_abs);
    CHECK(erf_abs < 5e-7f);
    k->tanh_f32(1, (float[]){1e-6f}, r1); /* no cancellation near zero */
    CHECK_NEAR(r1[0], 1e-6, 1e-12);
    nc_free_ctx(ctx);
}

static void test_registry(void) {
    CHECK(nc__op_count() >= 28);
    for (int i = 0; i < nc__op_count(); i++) {
        const nc_op_def *op = nc__op_at(i);
        CHECK(op->name && op->infer && op->forward);
        CHECK(((op->flags & NC_OPF_NONDIFF) != 0) == (op->backward == NULL));
        CHECK(((op->flags & NC_OPF_VIEW) != 0) == (op->view != NULL));
        CHECK(nc__op_find(op->name) == op);
    }
}

/* ------------------------------------------------------------------------ */
/* Conv / pool / batch norm / dropout / int8 / io / threads                  */
/* ------------------------------------------------------------------------ */

/* distinct values (gaps of 0.1 > 2 * eps) so max-pool argmax never flips */
static nc_tensor *gdistinct(nc_ctx *ctx, int ndim, const int64_t *shape) {
    nc_tensor *p = nc_parameter(ctx, NULL, ndim, shape);
    float *d = nc_data_f32(p);
    int64_t n = nc_numel(p);
    for (int64_t i = 0; i < n; i++) d[i] = 0.1f * (float)(i - n / 2);
    for (int64_t i = n - 1; i > 0; i--) {
        int64_t j = (int64_t)(nc__rand_uniform01(ctx) * (float)(i + 1)) % (i + 1);
        float t = d[i];
        d[i] = d[j], d[j] = t;
    }
    return p;
}

static nc_tensor *g_rm, *g_rv;
static nc_tensor *f_conv(nc_tensor **x, nc_tensor *a) { (void)a; return nc_conv2d(x[0], x[1], x[2], 1, 1, 1, 1); }
static nc_tensor *f_conv_s2(nc_tensor **x, nc_tensor *a) { (void)a; return nc_conv2d(x[0], x[1], NULL, 2, 2, 0, 0); }
static nc_tensor *f_maxpool(nc_tensor **x, nc_tensor *a) { (void)a; return nc_max_pool2d(x[0], 2, 2, 2, 2, 0, 0); }
static nc_tensor *f_maxpool_p(nc_tensor **x, nc_tensor *a) { (void)a; return nc_max_pool2d(x[0], 3, 3, 1, 1, 1, 1); }
static nc_tensor *f_avgpool(nc_tensor **x, nc_tensor *a) { (void)a; return nc_avg_pool2d(x[0], 3, 3, 2, 2, 1, 1); }
static nc_tensor *f_bn_train(nc_tensor **x, nc_tensor *a) {
    (void)a;
    return nc_batch_norm(x[0], x[1], x[2], g_rm, g_rv, 1, 0.1f, 1e-5f);
}
static nc_tensor *f_bn_eval(nc_tensor **x, nc_tensor *a) {
    (void)a;
    return nc_batch_norm(x[0], x[1], x[2], g_rm, g_rv, 0, 0.1f, 1e-5f);
}

static void test_conv_pool_bn(void) {
    nc_ctx *ctx = new_ctx();
    nc_tensor *in[3];

    /* known values: 1 channel 3x3 input, 2x2 kernel of ones, no padding */
    float xv[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, ones[4] = {1, 1, 1, 1};
    nc_tensor *y = nc_conv2d(nc_from_data(ctx, NC_F32, xv, 4, SHAPE(1, 1, 3, 3)),
                             nc_from_data(ctx, NC_F32, ones, 4, SHAPE(1, 1, 2, 2)), NULL, 1, 1, 0, 0);
    CHECK(y && nc_dim(y, 2) == 2 && nc_dim(y, 3) == 2);
    CHECK_NEAR(nc_get(y, SHAPE(0, 0, 1, 1)), 5 + 6 + 8 + 9, 1e-5);
    nc_tensor *mp = nc_max_pool2d(nc_from_data(ctx, NC_F32, xv, 4, SHAPE(1, 1, 3, 3)), 2, 2, 1, 1, 0, 0);
    CHECK_NEAR(nc_get(mp, SHAPE(0, 0, 0, 1)), 6, 0);
    CHECK(!nc_conv2d(nc_zeros(ctx, NC_F32, 4, SHAPE(1, 2, 3, 3)), nc_zeros(ctx, NC_F32, 4, SHAPE(1, 3, 2, 2)), NULL,
                     1, 1, 0, 0));
    CHECK(nc_status_code(ctx) == NC_ERR_SHAPE);
    nc_clear_error(ctx);
    nc_step_end(ctx);

    in[0] = gparam(ctx, 0, 4, SHAPE(2, 2, 5, 5)), in[1] = gparam(ctx, 0, 4, SHAPE(3, 2, 3, 3));
    in[2] = gparam(ctx, 0, 1, SHAPE(3));
    gradcheck_case(ctx, "conv2d", f_conv, NULL, 3, in);
    in[0] = gparam(ctx, 0, 4, SHAPE(1, 2, 6, 7)), in[1] = gparam(ctx, 0, 4, SHAPE(2, 2, 2, 3));
    gradcheck_case(ctx, "conv2d s2", f_conv_s2, NULL, 2, in);
    in[0] = gdistinct(ctx, 4, SHAPE(2, 2, 4, 4));
    gradcheck_case(ctx, "max_pool2d", f_maxpool, NULL, 1, in);
    in[0] = gdistinct(ctx, 4, SHAPE(1, 2, 4, 5));
    gradcheck_case(ctx, "max_pool pad", f_maxpool_p, NULL, 1, in);
    in[0] = gparam(ctx, 0, 4, SHAPE(2, 2, 5, 5));
    gradcheck_case(ctx, "avg_pool2d", f_avgpool, NULL, 1, in);

    nc_persistent_begin(ctx);
    g_rm = nc_zeros(ctx, NC_F32, 1, SHAPE(3));
    g_rv = nc_full(ctx, 1.0f, 1, SHAPE(3));
    nc_persistent_end(ctx);
    in[0] = gparam(ctx, 0, 4, SHAPE(4, 3, 2, 2)), in[1] = gparam(ctx, 1, 1, SHAPE(3));
    in[2] = gparam(ctx, 0, 1, SHAPE(3));
    gradcheck_case(ctx, "batch_norm", f_bn_train, NULL, 3, in);
    gradcheck_case(ctx, "batch_norm ev", f_bn_eval, NULL, 3, in);
    in[0] = gparam(ctx, 0, 2, SHAPE(5, 3));
    gradcheck_case(ctx, "batch_norm 2d", f_bn_train, NULL, 3, in);

    /* running stats: one step from (0, 1) with momentum 1 equals batch stats */
    float bx[4] = {1, 2, 3, 6};
    nc_persistent_begin(ctx);
    nc_tensor *rm = nc_zeros(ctx, NC_F32, 1, SHAPE(1)), *rv = nc_full(ctx, 1.0f, 1, SHAPE(1));
    nc_tensor *w1 = nc_full(ctx, 1.0f, 1, SHAPE(1)), *b0 = nc_zeros(ctx, NC_F32, 1, SHAPE(1));
    nc_persistent_end(ctx);
    nc_tensor *bn = nc_batch_norm(nc_from_data(ctx, NC_F32, bx, 2, SHAPE(4, 1)), w1, b0, rm, rv, 1, 1.0f, 0.0f);
    CHECK(bn != NULL);
    CHECK_NEAR(nc_item(rm), 3.0, 1e-6);
    CHECK_NEAR(nc_item(rv), 14.0 / 3.0, 1e-5); /* unbiased */
    CHECK_NEAR(nc_get(bn, SHAPE(3, 0)), 3.0 / sqrt(3.5), 1e-5);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_dropout(void) {
    nc_ctx *ctx = new_ctx();
    int64_t n = 20000;
    nc_tensor *x = nc_parameter(ctx, NULL, 1, &n);
    for (int64_t i = 0; i < n; i++) nc_data_f32(x)[i] = 1.0f;
    nc_tensor *y = nc_dropout(x, 0.25f);
    int64_t zeros = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = nc_data_f32(y)[i];
        zeros += v == 0.0f;
        if (v != 0.0f) CHECK_NEAR(v, 1.0 / 0.75, 1e-6);
    }
    CHECK_NEAR((double)zeros / n, 0.25, 0.02);
    CHECK(nc_backward(nc_sum(y, NC_ALL_AXES, 0)) == NC_OK);
    for (int64_t i = 0; i < 50; i++) CHECK_NEAR(nc_data_f32(nc_grad(x))[i], nc_data_f32(y)[i], 0);
    CHECK_NEAR(nc_item(nc_sum(nc_dropout(x, 0.0f), NC_ALL_AXES, 0)), (double)n, 1e-2);
    CHECK(!nc_dropout(x, 1.5f) && nc_status_code(ctx) == NC_ERR_ARG);
    nc_free_ctx(ctx);
}

static void test_int8(void) {
    nc_ctx *ctx = new_ctx();
    enum { M = 7, IN = 70, OUT = 13 };
    nc_tensor *x = nc_randn(ctx, 1.0f, 2, SHAPE(M, IN));
    nc_tensor *w = nc_randn(ctx, 0.2f, 2, SHAPE(OUT, IN));
    nc_tensor *b = nc_randn(ctx, 1.0f, 1, SHAPE(OUT));
    nc_tensor *scale = NULL;
    nc_tensor *q = nc_quantize_rows(w, &scale);
    CHECK(q && scale && nc_get_dtype(q) == NC_I8 && nc_dim(scale, 0) == OUT);
    nc_tensor *ref = nc_linear_fn(x, w, b), *got = nc_linear_int8(x, q, scale, b);
    CHECK(got && nc_dim(got, 1) == OUT);
    double num = 0, den = 0;
    for (int i = 0; got && i < M * OUT; i++) {
        double d = nc_data_f32(got)[i] - nc_data_f32(ref)[i];
        num += d * d;
        den += (double)nc_data_f32(ref)[i] * nc_data_f32(ref)[i];
    }
    printf("    int8 linear relative error %.4f\n", sqrt(num / den));
    CHECK(sqrt(num / den) < 0.02);

    /* int8 GEMM kernels are exact: compare the active table to the reference */
    int8_t A[5 * 37], B[6 * 37];
    int32_t C0[30], C1[30];
    for (int i = 0; i < 5 * 37; i++) A[i] = (int8_t)((i * 37) % 255 - 127);
    for (int i = 0; i < 6 * 37; i++) B[i] = (int8_t)((i * 91) % 255 - 127);
    nc__kernels_ref()->gemm_s8_nt(5, 6, 37, A, 37, B, 37, C0, 6);
    ctx->k->gemm_s8_nt(5, 6, 37, A, 37, B, 37, C1, 6);
    CHECK(memcmp(C0, C1, sizeof C0) == 0);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

static void test_save_load(void) {
    nc_ctx *ctx = new_ctx();
    const char *path = "nc_test_weights.ncw";
    nc_tensor *a = nc_parameter(ctx, nc_randn(ctx, 1.0f, 2, SHAPE(3, 4)), 2, SHAPE(3, 4));
    int32_t iv[3] = {7, -1, 42};
    nc_persistent_begin(ctx);
    nc_tensor *b = nc_from_data(ctx, NC_I32, iv, 1, SHAPE(3));
    nc_persistent_end(ctx);
    const char *names[2] = {"layer.weight", "counts"};
    nc_tensor *ts[2] = {nc_transpose(a, 0, 1), b}; /* non-contiguous view is written row-major */
    CHECK(nc_save(ctx, path, 2, names, ts) == NC_OK);

    nc_tensor *a2 = nc_parameter(ctx, NULL, 2, SHAPE(4, 3));
    nc_tensor *b2 = nc_zeros(ctx, NC_I32, 1, SHAPE(3));
    nc_tensor *out[2] = {a2, b2};
    const char *rev[2] = {"counts", "layer.weight"};
    nc_tensor *outr[2] = {b2, a2};
    CHECK(nc_load(ctx, path, 2, rev, outr) == NC_OK); /* order independent */
    CHECK_NEAR(nc_get(a2, SHAPE(3, 1)), nc_get(a, SHAPE(1, 3)), 0);
    CHECK(nc_data_i32(b2)[2] == 42);

    nc_tensor *wrong = nc_zeros(ctx, NC_F32, 2, SHAPE(3, 4));
    nc_tensor *bad[1] = {wrong};
    CHECK(nc_load(ctx, path, 1, names, bad) == NC_ERR_SHAPE);
    CHECK(strstr(nc_error(ctx), "[4, 3]") != NULL);
    nc_clear_error(ctx);
    const char *missing[1] = {"nope"};
    CHECK(nc_load(ctx, path, 1, missing, out) == NC_ERR_ARG);
    nc_clear_error(ctx);
    CHECK(nc_load(ctx, "does_not_exist.ncw", 1, names, out) == NC_ERR_ARG);
    remove(path);
    nc_free_ctx(ctx);
}

static void test_threads(void) {
    /* every output element is computed by one thread: bit-identical results */
    float *r[2] = {NULL, NULL};
    /* SN: wide enough that 3 x K x SN crosses the parallel threshold with M < N,
     * so nc__gemm splits columns, in chunks that are not multiples of 16. */
    enum { M = 129, K = 300, N = 257, SN = 4099 };
    for (int pass = 0; pass < 2; pass++) {
        nc_config cfg = {0};
        cfg.seed = 99;
        cfg.threads = pass == 0 ? 1 : 4;
        nc_ctx *ctx = nc_init(&cfg);
        CHECK(nc_num_threads(ctx) == cfg.threads);
        nc_tensor *a = nc_randn(ctx, 1.0f, 2, SHAPE(M, K)), *b = nc_randn(ctx, 1.0f, 2, SHAPE(K, N));
        nc_tensor *c = nc_matmul(a, b), *ct = nc_matmul(nc_transpose(b, 0, 1), nc_transpose(a, 0, 1));
        nc_tensor *wide = nc_randn(ctx, 1.0f, 2, SHAPE(K, SN));
        nc_tensor *small = nc_matmul(nc_randn(ctx, 1.0f, 2, SHAPE(3, K)), wide); /* column split path */
        r[pass] = (float *)malloc(sizeof(float) * (M * N * 2 + 3 * SN));
        memcpy(r[pass], nc_data_f32(c), sizeof(float) * M * N);
        memcpy(r[pass] + M * N, nc_data_f32(nc_contiguous(ct)), sizeof(float) * M * N);
        memcpy(r[pass] + 2 * M * N, nc_data_f32(small), sizeof(float) * 3 * SN);
        CHECK(nc_ok(ctx));
        nc_free_ctx(ctx);
    }
    CHECK(memcmp(r[0], r[1], sizeof(float) * (M * N * 2 + 3 * SN)) == 0);
    free(r[0]);
    free(r[1]);
}

static void test_arena_trim(void) {
    nc_ctx *ctx = new_ctx();
    nc_randn(ctx, 1.0f, 2, SHAPE(1024, 1024)); /* 4 MiB: forces extra chunks */
    nc_step_end(ctx);
    CHECK(nc_get_mem_stats(ctx).arena_bytes_reserved >= 4u << 20);
    nc_arena_trim(ctx);
    CHECK(nc_get_mem_stats(ctx).arena_bytes_reserved == 0);
    nc_tensor *t = nc_full(ctx, 3.0f, 1, SHAPE(8)); /* arena re-grows */
    CHECK(t && nc_data_f32(t)[7] == 3.0f);
    nc_free_ctx(ctx);
}

static void test_optim_extras(void) {
    nc_ctx *ctx = new_ctx();
    float wv[3] = {1.0f, -2.0f, 0.5f};

    /* AdamW step 1: w * (1 - lr*wd) - lr * sign(g) (m/sqrt(v) = sign(g) at t=1) */
    nc_tensor *w = nc_parameter(ctx, T(ctx, wv, 3), 1, SHAPE(3));
    nc_optim *aw = nc_adamw(ctx, &w, 1, (nc_optim_opts){.lr = 0.1f, .weight_decay = 0.5f});
    nc_backward(nc_sum(nc_scale(w, 2.0f), NC_ALL_AXES, 0)); /* g = 2 */
    nc_optim_step(aw);
    for (int i = 0; i < 3; i++) CHECK_NEAR(nc_data_f32(w)[i], wv[i] * 0.95f - 0.1f, 1e-5);
    nc_step_end(ctx);

    /* accumulation over 2 steps == one step on the averaged gradient */
    nc_tensor *a = nc_parameter(ctx, T(ctx, wv, 3), 1, SHAPE(3));
    nc_tensor *b = nc_parameter(ctx, T(ctx, wv, 3), 1, SHAPE(3));
    nc_optim *oa = nc_sgd(ctx, &a, 1, (nc_optim_opts){.lr = 0.1f});
    nc_optim *ob = nc_sgd(ctx, &b, 1, (nc_optim_opts){.lr = 0.1f});
    nc_optim_set_accumulation(oa, 2);
    float k[2] = {1.0f, 3.0f};
    for (int s = 0; s < 2; s++) {
        nc_optim_zero_grad(oa);
        nc_backward(nc_sum(nc_scale(a, k[s]), NC_ALL_AXES, 0));
        nc_optim_step(oa);
        if (s == 0) CHECK_NEAR(nc_data_f32(a)[0], wv[0], 0); /* no update yet */
        nc_step_end(ctx);
    }
    nc_optim_zero_grad(ob);
    nc_backward(nc_sum(nc_scale(b, 2.0f), NC_ALL_AXES, 0)); /* mean of 1 and 3 */
    nc_optim_step(ob);
    for (int i = 0; i < 3; i++) CHECK_NEAR(nc_data_f32(a)[i], nc_data_f32(b)[i], 1e-6);
    nc_optim_zero_grad(oa); /* after an update, zero_grad clears again */
    CHECK_NEAR(nc_data_f32(nc_grad(a))[0], 0, 0);
    nc_step_end(ctx);
    nc_optim_set_accumulation(oa, 0);
    CHECK(nc_status_code(ctx) == NC_ERR_ARG);
    nc_clear_error(ctx);

    /* clear_grad: reads as absent, optimizer skips it, next backward starts at 0 */
    nc_tensor *z = nc_parameter(ctx, T(ctx, wv, 3), 1, SHAPE(3));
    nc_optim *oz = nc_sgd(ctx, &z, 1, (nc_optim_opts){.lr = 1.0f, .weight_decay = 1.0f});
    nc_backward(nc_sum(z, NC_ALL_AXES, 0));
    nc_optim_clear_grad(oz);
    CHECK(nc_grad(z) == NULL);
    nc_optim_step(oz); /* skipped: weight decay must not touch z */
    CHECK_NEAR(nc_data_f32(z)[0], wv[0], 0);
    nc_backward(nc_sum(nc_scale(z, 2.0f), NC_ALL_AXES, 0));
    CHECK(nc_grad(z) && nc_data_f32(nc_grad(z))[1] == 2.0f); /* not 1 + 2 */
    nc_step_end(ctx);

    /* RMSprop step 1 from v = 0: g / (sqrt((1-a) g^2) + eps) = 1 / sqrt(1-a) */
    nc_tensor *r = nc_parameter(ctx, T(ctx, wv, 3), 1, SHAPE(3));
    nc_optim *orms = nc_rmsprop(ctx, &r, 1, (nc_optim_opts){.lr = 0.01f}, 0);
    nc_backward(nc_sum(nc_scale(r, 3.0f), NC_ALL_AXES, 0));
    nc_optim_step(orms);
    CHECK_NEAR(nc_data_f32(r)[0], wv[0] - 0.01 / sqrt(0.01), 1e-4);
    nc_step_end(ctx);

    /* clip_grad_norm: grads (3, 4) -> norm 5, clipped to norm 1 */
    float cv[2] = {0.0f, 0.0f}, gv[2] = {3.0f, 4.0f};
    nc_tensor *c = nc_parameter(ctx, T(ctx, cv, 2), 1, SHAPE(2));
    nc_backward(nc_sum(nc_mul(c, T(ctx, gv, 2)), NC_ALL_AXES, 0));
    CHECK_NEAR(nc_clip_grad_norm(&c, 1, 1.0f), 5.0, 1e-5);
    CHECK_NEAR(nc_data_f32(nc_grad(c))[1], 0.8, 1e-5);
    CHECK_NEAR(nc_clip_grad_norm(&c, 1, 10.0f), 1.0, 1e-5); /* below max: unchanged */
    CHECK_NEAR(nc_data_f32(nc_grad(c))[0], 0.6, 1e-5);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

/* ------------------------------------------------------------------------ */
/* End-to-end training                                                       */
/* ------------------------------------------------------------------------ */

/* 3-armed spiral: not linearly separable, needs a hidden layer. */
static void make_spiral(nc_ctx *ctx, int per_class, float *X, int32_t *Y) {
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < per_class; i++) {
            float r = (float)i / (float)per_class;
            float th = (float)c * 4.0f + r * 4.0f + 0.2f * nc__rand_normal(ctx);
            int j = c * per_class + i;
            X[2 * j] = r * sinf(th);
            X[2 * j + 1] = r * cosf(th);
            Y[j] = c;
        }
    }
}

static void test_train_spiral(void) {
    nc_ctx *ctx = new_ctx();
    enum { PER = 100, NS = 3 * PER };
    float X[NS * 2];
    int32_t Y[NS];
    make_spiral(ctx, PER, X, Y);

    nc_module *layers[] = {nc_linear(ctx, 2, 32, 1), nc_act(ctx, NC_ACT_RELU), nc_linear(ctx, 32, 32, 1),
                           nc_act(ctx, NC_ACT_RELU), nc_linear(ctx, 32, 3, 1)};
    nc_module *model = nc_sequential(ctx, 5, layers);
    nc_tensor *params[16];
    int np = nc_parameters(model, params, 16);
    CHECK(np == 6);
    nc_optim *opt = nc_adam(ctx, params, np, (nc_optim_opts){.lr = 0.01f});

    float first = 0, last = 0;
    for (int step = 0; step < 400; step++) {
        nc_tensor *x = nc_from_data(ctx, NC_F32, X, 2, SHAPE(NS, 2));
        nc_tensor *y = nc_from_data(ctx, NC_I32, Y, 1, SHAPE(NS));
        nc_tensor *loss = nc_cross_entropy(nc_forward(model, x), y);
        nc_optim_zero_grad(opt);
        nc_backward(loss);
        nc_optim_step(opt);
        if (step == 0) first = nc_item(loss);
        last = nc_item(loss);
        nc_step_end(ctx);
    }
    int prev = nc_set_grad_enabled(ctx, 0);
    nc_tensor *pred = nc_argmax(nc_forward(model, nc_from_data(ctx, NC_F32, X, 2, SHAPE(NS, 2))), 1);
    nc_set_grad_enabled(ctx, prev);
    int correct = 0;
    for (int i = 0; i < NS; i++) correct += nc_data_i32(pred)[i] == Y[i];
    float acc = (float)correct / NS;
    printf("    spiral: loss %.3f -> %.3f, accuracy %.1f%%\n", first, last, 100 * acc);
    CHECK(nc_ok(ctx));
    CHECK(last < first * 0.3f);
    CHECK(acc > 0.9f);
    nc_free_ctx(ctx);
}

static void test_train_regression_sgd(void) {
    nc_ctx *ctx = new_ctx();
    nc_module *lin = nc_linear(ctx, 1, 1, 1);
    nc_tensor *params[2];
    int np = nc_parameters(lin, params, 2);
    nc_optim *opt = nc_sgd(ctx, params, np, (nc_optim_opts){.lr = 0.1f, .momentum = 0.9f});
    for (int step = 0; step < 200; step++) {
        nc_tensor *x = nc_rand_uniform(ctx, -1, 1, 2, SHAPE(32, 1));
        nc_tensor *y = nc_add_scalar(nc_scale(x, 3.0f), 1.0f); /* y = 3x + 1 */
        nc_tensor *loss = nc_mse_loss(nc_forward(lin, x), y);
        nc_optim_zero_grad(opt);
        nc_backward(loss);
        nc_optim_step(opt);
        nc_step_end(ctx);
    }
    CHECK_NEAR(nc_item(params[0]), 3.0, 1e-2);
    CHECK_NEAR(nc_item(params[1]), 1.0, 1e-2);
    CHECK(nc_ok(ctx));
    nc_free_ctx(ctx);
}

/* ------------------------------------------------------------------------ */

typedef struct {
    const char *name;
    void (*fn)(void);
} test_entry;

int main(void) {
    const test_entry tests[] = {
        {"tensor_basics", test_tensor_basics},
        {"views", test_views},
        {"broadcast", test_broadcast},
        {"matmul_values", test_matmul_values},
        {"reduce_values", test_reduce_values},
        {"errors", test_errors},
        {"gradcheck", test_gradcheck},
        {"autograd_semantics", test_autograd_semantics},
        {"memory", test_memory},
        {"release", test_release},
        {"backward_release", test_backward_release},
        {"inference_release", test_inference_release},
        {"optim_state", test_optim_state},
        {"uint8", test_uint8},
        {"kernel_parity", test_kernel_parity},
        {"conv_pool_bn", test_conv_pool_bn},
        {"dropout", test_dropout},
        {"int8", test_int8},
        {"save_load", test_save_load},
        {"threads", test_threads},
        {"arena_trim", test_arena_trim},
        {"optim_extras", test_optim_extras},
        {"registry", test_registry},
        {"train_spiral", test_train_spiral},
        {"train_regression_sgd", test_train_regression_sgd},
    };
    int failed_tests = 0;
    for (size_t i = 0; i < sizeof tests / sizeof tests[0]; i++) {
        int before = g_fails;
        printf("[ RUN  ] %s\n", tests[i].name);
        tests[i].fn();
        int ok = g_fails == before;
        failed_tests += !ok;
        printf("[ %s ] %s\n", ok ? " OK " : "FAIL", tests[i].name);
    }
    printf("\n%d checks, %d failed, %d/%d tests failed\n", g_checks, g_fails, failed_tests,
           (int)(sizeof tests / sizeof tests[0]));
    return g_fails ? 1 : 0;
}
