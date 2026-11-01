/* Central-difference gradient check. Every op's backward is tested with it.
 * Graphs built here are not freed until the caller's nc_step_end(), so keep
 * inputs small. */
#include "../core/nc_internal.h"

#include <math.h>

float nc_gradcheck(nc_ctx *ctx, nc_scalar_fn fn, nc_tensor **inputs, int n_inputs, void *user,
                   float eps) {
    for (int i = 0; i < n_inputs; i++) nc_zero_grad(inputs[i]);
    nc_tensor *y = fn(ctx, inputs, user);
    if (!y || nc_backward(y) != NC_OK) return INFINITY;

    float max_err = 0.0f;
    int prev = nc_set_grad_enabled(ctx, 0);
    for (int i = 0; i < n_inputs && nc_ok(ctx); i++) {
        nc_tensor *x = inputs[i];
        if (!nc__needs_grad(x)) continue;
        float *d = nc_data_f32(x);
        if (!d) {
            nc__error(ctx, NC_ERR_ARG, "gradcheck: input %d must be contiguous f32", i);
            break;
        }
        const float *ana = x->grad ? (const float *)x->grad->data : NULL;
        for (int64_t j = 0, n = nc_numel(x); j < n; j++) {
            float orig = d[j];
            d[j] = orig + eps;
            float fp = nc_item(fn(ctx, inputs, user));
            d[j] = orig - eps;
            float fm = nc_item(fn(ctx, inputs, user));
            d[j] = orig;
            float num = (fp - fm) / (2.0f * eps);
            float a = ana ? ana[j] : 0.0f;
            float denom = fmaxf(1.0f, fmaxf(fabsf(num), fabsf(a)));
            float err = fabsf(num - a) / denom;
            if (!(err <= max_err)) max_err = err; /* also catches NaN */
        }
    }
    nc_set_grad_enabled(ctx, prev);
    return nc_ok(ctx) ? max_err : INFINITY;
}
