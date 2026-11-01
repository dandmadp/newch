/* Trains a small MLP on a 3-class spiral and reports speed and memory.
 * Doubles as a quick benchmark to compare boards:
 *     ./train_spiral [steps] [hidden]
 */
#include "nc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) { return (double)clock() / CLOCKS_PER_SEC; }

/* Deterministic generator independent of the engine RNG. */
static float frand(unsigned *s) {
    *s = *s * 1664525u + 1013904223u;
    return (float)(*s >> 8) / 16777216.0f;
}

int main(int argc, char **argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 500;
    int hidden = argc > 2 ? atoi(argv[2]) : 64;
    enum { PER = 200, NS = 3 * PER };

    static float X[NS * 2];
    static int32_t Y[NS];
    unsigned seed = 42;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < PER; i++) {
            float r = (float)i / PER;
            float th = c * 4.0f + r * 4.0f + 0.3f * (frand(&seed) - 0.5f);
            X[2 * (c * PER + i)] = r * sinf(th);
            X[2 * (c * PER + i) + 1] = r * cosf(th);
            Y[c * PER + i] = c;
        }
    }

    nc_ctx *ctx = nc_init(&(nc_config){.seed = 7});
    const nc_cpu_info *cpu = nc_cpu(ctx);
    printf("newch %d.%d.%d | arch %s | kernels %s | cores %d (%d big) | neon %d dotprod %d fp16 %d\n",
           NC_VERSION_MAJOR, NC_VERSION_MINOR, NC_VERSION_PATCH, cpu->arch, nc_kernel_backend(ctx), cpu->n_cores,
           cpu->n_big_cores, cpu->has_neon, cpu->has_dotprod, cpu->has_fp16);

    nc_module *layers[] = {
        nc_linear(ctx, 2, hidden, 1), nc_act(ctx, NC_ACT_RELU),
        nc_linear(ctx, hidden, hidden, 1), nc_act(ctx, NC_ACT_RELU),
        nc_linear(ctx, hidden, 3, 1),
    };
    nc_module *model = nc_sequential(ctx, 5, layers);
    nc_tensor *params[16];
    int np = nc_parameters(model, params, 16);
    nc_optim *opt = nc_adam(ctx, params, np, (nc_optim_opts){.lr = 0.005f});

    int64_t xs[2] = {NS, 2}, ys[1] = {NS};
    double t0 = now_sec();
    for (int step = 1; step <= steps; step++) {
        nc_tensor *x = nc_from_data(ctx, NC_F32, X, 2, xs);
        nc_tensor *y = nc_from_data(ctx, NC_I32, Y, 1, ys);
        nc_tensor *loss = nc_cross_entropy(nc_forward(model, x), y);
        nc_optim_zero_grad(opt);
        nc_backward(loss);
        nc_optim_step(opt);
        if (!nc_ok(ctx)) {
            fprintf(stderr, "error: %s\n", nc_error(ctx));
            return 1;
        }
        if (step % 100 == 0 || step == 1) printf("step %4d  loss %.4f\n", step, nc_item(loss));
        nc_step_end(ctx);
    }
    double dt = now_sec() - t0;

    nc_set_grad_enabled(ctx, 0);
    nc_tensor *pred = nc_argmax(nc_forward(model, nc_from_data(ctx, NC_F32, X, 2, xs)), 1);
    int correct = 0;
    for (int i = 0; i < NS; i++) correct += nc_data_i32(pred)[i] == Y[i];

    nc_mem_stats m = nc_get_mem_stats(ctx);
    printf("accuracy %.1f%% | %.2f ms/step | persistent %.1f KiB | arena peak %.1f KiB\n",
           100.0 * correct / NS, 1000.0 * dt / steps, m.persistent_bytes / 1024.0, m.arena_peak_bytes / 1024.0);
    nc_free_ctx(ctx);
    return 0;
}
