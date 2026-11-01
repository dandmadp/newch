/* Builds the kernel table for the running CPU: start from the reference
 * table, then let each available tier override the entries it implements. */
#include "kernels.h"

#include <stddef.h>

#define NC_MERGE(field) \
    if (t->field) dst->field = t->field

static void merge(nc_kernels *dst, const nc_kernels *t) {
    NC_MERGE(add_f32);
    NC_MERGE(sub_f32);
    NC_MERGE(mul_f32);
    NC_MERGE(div_f32);
    NC_MERGE(axpy_f32);
    NC_MERGE(scale_f32);
    NC_MERGE(relu_f32);
    NC_MERGE(sum_f32);
    NC_MERGE(exp_f32);
    NC_MERGE(sigmoid_f32);
    NC_MERGE(tanh_f32);
    NC_MERGE(erf_f32);
    NC_MERGE(gemm_f32);
    NC_MERGE(gemm_s8_nt);
    dst->name = t->name;
}

#undef NC_MERGE

/* Tables are immutable after init, so one static copy per configuration is
 * shared by every context. */
static nc_kernels selected_table;
static nc_kernels reference_table;

const nc_kernels *nc__select_kernels(const nc_cpu_info *cpu, int force_reference) {
    if (force_reference) {
        reference_table = *nc__kernels_ref();
        return &reference_table;
    }
    nc_kernels k = *nc__kernels_ref();
    const nc_kernels *neon = nc__kernels_neon();
    if (neon && cpu->has_neon) merge(&k, neon);
    const nc_kernels *avx2 = nc__kernels_avx2();
    if (avx2 && cpu->has_avx2 && cpu->has_fma) merge(&k, avx2);
    /* future tiers: neon_dotprod (int8), rvv */
    selected_table = k;
    return &selected_table;
}
