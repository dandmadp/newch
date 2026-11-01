/* Runtime CPU feature and topology detection. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "../core/nc_internal.h"

#include <stdio.h>

#if defined(__linux__)
#include <sys/auxv.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

static int count_cores(void) {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#else
    return 1;
#endif
}

/* big.LITTLE: Linux exposes relative core capacity (max 1024). Cores at the
 * maximum are "big". Uniform SoCs report the same value everywhere. */
static int count_big_cores(int n_cores) {
#if defined(__linux__)
    int caps[256];
    int n = n_cores < 256 ? n_cores : 256;
    int max_cap = 0, found = 0;
    for (int i = 0; i < n; i++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
        FILE *f = fopen(path, "r");
        caps[i] = -1;
        if (!f) continue;
        if (fscanf(f, "%d", &caps[i]) == 1) {
            found++;
            if (caps[i] > max_cap) max_cap = caps[i];
        }
        fclose(f);
    }
    if (!found) return n_cores;
    int big = 0;
    for (int i = 0; i < n; i++)
        if (caps[i] == max_cap) big++;
    return big;
#else
    return n_cores;
#endif
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static void cpuid(int leaf, int sub, unsigned r[4]) { __cpuidex((int *)r, leaf, sub); }
static unsigned long long xgetbv0(void) { return _xgetbv(0); }
#else
#include <cpuid.h>
static void cpuid(int leaf, int sub, unsigned r[4]) { __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]); }
static unsigned long long xgetbv0(void) {
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}
#endif

/* AVX2/FMA need CPU support and the OS saving YMM state (OSXSAVE + XCR0). */
static void x86_features(int *avx2, int *fma) {
    unsigned r[4];
    *avx2 = *fma = 0;
    cpuid(0, 0, r);
    if (r[0] < 7) return;
    cpuid(1, 0, r);
    if (!(r[2] & (1u << 27))) return; /* OSXSAVE */
    if ((xgetbv0() & 6) != 6) return; /* XMM + YMM enabled */
    *fma = (r[2] & (1u << 12)) != 0;
    cpuid(7, 0, r);
    *avx2 = (r[1] & (1u << 5)) != 0;
}
#endif

void nc__detect_cpu(nc_cpu_info *c) {
    memset(c, 0, sizeof *c);

#if defined(__aarch64__)
    c->arch = "aarch64";
    c->has_neon = 1; /* mandatory in AArch64 */
#if defined(__linux__)
    unsigned long hw = getauxval(AT_HWCAP);
    c->has_fp16 = (hw & (1ul << 10)) != 0;    /* HWCAP_ASIMDHP */
    c->has_dotprod = (hw & (1ul << 20)) != 0; /* HWCAP_ASIMDDP */
#endif
#elif defined(__arm__)
    c->arch = "armv7";
#if defined(__linux__)
    c->has_neon = (getauxval(AT_HWCAP) & (1ul << 12)) != 0; /* HWCAP_NEON */
#elif defined(__ARM_NEON)
    c->has_neon = 1;
#endif
#elif defined(__x86_64__) || defined(_M_X64)
    c->arch = "x86_64";
    x86_features(&c->has_avx2, &c->has_fma);
#elif defined(__i386__) || defined(_M_IX86)
    c->arch = "x86";
#elif defined(__riscv) && __riscv_xlen == 64
    c->arch = "riscv64";
#if defined(__linux__)
    c->has_rvv = (getauxval(AT_HWCAP) & (1ul << ('V' - 'A'))) != 0;
#endif
#else
    c->arch = "unknown";
#endif

    c->n_cores = count_cores();
    c->n_big_cores = count_big_cores(c->n_cores);
}
