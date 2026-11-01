/* Context: memory (step arena + persistent pool), errors, RNG, platform. */
#include "nc_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static uintptr_t align_up(uintptr_t p, size_t a) { return (p + (a - 1)) & ~(uintptr_t)(a - 1); }

/* ------------------------------------------------------------------------ */
/* Persistent pool: individually malloc'd blocks, all freed with the ctx.   */
/* ------------------------------------------------------------------------ */

/* Doubly linked so single blocks can be released early (nc_release). The
 * block address is stored just before the aligned user pointer. */
typedef struct nc_pblock {
    struct nc_pblock *next;
    struct nc_pblock *prev;
    size_t bytes;
} nc_pblock;

void *nc__persistent_alloc(nc_ctx *ctx, size_t bytes) {
    size_t total = sizeof(nc_pblock) + sizeof(void *) + NC_ALIGN + bytes;
    nc_pblock *blk = (nc_pblock *)malloc(total);
    if (!blk) {
        nc__error(ctx, NC_ERR_OOM, "out of memory (persistent, %zu bytes)", bytes);
        return NULL;
    }
    blk->prev = NULL;
    blk->next = (nc_pblock *)ctx->persistent_list;
    blk->bytes = bytes;
    if (blk->next) blk->next->prev = blk;
    ctx->persistent_list = blk;
    ctx->persistent_bytes += bytes;
    void *p = (void *)align_up((uintptr_t)(blk + 1) + sizeof(void *), NC_ALIGN);
    ((void **)p)[-1] = blk;
    memset(p, 0, bytes);
    return p;
}

void nc__persistent_free(nc_ctx *ctx, void *p) {
    if (!p) return;
    nc_pblock *blk = (nc_pblock *)((void **)p)[-1];
    if (blk->prev) blk->prev->next = blk->next;
    else ctx->persistent_list = blk->next;
    if (blk->next) blk->next->prev = blk->prev;
    ctx->persistent_bytes -= blk->bytes;
    free(blk);
}

/* ------------------------------------------------------------------------ */
/* Step arena: bump allocation in chunks; reset is O(1) and keeps chunks.   */
/* ------------------------------------------------------------------------ */

static nc_chunk *chunk_new(size_t cap) {
    nc_chunk *c = (nc_chunk *)malloc(sizeof(nc_chunk) + NC_ALIGN + cap);
    if (!c) return NULL;
    c->next = NULL;
    c->cap = cap;
    c->used = 0;
    c->base = (unsigned char *)align_up((uintptr_t)(c + 1), NC_ALIGN);
    return c;
}

void *nc__arena_alloc(nc_ctx *ctx, size_t bytes) {
    size_t need = (size_t)align_up(bytes ? bytes : 1, NC_ALIGN);
    nc_chunk *c = ctx->cur;
    while (c && c->cap - c->used < need) c = c->next;
    if (!c) {
        size_t cap = need > ctx->chunk_bytes ? need : ctx->chunk_bytes;
        c = chunk_new(cap);
        if (!c) {
            nc__error(ctx, NC_ERR_OOM, "out of memory (arena, %zu bytes)", bytes);
            return NULL;
        }
        /* append to keep reuse order stable across steps */
        if (!ctx->chunks) {
            ctx->chunks = c;
        } else {
            nc_chunk *t = ctx->chunks;
            while (t->next) t = t->next;
            t->next = c;
        }
        ctx->arena_reserved += cap;
    }
    void *p = c->base + c->used;
    c->used += need;
    ctx->cur = c;
    ctx->arena_in_use += need;
    if (ctx->arena_in_use > ctx->arena_peak) ctx->arena_peak = ctx->arena_in_use;
    return p;
}

void *nc__alloc(nc_ctx *ctx, size_t bytes, int persistent) {
    return persistent ? nc__persistent_alloc(ctx, bytes) : nc__arena_alloc(ctx, bytes);
}

int nc__is_persistent_mode(const nc_ctx *ctx) { return ctx->persistent_depth > 0; }

/* ------------------------------------------------------------------------ */
/* Lifecycle                                                                 */
/* ------------------------------------------------------------------------ */

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

nc_ctx *nc_init(const nc_config *cfg) {
    nc_ctx *ctx = (nc_ctx *)calloc(1, sizeof(nc_ctx));
    if (!ctx) return NULL;
    nc_config c = {0};
    if (cfg) c = *cfg;
    ctx->chunk_bytes = c.arena_chunk_bytes ? c.arena_chunk_bytes : NC_DEFAULT_CHUNK;
    ctx->grad_enabled = 1;
    nc_manual_seed(ctx, c.seed);
    nc__detect_cpu(&ctx->cpu);
    ctx->k = nc__select_kernels(&ctx->cpu, c.force_reference);
    ctx->n_threads = c.threads > 0 ? c.threads : ctx->cpu.n_cores;
    if (ctx->n_threads < 1) ctx->n_threads = 1;
    if (ctx->n_threads > 64) ctx->n_threads = 64;
    return ctx;
}

void nc_manual_seed(nc_ctx *ctx, uint64_t seed) {
    uint64_t s = seed ? seed : 0x6E63u; /* "nc" */
    for (int i = 0; i < 4; i++) ctx->rng[i] = splitmix64(&s);
}

void nc_free_ctx(nc_ctx *ctx) {
    if (!ctx) return;
    nc__pool_destroy(ctx->pool);
    nc_chunk *c = ctx->chunks;
    while (c) {
        nc_chunk *n = c->next;
        free(c);
        c = n;
    }
    nc_pblock *b = (nc_pblock *)ctx->persistent_list;
    while (b) {
        nc_pblock *n = b->next;
        free(b);
        b = n;
    }
    free(ctx);
}

void nc_step_end(nc_ctx *ctx) {
    for (nc_chunk *c = ctx->chunks; c; c = c->next) c->used = 0;
    ctx->cur = ctx->chunks;
    ctx->arena_in_use = 0;
    ctx->tape = NULL;
}

void nc_arena_trim(nc_ctx *ctx) {
    /* keep chunks up to the current one; free the rest (all, if idle) */
    nc_chunk **link = &ctx->chunks;
    if (ctx->arena_in_use) {
        while (*link && *link != ctx->cur) link = &(*link)->next;
        if (*link) link = &(*link)->next;
    }
    nc_chunk *c = *link;
    *link = NULL;
    while (c) {
        nc_chunk *n = c->next;
        ctx->arena_reserved -= c->cap;
        free(c);
        c = n;
    }
    if (!ctx->chunks) ctx->cur = NULL;
}

int nc_num_threads(const nc_ctx *ctx) { return ctx->n_threads; }

void nc_persistent_begin(nc_ctx *ctx) { ctx->persistent_depth++; }
void nc_persistent_end(nc_ctx *ctx) {
    if (ctx->persistent_depth > 0) ctx->persistent_depth--;
}

int nc_set_grad_enabled(nc_ctx *ctx, int enabled) {
    int prev = ctx->grad_enabled;
    ctx->grad_enabled = enabled ? 1 : 0;
    return prev;
}

/* ------------------------------------------------------------------------ */
/* Errors: the first error is kept until nc_clear_error().                  */
/* ------------------------------------------------------------------------ */

void nc__error(nc_ctx *ctx, nc_status code, const char *fmt, ...) {
    if (!ctx || ctx->status != NC_OK) return;
    ctx->status = code;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->err, sizeof ctx->err, fmt, ap);
    va_end(ap);
}

int nc_ok(const nc_ctx *ctx) { return ctx->status == NC_OK; }
nc_status nc_status_code(const nc_ctx *ctx) { return ctx->status; }
const char *nc_error(const nc_ctx *ctx) { return ctx->status == NC_OK ? "" : ctx->err; }
void nc_clear_error(nc_ctx *ctx) {
    ctx->status = NC_OK;
    ctx->err[0] = '\0';
}

/* ------------------------------------------------------------------------ */
/* Stats / platform                                                          */
/* ------------------------------------------------------------------------ */

nc_mem_stats nc_get_mem_stats(const nc_ctx *ctx) {
    nc_mem_stats s;
    s.persistent_bytes = ctx->persistent_bytes;
    s.arena_bytes_in_use = ctx->arena_in_use;
    s.arena_bytes_reserved = ctx->arena_reserved;
    s.arena_peak_bytes = ctx->arena_peak;
    return s;
}

void nc_reset_peak(nc_ctx *ctx) { ctx->arena_peak = ctx->arena_in_use; }

const nc_cpu_info *nc_cpu(const nc_ctx *ctx) { return &ctx->cpu; }
const char *nc_kernel_backend(const nc_ctx *ctx) { return ctx->k->name; }

/* ------------------------------------------------------------------------ */
/* RNG: xoshiro256**                                                         */
/* ------------------------------------------------------------------------ */

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(nc_ctx *ctx) {
    uint64_t *s = ctx->rng;
    uint64_t r = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return r;
}

float nc__rand_uniform01(nc_ctx *ctx) { return (float)(rng_next(ctx) >> 40) * (1.0f / 16777216.0f); }

float nc__rand_normal(nc_ctx *ctx) {
    float u1 = nc__rand_uniform01(ctx), u2 = nc__rand_uniform01(ctx);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
