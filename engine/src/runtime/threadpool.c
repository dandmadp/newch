/* Minimal thread pool for data-parallel loops, plus the parallel GEMM used
 * by every op. Workers grab chunks through an atomic counter, so uneven
 * cores (big.LITTLE) balance themselves. */
#include "../core/nc_internal.h"

#include <stdatomic.h>
#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HANDLE nc_thread;
typedef SRWLOCK nc_mutex;
typedef CONDITION_VARIABLE nc_cond;
#define MUTEX_INIT(m) InitializeSRWLock(m)
#define MUTEX_LOCK(m) AcquireSRWLockExclusive(m)
#define MUTEX_UNLOCK(m) ReleaseSRWLockExclusive(m)
#define MUTEX_FREE(m) ((void)0)
#define COND_INIT(c) InitializeConditionVariable(c)
#define COND_WAIT(c, m) SleepConditionVariableSRW(c, m, INFINITE, 0)
#define COND_SIGNAL(c) WakeConditionVariable(c)
#define COND_BROADCAST(c) WakeAllConditionVariable(c)
#define COND_FREE(c) ((void)0)
#else
#include <pthread.h>
typedef pthread_t nc_thread;
typedef pthread_mutex_t nc_mutex;
typedef pthread_cond_t nc_cond;
#define MUTEX_INIT(m) pthread_mutex_init(m, NULL)
#define MUTEX_LOCK(m) pthread_mutex_lock(m)
#define MUTEX_UNLOCK(m) pthread_mutex_unlock(m)
#define MUTEX_FREE(m) pthread_mutex_destroy(m)
#define COND_INIT(c) pthread_cond_init(c, NULL)
#define COND_WAIT(c, m) pthread_cond_wait(c, m)
#define COND_SIGNAL(c) pthread_cond_signal(c)
#define COND_BROADCAST(c) pthread_cond_broadcast(c)
#define COND_FREE(c) pthread_cond_destroy(c)
#endif

struct nc_pool {
    int n_workers;
    nc_thread *threads;
    nc_mutex mu;
    nc_cond work_cv, done_cv;
    uint64_t gen;
    int active;
    int stop;
    /* current job */
    nc_range_fn fn;
    void *arg;
    int64_t n, chunk, n_chunks;
    atomic_llong next;
};

/* Nested parallel_for calls (an op inside a parallel region) run serially. */
static _Thread_local int in_parallel;

static void run_chunks(struct nc_pool *p) {
    long long c;
    while ((c = atomic_fetch_add(&p->next, 1)) < p->n_chunks) {
        int64_t b = (int64_t)c * p->chunk;
        int64_t e = b + p->chunk < p->n ? b + p->chunk : p->n;
        p->fn(p->arg, b, e);
    }
}

#if defined(_WIN32)
static DWORD WINAPI worker(LPVOID arg)
#else
static void *worker(void *arg)
#endif
{
    struct nc_pool *p = (struct nc_pool *)arg;
    uint64_t seen = 0;
    in_parallel = 1;
    for (;;) {
        MUTEX_LOCK(&p->mu);
        while (p->gen == seen && !p->stop) COND_WAIT(&p->work_cv, &p->mu);
        if (p->stop) {
            MUTEX_UNLOCK(&p->mu);
            break;
        }
        seen = p->gen;
        MUTEX_UNLOCK(&p->mu);
        run_chunks(p);
        MUTEX_LOCK(&p->mu);
        if (--p->active == 0) COND_SIGNAL(&p->done_cv);
        MUTEX_UNLOCK(&p->mu);
    }
    return 0;
}

static struct nc_pool *pool_create(int n_workers) {
    struct nc_pool *p = (struct nc_pool *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->threads = (nc_thread *)calloc((size_t)n_workers, sizeof(nc_thread));
    if (!p->threads) {
        free(p);
        return NULL;
    }
    MUTEX_INIT(&p->mu);
    COND_INIT(&p->work_cv);
    COND_INIT(&p->done_cv);
    atomic_init(&p->next, 0);
    for (int i = 0; i < n_workers; i++) {
#if defined(_WIN32)
        p->threads[i] = CreateThread(NULL, 0, worker, p, 0, NULL);
        int ok = p->threads[i] != NULL;
#else
        int ok = pthread_create(&p->threads[i], NULL, worker, p) == 0;
#endif
        if (!ok) break;
        p->n_workers++;
    }
    if (p->n_workers == 0) {
        nc__pool_destroy(p);
        return NULL;
    }
    return p;
}

void nc__pool_destroy(struct nc_pool *p) {
    if (!p) return;
    MUTEX_LOCK(&p->mu);
    p->stop = 1;
    COND_BROADCAST(&p->work_cv);
    MUTEX_UNLOCK(&p->mu);
    for (int i = 0; i < p->n_workers; i++) {
#if defined(_WIN32)
        WaitForSingleObject(p->threads[i], INFINITE);
        CloseHandle(p->threads[i]);
#else
        pthread_join(p->threads[i], NULL);
#endif
    }
    COND_FREE(&p->work_cv);
    COND_FREE(&p->done_cv);
    MUTEX_FREE(&p->mu);
    free(p->threads);
    free(p);
}

void nc__parallel_for(nc_ctx *ctx, int64_t n, int64_t grain, nc_range_fn fn, void *arg) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    int T = ctx->n_threads;
    if (T <= 1 || n <= grain || in_parallel) {
        fn(arg, 0, n);
        return;
    }
    if (!ctx->pool && !(ctx->pool = pool_create(T - 1))) {
        ctx->n_threads = 1; /* could not start threads: stay serial */
        fn(arg, 0, n);
        return;
    }
    struct nc_pool *p = ctx->pool;
    int64_t chunks = (n + grain - 1) / grain;
    int64_t max_chunks = (int64_t)(p->n_workers + 1) * 4;
    if (chunks > max_chunks) chunks = max_chunks;
    p->fn = fn;
    p->arg = arg;
    p->n = n;
    p->chunk = (n + chunks - 1) / chunks;
    p->n_chunks = (n + p->chunk - 1) / p->chunk;
    atomic_store(&p->next, 0);

    MUTEX_LOCK(&p->mu);
    p->active = p->n_workers;
    p->gen++;
    COND_BROADCAST(&p->work_cv);
    MUTEX_UNLOCK(&p->mu);

    in_parallel = 1;
    run_chunks(p);
    in_parallel = 0;

    MUTEX_LOCK(&p->mu);
    while (p->active > 0) COND_WAIT(&p->done_cv, &p->mu);
    MUTEX_UNLOCK(&p->mu);
}

/* ------------------------------------------------------------------------ */
/* Parallel GEMM: split rows of C (or columns when M is small).             */
/* ------------------------------------------------------------------------ */

typedef struct {
    const nc_kernels *k;
    int ta, tb;
    int64_t M, N, K;
    float alpha, beta;
    const float *A, *B;
    int64_t lda, ldb, ldc;
    float *C;
} gemm_job;

static void gemm_rows(void *arg, int64_t b, int64_t e) {
    const gemm_job *j = (const gemm_job *)arg;
    const float *A = j->ta ? j->A + b : j->A + b * j->lda;
    j->k->gemm_f32(j->ta, j->tb, e - b, j->N, j->K, j->alpha, A, j->lda, j->B, j->ldb, j->beta, j->C + b * j->ldc,
                   j->ldc);
}

static void gemm_cols(void *arg, int64_t b, int64_t e) {
    const gemm_job *j = (const gemm_job *)arg;
    const float *B = j->tb ? j->B + b * j->ldb : j->B + b;
    j->k->gemm_f32(j->ta, j->tb, j->M, e - b, j->K, j->alpha, j->A, j->lda, B, j->ldb, j->beta, j->C + b, j->ldc);
}

void nc__gemm(nc_ctx *ctx, int ta, int tb, int64_t M, int64_t N, int64_t K, float alpha, const float *A,
              int64_t lda, const float *B, int64_t ldb, float beta, float *C, int64_t ldc) {
    /* Waking workers costs a few microseconds: only split problems of at
     * least ~2M multiply-adds, in chunks of at least ~256K each. */
    const double min_work = 1 << 18;
    int T = ctx->n_threads;
    gemm_job j = {ctx->k, ta, tb, M, N, K, alpha, beta, A, B, lda, ldb, ldc, C};
    double work = (double)M * (double)N * (double)(K ? K : 1);
    if (T <= 1 || work < 8 * min_work) {
        ctx->k->gemm_f32(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    double per_row = (double)N * (double)(K ? K : 1);
    if (M >= 4 * (int64_t)T || M >= N) {
        int64_t grain = (int64_t)(min_work / per_row) + 1;
        grain = (grain + 3) & ~(int64_t)3; /* whole 4-row micro-tiles */
        nc__parallel_for(ctx, M, grain, gemm_rows, &j);
    } else {
        int64_t grain = (int64_t)(min_work / ((double)M * (double)(K ? K : 1))) + 1;
        grain = (grain + 15) & ~(int64_t)15; /* whole 16-column micro-tiles */
        nc__parallel_for(ctx, N, grain, gemm_cols, &j);
    }
}
