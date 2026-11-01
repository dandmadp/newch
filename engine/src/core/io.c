/* Named-tensor files ("NCW1"). Layout, all little-endian:
 *
 *   char[4] magic "NCW1" | u32 version (1) | u32 count
 *   count x { u32 name_len | name bytes | u32 dtype | u32 ndim |
 *             i64 shape[ndim] | u64 nbytes | raw row-major data }
 *
 * Readers skip entries they were not asked for, so files can carry extra
 * tensors (e.g. optimizer state in a later version). */
#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
#define _FILE_OFFSET_BITS 64
#endif
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include "nc_internal.h"

#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define NC_FILE_MAGIC "NCW1"
#define NC_FILE_VERSION 1u
#define NC_MAX_NAME 4096

/* Paths are UTF-8 on every platform. */
static FILE *open_utf8(const char *path, const char *mode) {
#if defined(_WIN32)
    wchar_t wpath[1024], wmode[8];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024)) return NULL;
    if (!MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, 8)) return NULL;
    return _wfopen(wpath, wmode);
#else
    return fopen(path, mode);
#endif
}

static int skip(FILE *f, uint64_t n) {
#if defined(_WIN32)
    return _fseeki64(f, (long long)n, SEEK_CUR) == 0;
#else
    return fseeko(f, (off_t)n, SEEK_CUR) == 0;
#endif
}

static int put(FILE *f, const void *p, size_t n) { return fwrite(p, 1, n, f) == n; }
static int get(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

nc_status nc_save(nc_ctx *ctx, const char *path, int n, const char *const *names, nc_tensor *const *tensors) {
    if (ctx->status != NC_OK) return ctx->status;
    for (int i = 0; i < n; i++) {
        if (!tensors[i] || !names[i]) {
            nc__error(ctx, NC_ERR_ARG, "nc_save: entry %d has a NULL name or tensor", i);
            return ctx->status;
        }
    }
    FILE *f = open_utf8(path, "wb");
    if (!f) {
        nc__error(ctx, NC_ERR_ARG, "nc_save: cannot open '%s' for writing", path);
        return ctx->status;
    }
    uint32_t version = NC_FILE_VERSION, count = (uint32_t)n;
    int ok = put(f, NC_FILE_MAGIC, 4) && put(f, &version, 4) && put(f, &count, 4);
    for (int i = 0; ok && i < n; i++) {
        const nc_tensor *t = tensors[i];
        uint32_t len = (uint32_t)strlen(names[i]), dt = (uint32_t)t->dtype, nd = (uint32_t)t->ndim;
        uint64_t bytes = (uint64_t)nc_numel(t) * nc__dtype_size(t->dtype);
        ok = put(f, &len, 4) && put(f, names[i], len) && put(f, &dt, 4) && put(f, &nd, 4) &&
             put(f, t->shape, sizeof(int64_t) * nd) && put(f, &bytes, 8);
        if (!ok) break;
        if (nc_is_contiguous(t)) {
            ok = put(f, t->data, (size_t)bytes);
        } else {
            void *tmp = malloc((size_t)bytes + 1);
            ok = tmp != NULL;
            if (ok) {
                nc__copy_strided(t, tmp);
                ok = put(f, tmp, (size_t)bytes);
                free(tmp);
            }
        }
    }
    if (fclose(f) != 0) ok = 0;
    if (!ok) nc__error(ctx, NC_ERR_ARG, "nc_save: write to '%s' failed", path);
    return ctx->status;
}

nc_status nc_load(nc_ctx *ctx, const char *path, int n, const char *const *names, nc_tensor *const *tensors) {
    if (ctx->status != NC_OK) return ctx->status;
    FILE *f = open_utf8(path, "rb");
    if (!f) {
        nc__error(ctx, NC_ERR_ARG, "nc_load: cannot open '%s'", path);
        return ctx->status;
    }
    unsigned char *found = (unsigned char *)calloc((size_t)(n ? n : 1), 1);
    char *name = (char *)malloc(NC_MAX_NAME + 1);
    char magic[4];
    uint32_t version = 0, count = 0;
    if (!found || !name) {
        nc__error(ctx, NC_ERR_OOM, "nc_load: out of memory");
        goto done;
    }
    if (!get(f, magic, 4) || memcmp(magic, NC_FILE_MAGIC, 4) != 0 || !get(f, &version, 4) || !get(f, &count, 4)) {
        nc__error(ctx, NC_ERR_ARG, "nc_load: '%s' is not an NCW1 tensor file", path);
        goto done;
    }
    if (version != NC_FILE_VERSION) {
        nc__error(ctx, NC_ERR_ARG, "nc_load: '%s' has unsupported version %u", path, version);
        goto done;
    }
    for (uint32_t e = 0; e < count; e++) {
        uint32_t len, dt, nd;
        int64_t shape[NC_MAX_DIMS];
        uint64_t bytes;
        if (!get(f, &len, 4) || len > NC_MAX_NAME || !get(f, name, len) || !get(f, &dt, 4) || !get(f, &nd, 4) ||
            nd > NC_MAX_DIMS || !get(f, shape, sizeof(int64_t) * nd) || !get(f, &bytes, 8)) {
            nc__error(ctx, NC_ERR_ARG, "nc_load: '%s' is truncated or corrupt", path);
            goto done;
        }
        name[len] = '\0';
        int idx = -1;
        for (int i = 0; i < n; i++)
            if (!found[i] && strcmp(names[i], name) == 0) idx = i;
        if (idx < 0) {
            if (!skip(f, bytes)) {
                nc__error(ctx, NC_ERR_ARG, "nc_load: '%s' is truncated", path);
                goto done;
            }
            continue;
        }
        nc_tensor *t = tensors[idx];
        int same = t->dtype == (nc_dtype)dt && t->ndim == (int)nd;
        for (uint32_t d = 0; same && d < nd; d++) same = t->shape[d] == shape[d];
        if (!same) {
            char want[96], got[96];
            nc__error(ctx, NC_ERR_SHAPE, "nc_load: '%s' is %s %s in the file but %s %s in the model", name,
                      nc__dtype_name((nc_dtype)dt), nc__shape_str((int)nd, shape, got, sizeof got),
                      nc__dtype_name(t->dtype), nc__shape_str(t->ndim, t->shape, want, sizeof want));
            goto done;
        }
        if (!nc_is_contiguous(t) || !get(f, t->data, (size_t)bytes)) {
            nc__error(ctx, NC_ERR_ARG, "nc_load: cannot read '%s' (truncated file or non-contiguous tensor)", name);
            goto done;
        }
        found[idx] = 1;
    }
    for (int i = 0; i < n; i++) {
        if (!found[i]) {
            nc__error(ctx, NC_ERR_ARG, "nc_load: '%s' not found in '%s'", names[i], path);
            break;
        }
    }
done:
    free(found);
    free(name);
    fclose(f);
    return ctx->status;
}
