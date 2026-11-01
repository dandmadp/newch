/* Eager execution + tape-based reverse-mode autograd.
 *
 * Every op goes through nc__apply(). When autograd is on and any input
 * requires grad, the node is pushed onto ctx->tape. Since a node is always
 * recorded after the nodes that produced its inputs, walking the tape from
 * newest to oldest is a valid reverse topological order. */
#include "../core/nc_internal.h"

#include <math.h>

nc_tensor *nc__apply(const nc_op_def *op, nc_tensor **in, int n_in, const nc_op_params *p) {
    for (int i = 0; i < n_in; i++)
        if (!in[i]) return NULL; /* propagate earlier failures */
    nc_ctx *ctx = in[0]->ctx;
    if (ctx->status != NC_OK) return NULL;
    for (int i = 1; i < n_in; i++) {
        if (in[i]->ctx != ctx) {
            nc__error(ctx, NC_ERR_ARG, "%s: inputs belong to different contexts", op->name);
            return NULL;
        }
    }

    int persistent = nc__is_persistent_mode(ctx);
    nc_node *n = (nc_node *)nc__arena_alloc(ctx, sizeof(nc_node));
    if (!n) return NULL;
    memset(n, 0, sizeof *n);
    n->op = op;
    n->n_in = n_in;
    for (int i = 0; i < n_in; i++) n->in[i] = in[i];
    if (p) n->p = *p;

    nc_dtype dt;
    int nd = 0;
    int64_t shape[NC_MAX_DIMS];
    if (op->infer(n, &dt, &nd, shape) != NC_OK) return NULL;

    /* Persistent results always own their data, so each one can be released
     * on its own (nc_release) without invalidating another. */
    int can_view = op->view != NULL && !persistent;

    if (!(can_view && op->view(n, nd, shape))) {
        n->out = nc__tensor_new(ctx, dt, nd, shape, persistent);
        if (!n->out) return NULL;
        if (op->forward(n) != NC_OK) return NULL;
    }
    if (!n->out) return NULL;

    int record = ctx->grad_enabled && !persistent && op->backward != NULL;
    if (record) {
        record = 0;
        for (int i = 0; i < n_in; i++)
            if (nc__needs_grad(in[i])) record = 1;
    }
    if (record) {
        n->out->flags |= NC_TF_REQUIRES_GRAD;
        n->out->node = n;
        n->prev = ctx->tape;
        ctx->tape = n;
    }
    if (ctx->trace) ctx->trace(ctx, n, ctx->trace_user);
    return n->out;
}

nc_status nc_backward(nc_tensor *loss) {
    if (!loss) return NC_ERR_ARG;
    nc_ctx *ctx = loss->ctx;
    if (ctx->status != NC_OK) return ctx->status;
    if (nc_numel(loss) != 1 || loss->dtype != NC_F32) {
        nc__error(ctx, NC_ERR_SHAPE, "backward: loss must be a single f32 element");
        return ctx->status;
    }
    if (!nc__needs_grad(loss)) {
        nc__error(ctx, NC_ERR_STATE,
                  "backward: loss does not require grad (no parameter reached it, or grad was disabled)");
        return ctx->status;
    }

    /* Only nodes that lead to this loss take part. This keeps a second
     * backward() in the same step from re-propagating an earlier loss. */
    uint32_t ep = ++ctx->mark_epoch;
    if (loss->node) loss->node->mark = ep;
    for (nc_node *n = ctx->tape; n; n = n->prev) {
        if (n->mark != ep) continue;
        for (int i = 0; i < n->n_in; i++)
            if (n->in[i]->node && nc__needs_grad(n->in[i])) n->in[i]->node->mark = ep;
    }
    /* Intermediate grads start from zero; leaf grads keep accumulating. */
    for (nc_node *n = ctx->tape; n; n = n->prev)
        if (n->mark == ep && n->out->grad)
            memset(n->out->grad->data, 0, (size_t)nc_numel(n->out) * sizeof(float));

    nc_tensor *g = nc__grad_buf(loss);
    if (!g) return ctx->status;
    ((float *)g->data)[0] += 1.0f;

    for (nc_node *n = ctx->tape; n; n = n->prev)
        if (n->mark == ep && n->out->grad) n->op->backward(n);
    return ctx->status;
}

void nc_zero_grad(nc_tensor *t) {
    if (t && t->grad) memset(t->grad->data, 0, (size_t)nc_numel(t) * sizeof(float));
}
