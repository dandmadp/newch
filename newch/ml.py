"""Classic machine learning with scikit-learn's interface (fit / predict / transform).

    from newch import ml
    X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = ml.StandardScaler().fit(X_train)
    model = ml.Ridge(alpha=1.0).fit(scaler.transform(X_train), y_train)
    print(model.score(scaler.transform(X_test), y_test))

Heavy steps (X^T X, distance matrices, cluster sums) run on the engine's
GEMM in float32, inside a private context whose memory is released when the
call returns, so they never touch the caller's step arena.

Results follow the input type: numpy in -> numpy out, Tensor in -> Tensor
out (temporary, in the input's context), anything else -> nested lists.
Fitted attributes (``mean_``, ``coef_``, ...) are numpy arrays when ``fit``
got numpy input, otherwise lists.
"""

from __future__ import annotations

import array
import bisect
import itertools
import math
import random

from .tensor import Context, Tensor, _is_numpy, float32, int32, persistent, tensor, tensor_from_bytes

__all__ = [
    "KMeans", "LinearRegression", "MinMaxScaler", "NotFittedError", "Ridge", "StandardScaler", "accuracy_score",
    "mean_squared_error", "r2_score", "train_test_split",
]

_EPS32 = 1.1920929e-07


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit`` (as in scikit-learn)."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class _Work:
    """A private engine context for one call; all its memory goes at once on exit."""

    def __enter__(self) -> "_Work":
        self.ctx = Context()
        return self

    def __exit__(self, *exc) -> None:
        self.ctx.close()

    def matrix(self, X, allow_1d: bool = False) -> Tensor:
        """Persistent float32 copy of X: [n, d], or [n, 1] for 1-D input when allowed."""
        with persistent(self.ctx):
            if isinstance(X, Tensor):
                flat = X._flat()
                if X.dtype != float32:
                    flat = array.array("f", flat)
                t = tensor_from_bytes(flat.tobytes(), X.shape, float32, self.ctx)
            else:
                t = tensor(X, dtype=float32, ctx=self.ctx)
            if t.ndim == 1 and allow_1d:
                t = t.reshape(t.shape[0], 1)
        if t.ndim != 2:
            raise ValueError(f"need a 2-D [samples, features] array, got {t.ndim}-D")
        return t

    def vector(self, values) -> Tensor:
        return tensor(values, dtype=float32, ctx=self.ctx)


def _out(t: Tensor, like):
    """Converts a result computed in a _Work context to the caller's type."""
    if _is_numpy(like):
        return t.numpy()
    if isinstance(like, Tensor):
        return tensor_from_bytes(t._flat().tobytes(), t.shape, t.dtype, like.ctx)
    return t.tolist()


def _attr(values: list, like, dtype: str = float32):
    """Fitted attribute: numpy when the fit input was numpy, else the list itself."""
    if _is_numpy(like):
        import numpy as np

        return np.asarray(values, dtype=np.float32 if dtype == float32 else np.int32)
    return values


def _as_list(v) -> list:
    if _is_numpy(v) or isinstance(v, Tensor):
        return v.tolist()
    return list(v)


def _nonzero_scale(s: float) -> float:
    # scikit-learn's _handle_zeros_in_scale: constant features are left unscaled
    return 1.0 if s < 10 * _EPS32 else s


def _rng(random_state) -> random.Random:
    if isinstance(random_state, random.Random):
        return random_state
    if random_state is None or isinstance(random_state, int):
        return random.Random(random_state)
    raise TypeError("random_state must be None, an int or a random.Random")


class _Estimator:
    _fitted_attr = ""
    n_features_in_: int

    def _check_fitted(self) -> None:
        if not hasattr(self, self._fitted_attr):
            raise NotFittedError(f"{type(self).__name__} has not been fitted; call fit() first")

    def _check_features(self, t: Tensor) -> None:
        d = t.shape[1]
        if d != self.n_features_in_:
            raise ValueError(f"{type(self).__name__} was fitted on {self.n_features_in_} features, got {d}")

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.endswith("_") and not k.startswith("_"))
        return f"{type(self).__name__}({args})"


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


class StandardScaler(_Estimator):
    """(X - mean_) / scale_ per feature, population std (sklearn.preprocessing.StandardScaler)."""

    _fitted_attr = "n_samples_seen_"

    def __init__(self, *, copy: bool = True, with_mean: bool = True, with_std: bool = True):
        if not copy:
            raise NotImplementedError("in-place scaling (copy=False) is not supported")
        self.copy, self.with_mean, self.with_std = copy, with_mean, with_std

    def fit(self, X, y=None) -> "StandardScaler":
        with _Work() as w:
            t = w.matrix(X)
            mean = t.mean(0)
            diff = t - mean
            mean_l, var_l = mean.tolist(), (diff * diff).mean(0).tolist()
            shape = t.shape  # read while the work context is alive
        self.n_samples_seen_, self.n_features_in_ = shape
        self.mean_ = _attr(mean_l, X) if (self.with_mean or self.with_std) else None
        self.var_ = _attr(var_l, X) if self.with_std else None
        self.scale_ = _attr([_nonzero_scale(math.sqrt(v)) for v in var_l], X) if self.with_std else None
        return self

    def transform(self, X):
        self._check_fitted()
        with _Work() as w:
            t = w.matrix(X)
            self._check_features(t)
            if self.with_mean:
                t = t - w.vector(self.mean_)
            if self.with_std:
                t = t / w.vector(self.scale_)
            return _out(t, X)

    def inverse_transform(self, X):
        self._check_fitted()
        with _Work() as w:
            t = w.matrix(X)
            self._check_features(t)
            if self.with_std:
                t = t * w.vector(self.scale_)
            if self.with_mean:
                t = t + w.vector(self.mean_)
            return _out(t, X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class MinMaxScaler(_Estimator):
    """Scales each feature to ``feature_range`` (sklearn.preprocessing.MinMaxScaler)."""

    _fitted_attr = "n_samples_seen_"

    def __init__(self, feature_range=(0, 1), *, copy: bool = True, clip: bool = False):
        if not copy:
            raise NotImplementedError("in-place scaling (copy=False) is not supported")
        if clip:
            raise NotImplementedError("clip=True is not supported")
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range needs low < high, got {feature_range}")
        self.feature_range, self.copy, self.clip = (lo, hi), copy, clip

    def fit(self, X, y=None) -> "MinMaxScaler":
        with _Work() as w:
            t = w.matrix(X)
            dmax = t.amax(0).tolist()
            dmin = [-v for v in (t * -1.0).amax(0).tolist()]
            shape = t.shape
        lo, hi = self.feature_range
        rng = [b - a for a, b in zip(dmin, dmax)]
        scale = [(hi - lo) / _nonzero_scale(r) for r in rng]
        self.n_samples_seen_, self.n_features_in_ = shape
        self.data_min_, self.data_max_, self.data_range_ = _attr(dmin, X), _attr(dmax, X), _attr(rng, X)
        self.scale_ = _attr(scale, X)
        self.min_ = _attr([lo - a * s for a, s in zip(dmin, scale)], X)
        return self

    def transform(self, X):
        self._check_fitted()
        with _Work() as w:
            t = w.matrix(X)
            self._check_features(t)
            return _out(t * w.vector(self.scale_) + w.vector(self.min_), X)

    def inverse_transform(self, X):
        self._check_fitted()
        with _Work() as w:
            t = w.matrix(X)
            self._check_features(t)
            return _out((t - w.vector(self.min_)) / w.vector(self.scale_), X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# Model selection and metrics
# --------------------------------------------------------------------------


def _take(a, idx: list):
    if _is_numpy(a):
        return a[idx]
    if isinstance(a, Tensor):
        return a.index_select(0, idx)
    return [a[i] for i in idx]


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle: bool = True,
                     stratify=None) -> list:
    """Splits arrays into random train and test subsets (sklearn.model_selection.train_test_split).

    Returns ``[a_train, a_test, b_train, b_test, ...]``. Lists, numpy arrays and Tensors are supported.
    """
    if not arrays:
        raise ValueError("nothing to split: pass at least one array")
    if stratify is not None:
        raise NotImplementedError("stratify is not supported")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError(f"arrays differ in length: {[len(a) for a in arrays]}")
    if test_size is None and train_size is None:
        test_size = 0.25

    def count(size, rounding):
        if isinstance(size, float):
            if not 0.0 < size < 1.0:
                raise ValueError(f"a fractional size must be between 0 and 1, got {size}")
            return rounding(size * n)
        if not 0 < size < n:
            raise ValueError(f"a sample count must be between 1 and {n - 1}, got {size}")
        return int(size)

    n_test = count(test_size, math.ceil) if test_size is not None else None
    n_train = count(train_size, math.floor) if train_size is not None else None
    if n_test is None:
        n_test = n - n_train
    if n_train is None:
        n_train = n - n_test
    if n_train + n_test > n or n_train == 0:
        raise ValueError(f"cannot split {n} samples into train and test sets "
                         f"(test_size={test_size}, train_size={train_size})")
    if shuffle:
        perm = list(range(n))
        _rng(random_state).shuffle(perm)
        test, train = perm[:n_test], perm[n_test:n_test + n_train]
    else:
        train, test = list(range(n_train)), list(range(n_train, n_train + n_test))
    out = []
    for a in arrays:
        out += [_take(a, train), _take(a, test)]
    return out


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of exactly matching labels (sklearn.metrics.accuracy_score)."""
    t, p = _as_list(y_true), _as_list(y_pred)
    if len(t) != len(p):
        raise ValueError("y_true and y_pred have different lengths")
    return sum(1 for a, b in zip(t, p) if a == b) / max(len(t), 1)


def _columns(v: list) -> list:
    return [list(c) for c in zip(*v)] if v and isinstance(v[0], list) else [v]


def mean_squared_error(y_true, y_pred) -> float:
    """Mean of squared errors, averaged over outputs (sklearn.metrics.mean_squared_error)."""
    ct, cp = _columns(_as_list(y_true)), _columns(_as_list(y_pred))
    per = [sum((a - b) ** 2 for a, b in zip(t, p)) / max(len(t), 1) for t, p in zip(ct, cp)]
    return sum(per) / len(per)


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination, averaged over outputs (sklearn.metrics.r2_score)."""
    scores = []
    for t, p in zip(_columns(_as_list(y_true)), _columns(_as_list(y_pred))):
        mean = sum(t) / max(len(t), 1)
        ss_res = sum((a - b) ** 2 for a, b in zip(t, p))
        ss_tot = sum((a - mean) ** 2 for a in t)
        scores.append((1.0 if ss_res == 0 else 0.0) if ss_tot == 0 else 1.0 - ss_res / ss_tot)
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------
# Linear models (closed form)
# --------------------------------------------------------------------------


def _cholesky_solve(A: list, B: list, rel_tol: float, singular_msg: str) -> list:
    """Solves A W = B for symmetric positive definite A (d x d) and B (d x k), in double precision."""
    d = len(A)
    Lm = [[0.0] * d for _ in range(d)]
    top = max((A[i][i] for i in range(d)), default=0.0)
    for j in range(d):
        s = A[j][j] - sum(Lm[j][q] ** 2 for q in range(j))
        if s <= 0.0 or s <= rel_tol * top:
            raise ValueError(singular_msg)
        Lm[j][j] = math.sqrt(s)
        for i in range(j + 1, d):
            Lm[i][j] = (A[i][j] - sum(Lm[i][q] * Lm[j][q] for q in range(j))) / Lm[j][j]
    k = len(B[0]) if B else 0
    W = [[0.0] * k for _ in range(d)]
    for c in range(k):
        y = [0.0] * d
        for i in range(d):
            y[i] = (B[i][c] - sum(Lm[i][q] * y[q] for q in range(i))) / Lm[i][i]
        for i in reversed(range(d)):
            W[i][c] = (y[i] - sum(Lm[q][i] * W[q][c] for q in range(i + 1, d))) / Lm[i][i]
    return W


class _LinearModel(_Estimator):
    _fitted_attr = "coef_"
    fit_intercept = True

    def _solve(self, X, y, alpha: float, rel_tol: float, singular_msg: str):
        y1d = not (_is_numpy(y) and y.ndim == 2) and not (isinstance(y, Tensor) and y.ndim == 2) \
            and not (isinstance(y, (list, tuple)) and y and isinstance(y[0], (list, tuple)))
        with _Work() as w:
            t = w.matrix(X)
            yt = w.matrix(y, allow_1d=True)
            n, d = t.shape
            if yt.shape[0] != n:
                raise ValueError(f"X has {n} samples but y has {yt.shape[0]}")
            k = yt.shape[1]
            if self.fit_intercept:
                xm, ym = t.mean(0), yt.mean(0)
                xm_l, ym_l = xm.tolist(), ym.tolist()
                t, yt = t - xm, yt - ym
            else:
                xm_l, ym_l = [0.0] * d, [0.0] * k
            G = (t.T @ t).tolist()
            B = (t.T @ yt).tolist()
        for i in range(d):
            G[i][i] += alpha
        W = _cholesky_solve(G, B, rel_tol, singular_msg)  # d x k
        b = [ym_l[c] - sum(xm_l[i] * W[i][c] for i in range(d)) for c in range(k)]
        self._W, self._b, self._y1d = W, b, y1d
        self.n_features_in_ = d
        coef = [W[i][0] for i in range(d)] if y1d else [[W[i][c] for i in range(d)] for c in range(k)]
        self.coef_ = _attr(coef, X)
        if not self.fit_intercept:
            self.intercept_ = 0.0 if y1d else _attr([0.0] * k, X)
        else:
            self.intercept_ = b[0] if y1d else _attr(b, X)
        return self

    def predict(self, X):
        self._check_fitted()
        with _Work() as w:
            t = w.matrix(X)
            self._check_features(t)
            out = t @ w.vector(self._W) + w.vector(self._b)
            if self._y1d:
                out = out.reshape(-1)
            return _out(out, X)

    def score(self, X, y) -> float:
        """R^2 of the prediction (as scikit-learn's regressors)."""
        return r2_score(y, self.predict(X))


class LinearRegression(_LinearModel):
    """Ordinary least squares via the normal equations (sklearn.linear_model.LinearRegression).

    Solved in closed form, so it suits up to a few hundred features. Rank-deficient
    X (collinear features) raises instead of returning a minimum-norm solution; use Ridge.
    """

    def __init__(self, *, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y) -> "LinearRegression":
        return self._solve(X, y, 0.0, 1e-5, "X is rank deficient (collinear or constant features); "
                                            "LinearRegression needs full column rank, use Ridge instead")


class Ridge(_LinearModel):
    """Least squares with L2 penalty alpha * ||w||^2; the intercept is not penalized (sklearn.linear_model.Ridge)."""

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha, self.fit_intercept = alpha, fit_intercept

    def fit(self, X, y) -> "Ridge":
        return self._solve(X, y, float(self.alpha), 0.0, "X^T X + alpha I is singular; increase alpha")


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def _sq_dist(t: Tensor, xx: Tensor, centers: list, ctx: Context) -> Tensor:
    """||x - c||^2 for every row of t and every center: [n, k] = |x|^2 - 2 x.c + |c|^2."""
    c = tensor(centers, dtype=float32, ctx=ctx)
    return xx - (t @ c.T) * 2.0 + (c * c).sum(1)


def _assign(t: Tensor, xx: Tensor, centers: list, ctx: Context):
    neg = _sq_dist(t, xx, centers, ctx) * -1.0
    labels = neg.argmax(1)._flat()  # first nearest center on ties
    mind = [max(0.0, -v) for v in neg.amax(1)._flat()]
    return labels, mind


class KMeans(_Estimator):
    """Lloyd's algorithm with greedy k-means++ seeding (sklearn.cluster.KMeans).

    Same algorithm, tolerance rule and empty-cluster handling as scikit-learn;
    the random streams differ, so seeds give different (equally valid) runs.
    """

    _fitted_attr = "cluster_centers_"

    def __init__(self, n_clusters: int = 8, *, init="k-means++", n_init="auto", max_iter: int = 300,
                 tol: float = 1e-4, random_state=None):
        if isinstance(init, str) and init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++', 'random' or an array of initial centers")
        self.n_clusters, self.init, self.n_init = n_clusters, init, n_init
        self.max_iter, self.tol, self.random_state = max_iter, tol, random_state

    def _runs(self) -> int:
        if self.n_init == "auto":
            return 10 if self.init == "random" else 1
        if not isinstance(self.init, str) and self.n_init != 1:
            raise ValueError("explicit initial centers need n_init=1")
        return int(self.n_init)

    def _seed(self, t: Tensor, xx: Tensor, rng: random.Random, ctx: Context) -> list:
        n, k = t.shape[0], self.n_clusters
        if not isinstance(self.init, str):
            centers = _as_list(self.init)
            if len(centers) != k or any(len(c) != t.shape[1] for c in centers):
                raise ValueError(f"init must have shape ({k}, {t.shape[1]})")
            return [[float(v) for v in c] for c in centers]
        if self.init == "random":
            return t.index_select(0, rng.sample(range(n), k)).tolist()
        # greedy k-means++: several candidates per step, keep the one that lowers the potential most
        trials = 2 + int(math.log(k))
        centers = t.index_select(0, [rng.randrange(n)]).tolist()
        closest = [max(0.0, v) for v in _sq_dist(t, xx, centers, ctx)._flat()]
        for _ in range(1, k):
            cum = list(itertools.accumulate(closest))
            pot = cum[-1]
            cand = [min(bisect.bisect_left(cum, rng.random() * pot), n - 1) for _ in range(trials)]
            rows = t.index_select(0, cand).tolist()
            cl = tensor(closest, dtype=float32, ctx=ctx).reshape(n, 1)
            d = _sq_dist(t, xx, rows, ctx)
            newd = cl - (cl - d).relu()  # min(closest, d) per candidate
            pots = newd.sum(0).tolist()
            best = min(range(trials), key=pots.__getitem__)
            closest = [max(0.0, v) for v in newd._flat()[best::trials]]
            centers.append(rows[best])
            ctx.step_end()
        return centers

    def _update(self, t: Tensor, labels, mind: list, ctx: Context) -> list:
        n, k = t.shape[0], self.n_clusters
        onehot = array.array("f", bytes(4 * n * k))
        counts = [0] * k
        for i, lab in enumerate(labels):
            onehot[i * k + lab] = 1.0
            counts[lab] += 1
        sums = (tensor_from_bytes(onehot.tobytes(), (n, k), float32, ctx).T @ t).tolist()
        empty = [j for j in range(k) if counts[j] == 0]
        if empty:  # relocate empty clusters to the points farthest from their centers
            far = sorted(range(n), key=lambda i: -mind[i])[:len(empty)]
            for j, row in zip(empty, t.index_select(0, far).tolist()):
                sums[j], counts[j] = row, 1
        return [[v / counts[j] for v in sums[j]] for j in range(k)]

    def _lloyd(self, t: Tensor, xx: Tensor, centers: list, tol_abs: float, ctx: Context):
        prev = None
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            labels, mind = _assign(t, xx, centers, ctx)
            new = self._update(t, labels, mind, ctx)
            shift = sum((a - b) ** 2 for ca, cb in zip(new, centers) for a, b in zip(ca, cb))
            centers = new
            ctx.step_end()
            if prev is not None and labels == prev:
                break  # strict convergence: assignments did not change
            prev = labels
            if shift <= tol_abs:
                break
        labels, mind = _assign(t, xx, centers, ctx)  # labels consistent with the final centers
        ctx.step_end()
        return list(labels), sum(mind), centers, n_iter

    def fit(self, X, y=None) -> "KMeans":
        rng = _rng(self.random_state)
        with _Work() as w:
            ctx = w.ctx
            t = w.matrix(X)
            n, d = t.shape
            if n < self.n_clusters:
                raise ValueError(f"{self.n_clusters} clusters need at least as many samples, got {n}")
            with persistent(ctx):
                xx = (t * t).sum(1, keepdim=True)
                diff = t - t.mean(0)
                tol_abs = self.tol * (diff * diff).mean(0).mean().item()  # tol relative to the data variance
            best = None
            for _ in range(self._runs()):
                run = self._lloyd(t, xx, self._seed(t, xx, rng, ctx), tol_abs, ctx)
                if best is None or run[1] < best[1]:
                    best = run
        labels, inertia, centers, n_iter = best
        self.n_features_in_ = d
        self.cluster_centers_ = _attr(centers, X)
        self.labels_ = _attr(labels, X, int32)
        self.inertia_, self.n_iter_ = inertia, n_iter
        return self

    def _labels(self, X):
        self._check_fitted()
        with _Work() as w:
            t = w.matrix(X)
            self._check_features(t)
            xx = (t * t).sum(1, keepdim=True)
            labels, mind = _assign(t, xx, _as_list(self.cluster_centers_), w.ctx)
        return list(labels), mind

    def predict(self, X):
        labels, _ = self._labels(X)
        if isinstance(X, Tensor):
            return tensor(labels, dtype=int32, ctx=X.ctx)
        return _attr(labels, X, int32)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def score(self, X, y=None) -> float:
        """Negative sum of squared distances to the nearest center (as scikit-learn)."""
        return -sum(self._labels(X)[1])
