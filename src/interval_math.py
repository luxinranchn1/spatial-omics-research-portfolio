"""Established interval formulas and exact low-rank pooled Ridge algebra."""
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def split_quantile(scores, coverage=.9):
    n = len(scores)
    rank = math.ceil((n + 1) * coverage)
    return np.full(scores.shape[1:], np.inf) if rank > n else np.partition(scores, rank - 1, axis=0)[rank - 1]


def weighted_quantiles(scores, calibration_weights, query_weights, coverage=.9):
    """Unnormalized masses, including the query's atom at positive infinity."""
    order = np.argsort(scores, axis=0, kind='stable')
    ordered_scores = np.take_along_axis(scores, order, axis=0)
    cumulative = np.cumsum(np.asarray(calibration_weights)[order], axis=0)
    thresholds = coverage * (np.sum(calibration_weights) + np.asarray(query_weights))
    out = np.full((len(query_weights), scores.shape[1]), np.inf)
    for gene in range(scores.shape[1]):
        indices = np.searchsorted(cumulative[:, gene], thresholds, side='left')
        finite = indices < len(scores)
        out[finite, gene] = ordered_scores[indices[finite], gene]
    return out


def cvplus_bounds(fold_predictions, residuals, folds, alpha=.1):
    """CV+ order statistics of each held-out point's fold-specific prediction."""
    n = len(residuals)
    lower_rank, upper_rank = math.floor((n + 1) * alpha), math.ceil((n + 1) * (1 - alpha))
    shape = fold_predictions.shape[1:]
    lo, hi = np.full(shape, -np.inf), np.full(shape, np.inf)
    # Avoid allocating n * all query points * all genes at once.
    for first in range(0, shape[0], 128):
        predictions = fold_predictions[folds, first:first + 128]
        if lower_rank >= 1:
            lo[first:first + 128] = np.partition(predictions - residuals[:, None], lower_rank - 1, axis=0)[lower_rank - 1]
        if upper_rank <= n:
            hi[first:first + 128] = np.partition(predictions + residuals[:, None], upper_rank - 1, axis=0)[upper_rank - 1]
    return lo, hi


class PooledKernel:
    """Exact prediction-only Woodbury solve, with one source Cholesky factor."""
    def __init__(self, statistics, legacy):
        s = statistics
        self.ms = s['feature_mean'] if legacy == 'breast' else np.zeros_like(s['feature_mean'])
        self.ys = s['target_mean']
        covariance = .5 * (s['second'] - np.outer(self.ms, self.ms))
        covariance = covariance.copy()
        covariance.flat[::len(covariance) + 1] += 1000 / int(s['n_source'])
        self.factor = cho_factor(covariance, lower=True, check_finite=False)
        self.base_cross = .5 * (s['cross'] - np.outer(self.ms, self.ys))

    def bind(self, z):
        r = np.concatenate([z, self.ms[None]], axis=0)
        cinv_r = cho_solve(self.factor, r.T, check_finite=False)
        return BoundPooledKernel(r @ cinv_r, r @ cho_solve(self.factor, self.base_cross, check_finite=False), self.ys)


class BoundPooledKernel:
    def __init__(self, gram, base_prediction, source_mean_y):
        self.h, self.t, self.ys = gram, base_prediction, source_mean_y
        self.source_index = len(gram) - 1

    def predict(self, labels, y, queries):
        labels, queries = np.asarray(labels), np.asarray(queries)
        n, h, ms = len(labels), self.h, self.source_index
        yt = y.mean(axis=0)
        yc, dy = y - yt, self.ys - yt
        hll = h[np.ix_(labels, labels)]
        hlm = h[labels, ms]
        centered = hll - hll.mean(0)[None] - hll.mean(1)[:, None] + hll.mean()
        cross = hlm - hll.mean(1) - hlm.mean() + hll.mean()
        bottom = h[ms, ms] - 2 * hlm.mean() + hll.mean()
        a = np.sqrt(.5 / n)
        small = np.eye(n + 1)
        small[:n, :n] += a * a * centered
        small[:n, n] = small[n, :n] = a * .5 * cross
        small[n, n] += .25 * bottom
        rows = np.concatenate([queries, labels, [ms]])
        hql = h[np.ix_(rows, labels)]
        hcenter = hql - hql.mean(1)[:, None]
        hds = h[rows, ms] - hql.mean(1)
        v0 = self.t[rows] + .5 / n * (hcenter @ yc) + .25 * hds[:, None] * dy
        nv = len(queries)
        vl = v0[nv:nv + n]
        vm = v0[-1]
        rhs = np.concatenate([a * (vl - vl.mean(0)), (.5 * (vm - vl.mean(0)))[None]])
        correction = cho_solve(cho_factor(small, lower=True, check_finite=False), rhs, check_finite=False)
        v = v0 - np.column_stack([a * hcenter, .5 * hds]) @ correction
        intercept = .5 * (self.ys + yt) - .5 * (v[-1] + v[nv:nv + n].mean(0))
        return v[:nv] + intercept


def selftest():
    rng = np.random.default_rng(19)
    # Include a deliberately nonzero source mean to exercise both legacy cases.
    xs, ys = rng.normal(size=(100, 12)) + .8, rng.normal(size=(100, 4))
    z, y = rng.normal(size=(60, 12)), rng.normal(size=(15, 4))
    stat = dict(second=xs.T @ xs / 100, cross=xs.T @ ys / 100,
                feature_mean=xs.mean(0), target_mean=ys.mean(0), n_source=100)
    errors = []
    for legacy in ['breast', 'multicancer']:
        kernel = PooledKernel(stat, legacy)
        labels = np.arange(15)
        mx, my = .5 * (kernel.ms + z[labels].mean(0)), .5 * (kernel.ys + y.mean(0))
        gram = .5 * (stat['second'] + z[labels].T @ z[labels] / 15) - np.outer(mx, mx) + 10 * np.eye(12)
        cross = .5 * (stat['cross'] + z[labels].T @ y / 15) - np.outer(mx, my)
        beta = np.linalg.solve(gram, cross)
        expected = z @ beta + my - mx @ beta
        error = np.max(np.abs(expected - kernel.bind(z).predict(labels, y, np.arange(60))))
        assert error < 1e-10, error
        errors.append(float(error))
    scores = np.arange(1, 11, dtype=float)[:, None]
    w = weighted_quantiles(scores, np.ones(10), np.array([1., 100.]))
    assert w[0, 0] == 10 and np.isinf(w[1, 0])
    resid = np.arange(1, 51, dtype=float)[:, None]
    folds = np.arange(50) % 5
    constant = np.zeros((5, 3, 1))
    lo, hi = cvplus_bounds(constant, resid, folds)
    assert np.all(lo == -46) and np.all(hi == 46)
    block_lo, block_hi = cvplus_bounds(constant, np.ones((5, 1)), np.arange(5))
    assert np.isneginf(block_lo).all() and np.isposinf(block_hi).all()
    return dict(algebra_max_errors=errors, infinity_atom=True, cvplus_finite_ranks=True)


if __name__ == '__main__':
    import json
    print(json.dumps(selftest()))
