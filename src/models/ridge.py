"""Model A -- Ridge on all 1150 features.

The design changed after the survey: PLAN.md 2.8 showed the feature set is *not*
redundant (258 PCs for 90% of variance, median |corr| between features = 0.021),
so compressing to a couple hundred cluster representatives would throw away real
signal. Instead we keep every feature and let a large alpha do the shrinking.

Fitting is done from accumulated second moments rather than a design matrix: the
full float32 matrix would be ~7 GB and this machine cannot hold it. Streaming the
Gram costs 1150x1150 floats regardless of how many rows we train on, which also
means the same code works on the cloud's larger dataset.
"""
from __future__ import annotations

import numpy as np


class Gram:
    """Accumulates X'X, X'y and column sums so ridge can be solved in closed form."""

    def __init__(self, n_features: int):
        self.d = n_features
        self.XtX = np.zeros((n_features, n_features), dtype=np.float64)
        self.Xty = np.zeros(n_features, dtype=np.float64)
        self.xsum = np.zeros(n_features, dtype=np.float64)
        self.ysum = 0.0
        self.n = 0

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        Xf = np.asarray(X, dtype=np.float32)
        yf = np.asarray(y, dtype=np.float32)
        self.XtX += (Xf.T @ Xf).astype(np.float64)
        self.Xty += (Xf.T @ yf).astype(np.float64)
        self.xsum += Xf.sum(axis=0, dtype=np.float64)
        self.ysum += float(yf.sum(dtype=np.float64))
        self.n += len(Xf)

    def copy(self) -> "Gram":
        g = Gram(self.d)
        g.XtX, g.Xty = self.XtX.copy(), self.Xty.copy()
        g.xsum, g.ysum, g.n = self.xsum.copy(), self.ysum, self.n
        return g

    def solve(self, alpha: float) -> np.ndarray:
        """Ridge coefficients on centred data. The intercept is irrelevant to Pearson."""
        n = self.n
        xbar, ybar = self.xsum / n, self.ysum / n
        # Centring identities: (X-xbar)'(X-xbar) = X'X - n xbar xbar'
        A = self.XtX - n * np.outer(xbar, xbar)
        b = self.Xty - n * xbar * ybar
        A.flat[:: self.d + 1] += alpha
        return np.linalg.solve(A, b)


def fit_expanding(cache, train_ends: list[np.ndarray], alphas: list[float]) -> dict:
    """Fit ridge for several expanding training windows in a single pass over the data.

    `train_ends` must be expanding (each a prefix of the next), which is what
    cv.walk_forward and cv.gap_folds produce.

    Returns {(last_train_moon, alpha): beta}.
    """
    boundaries = sorted({int(m[-1]) for m in train_ends})
    g = Gram(len(cache.features))
    out: dict = {}
    todo = list(boundaries)

    for moon in cache.moon_values:
        moon = int(moon)
        if moon > boundaries[-1]:
            break
        sl = cache.rows(moon)
        g.add(cache.X[sl], cache.target[sl])
        if todo and moon == todo[0]:
            for alpha in alphas:
                out[(moon, alpha)] = g.solve(alpha)
            todo.pop(0)
    return out


def predict(cache, moons: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (prediction, target, moon) stacked over the given moons."""
    preds, targets, moon_ids = [], [], []
    for m in moons:
        sl = cache.rows(int(m))
        X = cache.X[sl].astype(np.float32)
        preds.append(X @ beta.astype(np.float32))
        targets.append(cache.target[sl])
        moon_ids.append(np.full(X.shape[0], int(m), dtype=np.int32))
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(moon_ids)
