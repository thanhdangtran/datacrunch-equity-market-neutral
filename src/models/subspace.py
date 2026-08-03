"""Model F -- ridge restricted to one sign-family of features.

Not an attempt to beat A -- A already uses all 1150 features and §2.8a showed
the feature set is not redundant, so throwing features away can only lose
signal. The point here is portfolio-level, not per-model: JOURNAL.md
2026-07-29 found corr(B, D) = 0.835, so two of the four slots are close to
the same bet. This tests whether restricting the *same* linear/L2 machinery
to one sign-family of features (§2.8b: a stable negative "reversal" family
around Feature_1-7/43-49, and a stable positive family around
Feature_64-99/1087-1096) produces a prediction that is genuinely less
correlated with full-feature Ridge (A) -- a different question from "is it
individually stronger".

Feature selection is computed from the training window only, per fold --
using full-history IC to pick features would leak future information into
early folds.
"""
from __future__ import annotations

import numpy as np

from src.models.ridge import Gram

ALPHA = 1e5  # smaller than A's 1e6: fewer, more homogeneous features need less shrinkage


def feature_ic(cache, moons: np.ndarray) -> np.ndarray:
    """Per-feature Pearson IC against target, averaged over the given moons."""
    ic = np.zeros(len(cache.features), dtype=np.float64)
    n = 0
    for m in moons:
        sl = cache.rows(int(m))
        y = cache.target[sl].astype(np.float32)
        yc = y - y.mean()
        y_norm = np.linalg.norm(yc)
        if y_norm == 0:
            continue
        x = cache.X[sl].astype(np.float32)
        x = x - x.mean(axis=0, keepdims=True)
        x_norm = np.linalg.norm(x, axis=0)
        x_norm[x_norm == 0] = np.inf
        ic += (x.T @ yc) / (x_norm * y_norm)
        n += 1
    return ic / max(n, 1)


def select(cache, train_moons: np.ndarray, family: str) -> np.ndarray:
    """-> sorted column indices whose train-window mean IC has the requested sign."""
    assert family in ("pos", "neg")
    ic = feature_ic(cache, train_moons)
    idx = np.where(ic > 0)[0] if family == "pos" else np.where(ic < 0)[0]
    return np.sort(idx)


def fit(cache, train_moons: np.ndarray, family: str, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """-> (feature_idx, beta). feature_idx picked from train_moons only (no leakage)."""
    feature_idx = select(cache, train_moons, family)
    g = Gram(len(feature_idx))
    for m in train_moons:
        sl = cache.rows(int(m))
        g.add(cache.X[sl][:, feature_idx], cache.target[sl])
    beta = g.solve(alpha)
    return feature_idx, beta


def predict_raw(cache_or_X, feature_idx: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X = cache_or_X[:, feature_idx].astype(np.float32)
    return X @ beta.astype(np.float32)


def predict(cache, moons: np.ndarray, feature_idx: np.ndarray, beta: np.ndarray):
    """-> (prediction, target, moon) stacked over the given moons, for the CV harness."""
    preds, targets, mids = [], [], []
    for m in moons:
        sl = cache.rows(int(m))
        preds.append(predict_raw(cache.X[sl], feature_idx, beta))
        targets.append(cache.target[sl])
        mids.append(np.full(sl.stop - sl.start, int(m), dtype=np.int32))
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(mids)
