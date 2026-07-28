"""Model B -- LightGBM with max_bin=7.

The features take exactly 7 values, so max_bin=7 reproduces them exactly: the
binning step is lossless and much cheaper than the default 255 bins.

Memory: the full training matrix is ~6.9 GB in float32 and this box has ~3.5 GB
free, so we subsample *whole moons* (stride) rather than rows. Sampling rows
would split a cross-section, which PLAN.md 3.3 rules out -- rows inside a moon
are not independent.
"""
from __future__ import annotations

import numpy as np

BASE_PARAMS = dict(
    objective="regression",          # L2 on the raw target -- PLAN.md 3.2
    max_bin=7,                       # exactly the number of feature levels
    num_leaves=15,
    max_depth=5,
    learning_rate=0.02,
    feature_fraction=0.15,
    bagging_fraction=1.0,            # no row bagging: it would leak within a moon
    min_child_samples=500,
    lambda_l2=50.0,
    verbose=-1,
    num_threads=0,
    deterministic=True,
    force_col_wise=True,
    seed=42,
    feature_fraction_seed=42,
)


def stack(cache, moons: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a float32 design matrix for the given moons. -> (X, y, moon_id)"""
    rows = sum(cache.rows(int(m)).stop - cache.rows(int(m)).start for m in moons)
    X = np.empty((rows, len(cache.features)), dtype=np.float32)
    y = np.empty(rows, dtype=np.float32)
    mid = np.empty(rows, dtype=np.int32)
    at = 0
    for m in moons:
        sl = cache.rows(int(m))
        n = sl.stop - sl.start
        X[at : at + n] = cache.X[sl]
        y[at : at + n] = cache.target[sl]
        mid[at : at + n] = int(m)
        at += n
    return X, y, mid


def subsample_moons(moons: np.ndarray, stride: int) -> np.ndarray:
    """Keep every `stride`-th moon, anchored at the most recent one."""
    return moons[::-1][::stride][::-1]


def fit(cache, train_moons: np.ndarray, stride: int, params: dict,
        n_estimators: int) -> "object":
    import lightgbm as lgb

    sel = subsample_moons(train_moons, stride)
    X, y, _ = stack(cache, sel)
    ds = lgb.Dataset(X, label=y, free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=n_estimators)
    del X, y, ds
    return booster


def predict(cache, moons: np.ndarray, booster) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, targets, mids = [], [], []
    for m in moons:
        sl = cache.rows(int(m))
        X = cache.X[sl].astype(np.float32)
        preds.append(booster.predict(X).astype(np.float32))
        targets.append(cache.target[sl])
        mids.append(np.full(X.shape[0], int(m), dtype=np.int32))
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(mids)
