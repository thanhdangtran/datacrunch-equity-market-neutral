"""Model D -- two-stage exploitation of the 88% exact-zero target.

PLAN.md 2.3: 88.2% of targets are exactly 0; the rest carry mass out to +-1.
Models A/B/C all fit E[y|x] directly and never see that structure explicitly.
The rank-prediction experiment on Model B (JOURNAL.md 2026-07-28) showed the
edge lives in correctly flagging the extreme movers, not in the bulk -- which
is exactly what a two-stage model is built to exploit directly:

  Stage 1 (classifier): P(target != 0 | x)
  Stage 2 (regressor):  E[target | x, target != 0], fit only on the ~12% of
                         rows with a nonzero target
  Combined:             pred = P(nonzero) * E[target | nonzero]

Both stages use max_bin=7 (lossless for the 7 feature levels) and no row
bagging (rows within a moon are not independent -- PLAN.md 3.3).
"""
from __future__ import annotations

import numpy as np

CLF_PARAMS = dict(
    objective="binary",
    max_bin=7,
    num_leaves=15,
    max_depth=5,
    learning_rate=0.03,
    feature_fraction=0.15,
    bagging_fraction=1.0,
    min_child_samples=500,
    lambda_l2=50.0,
    verbose=-1,
    num_threads=0,
    deterministic=True,
    force_col_wise=True,
    seed=42,
    feature_fraction_seed=42,
)

REG_PARAMS = dict(
    objective="regression",
    max_bin=7,
    num_leaves=15,
    max_depth=5,
    learning_rate=0.03,
    feature_fraction=0.15,
    bagging_fraction=1.0,
    min_child_samples=200,   # smaller: stage 2 only sees ~12% of rows
    lambda_l2=50.0,
    verbose=-1,
    num_threads=0,
    deterministic=True,
    force_col_wise=True,
    seed=42,
    feature_fraction_seed=42,
)

CLF_ESTIMATORS = 500
REG_ESTIMATORS = 500


def stack(cache, moons: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (X, y, moon_id) float32 design matrix for the given moons."""
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
    return moons[::-1][::stride][::-1]


def fit(cache, train_moons: np.ndarray, stride: int,
        clf_params: dict = CLF_PARAMS, reg_params: dict = REG_PARAMS,
        clf_estimators: int = CLF_ESTIMATORS, reg_estimators: int = REG_ESTIMATORS):
    import lightgbm as lgb

    sel = subsample_moons(train_moons, stride)
    X, y, _ = stack(cache, sel)

    nonzero = y != 0
    clf_ds = lgb.Dataset(X, label=nonzero.astype(np.float32), free_raw_data=True)
    classifier = lgb.train(clf_params, clf_ds, num_boost_round=clf_estimators)
    del clf_ds

    reg_ds = lgb.Dataset(X[nonzero], label=y[nonzero], free_raw_data=True)
    regressor = lgb.train(reg_params, reg_ds, num_boost_round=reg_estimators)
    del X, y, reg_ds

    return classifier, regressor


def predict_raw(cache_or_X, classifier, regressor, moons=None) -> np.ndarray:
    """Accepts either (cache, moons) or a bare (n, d) float32 array."""
    if moons is not None:
        preds = []
        for m in moons:
            sl = cache_or_X.rows(int(m))
            X = cache_or_X.X[sl].astype(np.float32)
            preds.append(classifier.predict(X) * regressor.predict(X))
        return np.concatenate(preds)
    X = cache_or_X
    return classifier.predict(X) * regressor.predict(X)


def predict(cache, moons: np.ndarray, classifier, regressor):
    """-> (prediction, target, moon) stacked over the given moons, for the CV harness."""
    preds, targets, mids = [], [], []
    for m in moons:
        sl = cache.rows(int(m))
        X = cache.X[sl].astype(np.float32)
        preds.append(classifier.predict(X) * regressor.predict(X))
        targets.append(cache.target[sl])
        mids.append(np.full(X.shape[0], int(m), dtype=np.int32))
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(mids)
