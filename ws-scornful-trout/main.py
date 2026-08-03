"""Model D -- two-stage exploitation of the 88% exact-zero target.

Best local CV of the four models built for this competition:

    mean_ic 0.0479  ic_std 0.0416  Sharpe 1.15  gap311 0.0375

(vs. ridge 0.0329/1.25/0.0304 and single-stage LightGBM 0.0429/1.07/0.0267;
gap311 = train, skip 311 moons, then validate -- the local->live distance.)

88.2% of targets are exactly 0; the rest carry mass out to +-1 (PLAN.md 2.3).
A single regressor fitting L2 on the raw target (Models A/B) has to spend most
of its capacity getting the flat 88% right and only implicitly learns to spot
the extreme movers. This model splits the two problems:

    Stage 1 (classifier): P(target != 0 | x)
    Stage 2 (regressor):  E[target | x, target != 0], fit only on the ~12% of
                           rows that actually moved
    Combined:              pred = P(nonzero) * E[target | nonzero]

This design was chosen *because* an earlier experiment on the LightGBM model
showed the edge lives in the extreme predictions, not the bulk: rank-transforming
its predictions per moon cut variance (0.0400 -> 0.0298) but cut mean_ic even
more (0.0429 -> 0.0262), i.e. smoothing away the extremes destroys the signal
along with the noise. Modelling the two halves of the target separately, rather
than flattening the output of one combined model, is what actually helped.

Both stages use max_bin=7 (lossless for the 7 feature levels) and no row
bagging -- rows within a moon are not independent (PLAN.md 3.3).
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

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
    min_child_samples=200,   # smaller: this stage only sees ~12% of rows
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


def _clf_path(model_directory_path: str) -> str:
    return os.path.join(model_directory_path, "classifier.txt")


def _reg_path(model_directory_path: str) -> str:
    return os.path.join(model_directory_path, "regressor.txt")


def get_feature_columns(X: pd.DataFrame) -> list:
    return [c for c in X.columns if c.startswith("Feature_")]


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
    # loop_moon: int = None,
    # embargo: int = None,
) -> None:
    features = get_feature_columns(X_train)
    X = X_train[features].to_numpy(dtype=np.float32)
    y = y_train["target"].to_numpy(dtype=np.float32)
    nonzero = y != 0

    clf_ds = lgb.Dataset(X, label=nonzero.astype(np.float32), free_raw_data=True)
    classifier = lgb.train(CLF_PARAMS, clf_ds, num_boost_round=CLF_ESTIMATORS)
    classifier.save_model(_clf_path(model_directory_path))
    del clf_ds

    reg_ds = lgb.Dataset(X[nonzero], label=y[nonzero], free_raw_data=True)
    regressor = lgb.train(REG_PARAMS, reg_ds, num_boost_round=REG_ESTIMATORS)
    regressor.save_model(_reg_path(model_directory_path))


def _feature_names(booster: lgb.Booster, fallback: pd.DataFrame) -> list:
    names = booster.feature_name()
    if not names or names[0].startswith("Column_"):
        return get_feature_columns(fallback)
    return names


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
    # loop_moon: int = None,
    # embargo: int = None,
) -> pd.DataFrame:
    classifier = lgb.Booster(model_file=_clf_path(model_directory_path))
    regressor = lgb.Booster(model_file=_reg_path(model_directory_path))

    features = _feature_names(classifier, X_test)
    Xf = X_test[features].to_numpy(dtype=np.float32)
    raw = classifier.predict(Xf) * regressor.predict(Xf)

    out = X_test[["id", "moon"]].copy()
    pred = np.zeros(len(raw), dtype=np.float32)

    # Centre then scale to peak magnitude 1, per moon. Both affine within a moon,
    # so Pearson is untouched, and the [-1, 1] requirement holds without clipping.
    for _, idx in out.groupby("moon").indices.items():
        block = raw[idx] - raw[idx].mean()
        peak = np.abs(block).max()
        if peak > 0:
            pred[idx] = block / peak
        else:
            pred[idx] = np.linspace(-1e-6, 1e-6, len(idx), dtype=np.float32)

    out["prediction"] = pred
    return out
