"""Model B -- LightGBM with max_bin=7.

Deliberately *not* a replacement for Model A. Local CV over matched folds:

    Model A (ridge)  mean_ic 0.0329  ic_std 0.0264  Sharpe 1.25
    Model B (this)   mean_ic 0.0429  ic_std 0.0400  Sharpe 1.07

Higher mean, worse Sharpe. It sits on its own slot because the reward curve is
e**20 -- only the top few percent of a given week pay, so a decorrelated model
with a higher mean is worth an independent ticket even though its variance is
larger. Two experiments already ruled out taming that variance:

  * num_leaves 15 -> 7: mean_ic fell to 0.0400, ic_std unchanged at 0.0401.
  * rank-transforming predictions per moon: ic_std fell to 0.0298 but mean_ic
    collapsed to 0.0262.

The variance is where the edge lives -- the target is 88% exact zeros with mass
at +-1, so the signal *is* spotting the extreme movers. Do not smooth it.

max_bin=7 matches the seven feature levels exactly, so binning is lossless.
Unlike the local runs, this trains on every moon: local RAM forced a stride-4
subsample, the cloud box has 120 GB.
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(
    objective="regression",          # L2 on the raw target -- never a ranked one
    max_bin=7,                       # exactly the number of feature levels
    num_leaves=15,
    max_depth=5,
    learning_rate=0.02,
    feature_fraction=0.15,
    bagging_fraction=1.0,            # row bagging would leak within a cross-section
    min_child_samples=500,
    lambda_l2=50.0,
    verbose=-1,
    num_threads=0,
    deterministic=True,
    force_col_wise=True,
    seed=42,
    feature_fraction_seed=42,
)
N_ESTIMATORS = 800


def get_model_path(model_directory_path: str) -> str:
    return os.path.join(model_directory_path, "model.txt")


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

    dataset = lgb.Dataset(X, label=y, free_raw_data=True)
    booster = lgb.train(PARAMS, dataset, num_boost_round=N_ESTIMATORS)
    booster.save_model(get_model_path(model_directory_path))


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
    # loop_moon: int = None,
    # embargo: int = None,
) -> pd.DataFrame:
    booster = lgb.Booster(model_file=get_model_path(model_directory_path))

    features = booster.feature_name()
    # LightGBM names columns Column_0.. when trained from a bare array, so fall
    # back to the frame's own feature columns in that case.
    if not features or features[0].startswith("Column_"):
        features = get_feature_columns(X_test)

    raw = booster.predict(X_test[features].to_numpy(dtype=np.float32))

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
