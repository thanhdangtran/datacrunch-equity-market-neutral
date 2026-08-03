"""Model A -- ridge on all 1150 features, fitted from streamed second moments.

Chosen by src/run_ridge.py: alpha 1e6 scored mean_ic 0.0329 / Sharpe 1.25 over six
purged walk-forward folds (6/6 positive), and 0.0304 / 1.31 across the 311-moon gap
folds that stand in for the local->live distance. The survey baseline it has to beat
was 0.0251 / 1.19.

Two deliberate choices, both from PLAN.md:

  * L2 on the *raw* target, never a ranked one. Pearson is maximised by E[y|x] and
    L2 estimates exactly that. 88% of targets are exactly 0, so rank-transforming
    the target spreads that tied mass across the whole range according to row order
    -- measured correlation between assigned rank and row position was +0.72.

  * The design matrix is never materialised. X'X is 1150x1150 no matter how many
    rows we train on, so this fits in a fixed ~11 MB regardless of how much data
    the cloud hands us, which local RAM could not have held anyway.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# Ridge strength, expressed per training row so the effective shrinkage stays put
# when the cloud trains on more moons than we have locally. Tuned as alpha = 1e6
# at n = 1_637_276 rows on the 0..6 feature scale.
ALPHA_PER_ROW = 1e6 / 1_637_276

# Features arrive as {0, .17, .33, .5, .67, .83, 1.0}; the tuning above was done on
# the integer codes {0..6}, and ridge is not scale invariant, so match that scale.
FEATURE_SCALE = 6.0


def get_model_path(model_directory_path: str) -> str:
    return os.path.join(model_directory_path, "model.npz")


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
    d = len(features)

    XtX = np.zeros((d, d), dtype=np.float64)
    Xty = np.zeros(d, dtype=np.float64)
    xsum = np.zeros(d, dtype=np.float64)
    ysum = 0.0
    n = 0

    target = y_train["target"].to_numpy(dtype=np.float32)
    # Column positions, so each moon can be sliced without ever building the full
    # (n_rows x 1150) matrix -- that is ~7 GB and the point of this whole approach.
    col_pos = [X_train.columns.get_loc(c) for c in features]

    # One moon at a time: bounded memory, and it keeps cross-sections intact.
    for _, idx in X_train.groupby("moon").indices.items():
        Xm = X_train.iloc[idx, col_pos].to_numpy(dtype=np.float32) * FEATURE_SCALE
        ym = target[idx]
        XtX += (Xm.T @ Xm).astype(np.float64)
        Xty += (Xm.T @ ym).astype(np.float64)
        xsum += Xm.sum(axis=0, dtype=np.float64)
        ysum += float(ym.sum(dtype=np.float64))
        n += len(Xm)

    # Ridge on centred data; the intercept cannot affect a correlation.
    xbar, ybar = xsum / n, ysum / n
    A = XtX - n * np.outer(xbar, xbar)
    b = Xty - n * xbar * ybar
    A.flat[:: d + 1] += ALPHA_PER_ROW * n
    beta = np.linalg.solve(A, b).astype(np.float32)

    np.savez(get_model_path(model_directory_path), beta=beta,
             features=np.array(features), n_train=n)


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
    # loop_moon: int = None,
    # embargo: int = None,
) -> pd.DataFrame:
    art = np.load(get_model_path(model_directory_path), allow_pickle=False)
    beta = art["beta"]
    features = [str(f) for f in art["features"]]

    X = X_test[features].to_numpy(dtype=np.float32) * FEATURE_SCALE
    raw = X @ beta

    out = X_test[["id", "moon"]].copy()
    pred = np.zeros(len(raw), dtype=np.float32)

    # Centre each moon, then scale to peak magnitude 1. Both steps are affine within
    # a moon, so the Pearson score is untouched, and together they guarantee the
    # [-1, 1] range the platform requires without the clipping that would bend the
    # ranking. Centring also stops the whole moon sitting on one side of zero.
    for _, idx in out.groupby("moon").indices.items():
        block = raw[idx] - raw[idx].mean()
        peak = np.abs(block).max()
        if peak > 0:
            pred[idx] = block / peak
        else:
            # Degenerate moon: a constant column scores 0, so emit a tiny spread
            # rather than a flat vector.
            pred[idx] = np.linspace(-1e-6, 1e-6, len(idx), dtype=np.float32)

    out["prediction"] = pred
    return out
