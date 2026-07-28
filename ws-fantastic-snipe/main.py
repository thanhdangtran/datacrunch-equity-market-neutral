#%pip install crunch-cli --upgrade --quiet --progress-bar off
#!crunch setup-notebook datacrunch-2 YOUR_TOKEN_HERE --size small


import os
import warnings
#warnings.filterwarnings("ignore", message="X does not have valid feature names")
import json
import joblib  # == 1.3.2
import numpy as np  # == 1.24.3
import pandas as pd  # == 2.1.0

import sklearn  # == 1.1.3
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import lightgbm as lgb  # == 4.3.0


import crunch
#crunch_tools = crunch.load_notebook()
#X_train, y_train, X_test = crunch_tools.load_data()


# @crunch/keep:on
def get_config() -> dict:
    SEED = 42
    return {
        "SEED": SEED,
        # blend
        "BLEND_WEIGHT_LINEAR": 0.5,      # rank weight on linear; GBM gets the rest
        # latent factors (the "C" upgrade)
        "N_PCA": 40,                     # number of principal components to add as features
        # per-moon processing
        "RANK_TARGET_PER_MOON": True,    # train on per-moon gauss-ranked target
        # neutralization (per moon, on the blended score)
        "NEUTRALIZATION_ALPHA": 0.5,     # 0=off, 1=full; 0.3-0.5 is the Numerai sweet spot
        "NEUTRALIZATION_RIDGE": 1e-2,
        # linear
        "RIDGE_ALPHA": 100.0,
        # conservative LightGBM
        "LGB_PARAMS": dict(
            objective="regression",
            n_estimators=2000, learning_rate=0.01,
            num_leaves=31, max_depth=5,
            feature_fraction=0.10, bagging_fraction=0.70, bagging_freq=1,
            min_child_samples=200, lambda_l2=10.0, max_bin=15,
            seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED,
            deterministic=True, force_col_wise=True, n_jobs=-1, verbose=-1,
        ),
    }
# @crunch/keep:off


def get_model_path(model_directory_path: str) -> str:
    return os.path.join(model_directory_path, "model.joblib")

def get_feature_columns(X: pd.DataFrame) -> list:
    return [c for c in X.columns if c.startswith("Feature_")]

def gauss_rank(a: np.ndarray) -> np.ndarray:
    """Map values to a centered Gaussian-ish rank in (-1, 1). Outlier-robust, helps Pearson."""
    order = np.argsort(np.argsort(a))
    u = (order + 0.5) / len(a)            # uniform ranks in (0,1)
    return (u - 0.5) * 2.0                 # to (-1, 1)

def per_moon_gauss_rank_target(df, col, moon_col="moon"):
    out = np.empty(len(df), dtype=np.float32)
    for _, idx in df.groupby(moon_col).indices.items():
        out[idx] = gauss_rank(df[col].to_numpy()[idx])
    return out

def neutralize_per_moon(scores, feats, moons, alpha, ridge_lambda):
    """Residualize scores against features within each moon (partial, proportion=alpha)."""
    out = scores.astype(np.float32).copy()
    for m in np.unique(moons):
        mask = moons == m
        F = feats[mask]
        s = scores[mask]
        # ridge-stabilized projection of s onto F
        FtF = F.T @ F + ridge_lambda * np.eye(F.shape[1], dtype=F.dtype)
        beta = np.linalg.solve(FtF, F.T @ s)
        proj = F @ beta
        out[mask] = s - alpha * proj
    return out


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
    # loop_moon: int = None,
    # embargo: int = None,
) -> None:
    cfg = get_config()
    np.random.seed(cfg["SEED"])

    feats = get_feature_columns(X_train)
    Xf = X_train[feats].to_numpy(dtype=np.float32)

    # --- Latent factors (C): fit PCA, append components as extra features ---
    pca = PCA(n_components=cfg["N_PCA"], random_state=cfg["SEED"])
    Xpca = pca.fit_transform(Xf).astype(np.float32)
    X_aug = np.hstack([Xf, Xpca])

    # --- Target: per-moon gauss rank aligns with the rank-correlation objective ---
    if cfg["RANK_TARGET_PER_MOON"]:
        y = per_moon_gauss_rank_target(y_train.assign(moon=X_train["moon"].values),
                                       "target", "moon")
    else:
        y = y_train["target"].to_numpy(dtype=np.float32)

    # --- Two diverse base learners ---
    linear = Ridge(alpha=cfg["RIDGE_ALPHA"], random_state=cfg["SEED"])
    linear.fit(X_aug, y)

    gbm = lgb.LGBMRegressor(**cfg["LGB_PARAMS"])
    gbm.fit(X_aug, y)

    joblib.dump({
        "pca": pca,
        "linear": linear,
        "gbm": gbm,
        "feats": feats,
        "config": {
            "BLEND_WEIGHT_LINEAR": cfg["BLEND_WEIGHT_LINEAR"],
            "NEUTRALIZATION_ALPHA": cfg["NEUTRALIZATION_ALPHA"],
            "NEUTRALIZATION_RIDGE": cfg["NEUTRALIZATION_RIDGE"],
        },
    }, get_model_path(model_directory_path))


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
    # loop_moon: int = None,
    # embargo: int = None,
) -> pd.DataFrame:
    art = joblib.load(get_model_path(model_directory_path))
    pca, linear, gbm = art["pca"], art["linear"], art["gbm"]
    feats, cfg = art["feats"], art["config"]

    Xf = X_test[feats].to_numpy(dtype=np.float32)
    X_aug = np.hstack([Xf, pca.transform(Xf).astype(np.float32)])

    moons = X_test["moon"].to_numpy()

    # base predictions -> per-moon ranks -> weighted blend
    def per_moon_rank(p):
        out = np.empty_like(p, dtype=np.float32)
        for m in np.unique(moons):
            mask = moons == m
            out[mask] = gauss_rank(p[mask])
        return out

    p_lin = per_moon_rank(linear.predict(X_aug))
    p_gbm = per_moon_rank(gbm.predict(X_aug))
    w = cfg["BLEND_WEIGHT_LINEAR"]
    blended = w * p_lin + (1.0 - w) * p_gbm

    # per-moon feature neutralization for OOS stability
    blended = neutralize_per_moon(
        blended, Xf, moons,
        alpha=cfg["NEUTRALIZATION_ALPHA"],
        ridge_lambda=cfg["NEUTRALIZATION_RIDGE"],
    )

    # final per-moon gauss rank + clip to [-1, 1] (never constant -> avoids score 0)
    final = per_moon_rank(blended)

    out = X_test[["id", "moon"]].copy()
    out["prediction"] = np.clip(final, -1.0, 1.0)
    return out


#crunch_tools.test(
#    force_first_train=True,
#    train_frequency=0,
#)


#prediction = pd.read_parquet("prediction/prediction.parquet")
#y_test = pd.read_parquet(
#    "data/y.reduced.parquet",
#    filters=[("moon", "in", prediction["moon"].unique().tolist())],
#)
#merged = y_test.merge(prediction, on=["moon", "id"])

#corr_by_moon = (merged.groupby("moon")
#    .apply(lambda g: g["prediction"].corr(g["target"], method="pearson"), include_groups=False)
#    .fillna(0.0))

#corr_mean = float(corr_by_moon.mean())
#corr_std  = float(corr_by_moon.std())
#sharpe    = corr_mean / corr_std if corr_std > 0 else 0.0

# EDA snapshot
#t = y_train["target"].to_numpy()
#hist, _ = np.histogram(t, bins=15)
#cfg = get_config()

#results = {
#    "metric": "pearson",
#    "corr_mean": corr_mean,
#    "corr_std": corr_std,
#    "sharpe": float(sharpe),
#    "corr_by_moon": [float(x) for x in corr_by_moon.tolist()],
#    "moons_scored": [int(x) for x in corr_by_moon.index.tolist()],
#    "eda": {
#        "n_moons": int(X_train["moon"].nunique()),
#        "n_rows": int(len(X_train)),
#        "n_feats": int(len(get_feature_columns(X_train))),
#        "avg_stocks": int(round(len(X_train) / max(1, X_train["moon"].nunique()))),
#        "pct_zero": float((t == 0).mean() * 100),
#        "target_hist": [int(x) for x in hist.tolist()],
#    },
#    "config": {k: (v if not isinstance(v, dict) else "…") for k, v in cfg.items() if k != "LGB_PARAMS"},
#}

#with open("results.json", "w") as f:
#    json.dump(results, f, indent=2)

#print(f"Pearson mean: {corr_mean:+.4f}  | std {corr_std:.4f} | Sharpe {sharpe:.2f}")
#print("Wrote results.json — open dashboard.html to view the full report.")


#crunch_tools.submit(
#    message="rank-blend + PCA latent factors",
#    include_installed_packages_version=True,
#)
