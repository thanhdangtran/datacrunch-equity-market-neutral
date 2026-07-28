"""Scoring. The competition metric is a per-moon Pearson, so everything here is
per-moon first and aggregated second -- never pooled across moons.

See PLAN.md 5.4 for what each reported field is for.
"""
from __future__ import annotations

import numpy as np


def per_moon_ic(pred: np.ndarray, target: np.ndarray, moon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson per moon. Returns (moons, ics); a constant moon scores 0, as on the platform."""
    moons = np.unique(moon)
    ics = np.zeros(len(moons), dtype=np.float64)
    for i, m in enumerate(moons):
        mask = moon == m
        p, t = pred[mask].astype(np.float64), target[mask].astype(np.float64)
        if p.std() == 0 or t.std() == 0:
            continue
        ics[i] = np.corrcoef(p, t)[0, 1]
    return moons, ics


def summarize(moons: np.ndarray, ics: np.ndarray) -> dict:
    """The standard report card. Sharpe is the decision metric, not mean."""
    ics = np.asarray(ics, dtype=np.float64)
    std = ics.std()
    return {
        "n_moons": int(len(ics)),
        "mean_ic": float(ics.mean()),
        "ic_std": float(std),
        "sharpe": float(ics.mean() / std) if std > 0 else 0.0,
        "hit_rate": float((ics > 0).mean()),
        "worst_ic": float(ics.min()),
        "best_ic": float(ics.max()),
        "mean_ic_last104": float(ics[-104:].mean()) if len(ics) >= 104 else float(ics.mean()),
        "moon_first": int(moons[0]),
        "moon_last": int(moons[-1]),
    }


def score(pred: np.ndarray, target: np.ndarray, moon: np.ndarray) -> dict:
    return summarize(*per_moon_ic(pred, target, moon))
