"""Purged walk-forward splits over moons.

Two split families, both in PLAN.md:

  walk_forward()  -- 5.3: train [.., T], skip `embargo` moons, validate the next span.
                     Measures "does this work next quarter".

  gap_folds()     -- 2.7: train [.., T], skip ~300 moons, validate far in the future.
                     Measures "does this still work 6 years later", which is the
                     actual local->live situation: local data ends at moon 781 and
                     the scored moon is 1092.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMBARGO = 4
LIVE_GAP = 311  # 1092 - 781


@dataclass(frozen=True)
class Fold:
    name: str
    train: np.ndarray  # moons used for fitting
    val: np.ndarray    # moons scored

    def __repr__(self) -> str:
        return (f"Fold({self.name}: train {self.train[0]}-{self.train[-1]} "
                f"[{len(self.train)}], val {self.val[0]}-{self.val[-1]} [{len(self.val)}])")


def _check(fold: Fold) -> Fold:
    if np.intersect1d(fold.train, fold.val).size:
        raise ValueError(f"{fold.name}: train and val overlap")
    if fold.val[0] - fold.train[-1] <= EMBARGO:
        raise ValueError(f"{fold.name}: gap {fold.val[0] - fold.train[-1]} <= embargo {EMBARGO}")
    return fold


def walk_forward(moons: np.ndarray, n_folds: int = 6, val_span: int = 52,
                 min_train: int = 260, embargo: int = EMBARGO) -> list[Fold]:
    """Expanding-window folds ending at the most recent moon."""
    moons = np.sort(moons)
    folds: list[Fold] = []
    end = len(moons)
    for k in range(n_folds):
        val_hi = end - k * val_span
        val_lo = val_hi - val_span
        train_hi = val_lo - embargo
        if val_lo <= 0 or train_hi < min_train:
            break
        folds.append(_check(Fold(
            name=f"wf{n_folds - k}",
            train=moons[:train_hi],
            val=moons[val_lo:val_hi],
        )))
    return folds[::-1]


def gap_folds(moons: np.ndarray, gap: int = LIVE_GAP, val_span: int = 52,
              n_folds: int = 3, min_train: int = 200) -> list[Fold]:
    """Train, then skip `gap` moons before validating -- the PLAN.md 2.7 stress test."""
    moons = np.sort(moons)
    folds: list[Fold] = []
    end = len(moons)
    for k in range(n_folds):
        val_hi = end - k * val_span
        val_lo = val_hi - val_span
        train_hi = val_lo - gap
        if val_lo <= 0 or train_hi < min_train:
            break
        folds.append(_check(Fold(
            name=f"gap{gap}_{n_folds - k}",
            train=moons[:train_hi],
            val=moons[val_lo:val_hi],
        )))
    return folds[::-1]
