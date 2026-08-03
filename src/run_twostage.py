"""Model D evaluation.

    python -m src.run_twostage

Same fold budget as Model B (3 walk-forward + 2 gap311) and same stride, for a
direct comparison. Bar to clear: survey baseline (0.0251/1.19), ridge (0.0329/1.25),
and ideally something B doesn't already cover -- B's edge already lives in the
extreme movers (JOURNAL.md rank-pred experiment), so D succeeding would mean
splitting classification from regression captures that structure *better*, not
just differently.
"""
from __future__ import annotations

import json
import time

import numpy as np

from src.cv import gap_folds, walk_forward
from src.data import REPO, load_cache
from src.explog import record
from src.metrics import per_moon_ic, summarize
from src.models.twostage import CLF_ESTIMATORS, CLF_PARAMS, REG_ESTIMATORS, REG_PARAMS, fit, predict

STRIDE = 4
BEAT = {"survey_baseline": {"mean_ic": 0.0251, "sharpe": 1.19},
        "model_a_ridge": {"mean_ic": 0.0329, "sharpe": 1.25},
        "model_b_lgbm": {"mean_ic": 0.0429, "sharpe": 1.07}}


def evaluate(cache, folds, label: str) -> dict:
    all_moons, all_ics, per_fold = [], [], {}
    for f in folds:
        t0 = time.time()
        clf, reg = fit(cache, f.train, STRIDE)
        pred, target, moon = predict(cache, f.val, clf, reg)
        m, ic = per_moon_ic(pred, target, moon)
        per_fold[f.name] = float(ic.mean())
        all_moons.append(m)
        all_ics.append(ic)
        print(f"  {label}/{f.name}: ic={ic.mean():+.4f}  ({time.time() - t0:.0f}s)", flush=True)
        del clf, reg
    moons, ics = np.concatenate(all_moons), np.concatenate(all_ics)
    order = np.argsort(moons)
    s = summarize(moons[order], ics[order])
    s["by_fold"] = per_fold
    return s


def main() -> None:
    cache = load_cache()
    wf = walk_forward(cache.moon_values)[-3:]
    gf = gap_folds(cache.moon_values)[-2:]

    print(f"stride={STRIDE}, clf_estimators={CLF_ESTIMATORS}, reg_estimators={REG_ESTIMATORS}")
    out = {"stride": STRIDE, "clf_estimators": CLF_ESTIMATORS, "reg_estimators": REG_ESTIMATORS,
           "beat": BEAT,
           "walk_forward": evaluate(cache, wf, "wf"),
           "gap311": evaluate(cache, gf, "gap")}

    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / "exp_twostage.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'family':>12} {'mean_ic':>9} {'ic_std':>8} {'sharpe':>7} {'hit':>6} {'last104':>8}")
    for fam in ("walk_forward", "gap311"):
        s = out[fam]
        print(f"{fam:>12} {s['mean_ic']:>9.4f} {s['ic_std']:>8.4f} {s['sharpe']:>7.2f} "
              f"{s['hit_rate']:>6.2f} {s['mean_ic_last104']:>8.4f}   "
              + " ".join(f"{v:+.4f}" for v in s["by_fold"].values()))
    print(f"\nto beat: {BEAT}")

    record(
        "exp_twostage",
        config={"model": "twostage", "stride": STRIDE, "clf_estimators": CLF_ESTIMATORS,
                "reg_estimators": REG_ESTIMATORS, "max_bin": CLF_PARAMS["max_bin"]},
        results={"walk_forward": out["walk_forward"], "gap311": out["gap311"]},
        conclusion="(điền tay sau khi xem số)",
    )


if __name__ == "__main__":
    main()
