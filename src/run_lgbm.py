"""Model B evaluation.

    python -m src.run_lgbm

Fewer folds than Model A on purpose: LightGBM refits from scratch per fold while
ridge shares one streamed Gram, so the fold budget has to be spent carefully.
Bar to clear is PLAN.md 2.8a (0.0251 / 1.19) and now Model A (0.0329 / 1.25).
"""
from __future__ import annotations

import json
import time

import numpy as np

import argparse

from src.cv import gap_folds, walk_forward
from src.data import REPO, load_cache
from src.explog import record
from src.metrics import per_moon_ic, summarize
from src.models.lgbm import BASE_PARAMS, fit, predict

STRIDE = 4          # keep every 4th moon -> ~400k rows, fits in the RAM we have
N_ESTIMATORS = 800
BEAT = {"survey_baseline": {"mean_ic": 0.0251, "sharpe": 1.19},
        "model_a_ridge": {"mean_ic": 0.0329, "sharpe": 1.25}}


def evaluate(cache, folds, label: str, params: dict, n_estimators: int) -> dict:
    all_moons, all_ics, per_fold = [], [], {}
    for f in folds:
        t0 = time.time()
        booster = fit(cache, f.train, STRIDE, params, n_estimators)
        pred, target, moon = predict(cache, f.val, booster)
        m, ic = per_moon_ic(pred, target, moon)
        per_fold[f.name] = float(ic.mean())
        all_moons.append(m)
        all_ics.append(ic)
        print(f"  {label}/{f.name}: ic={ic.mean():+.4f}  ({time.time() - t0:.0f}s)", flush=True)
        del booster
    moons, ics = np.concatenate(all_moons), np.concatenate(all_ics)
    order = np.argsort(moons)
    s = summarize(moons[order], ics[order])
    s["by_fold"] = per_fold
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp_lgbm", help="name used in the journal")
    ap.add_argument("--num-leaves", type=int, default=BASE_PARAMS["num_leaves"])
    ap.add_argument("--lambda-l2", type=float, default=BASE_PARAMS["lambda_l2"])
    ap.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    ap.add_argument("--conclusion", default="(chưa ghi)")
    args = ap.parse_args()

    params = dict(BASE_PARAMS, num_leaves=args.num_leaves, lambda_l2=args.lambda_l2)
    globals()["BASE_PARAMS"] = params

    cache = load_cache()
    wf = walk_forward(cache.moon_values)[-3:]      # 3 most recent walk-forward folds
    gf = gap_folds(cache.moon_values)[-2:]         # 2 most recent gap folds

    print(f"stride={STRIDE}, n_estimators={args.n_estimators}, "
          f"num_leaves={args.num_leaves}, lambda_l2={args.lambda_l2}")
    out = {"params": params, "stride": STRIDE, "n_estimators": args.n_estimators,
           "beat": BEAT,
           "walk_forward": evaluate(cache, wf, "wf", params, args.n_estimators),
           "gap311": evaluate(cache, gf, "gap", params, args.n_estimators)}

    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / f"{args.exp}.json").write_text(json.dumps(out, indent=2))
    record(
        args.exp,
        config={"model": "lgbm", "stride": STRIDE, "n_estimators": args.n_estimators,
                "num_leaves": args.num_leaves, "lambda_l2": args.lambda_l2,
                "max_bin": params["max_bin"], "feature_fraction": params["feature_fraction"]},
        results={"walk_forward": out["walk_forward"], "gap311": out["gap311"]},
        conclusion=args.conclusion,
    )

    print(f"\n{'family':>12} {'mean_ic':>9} {'ic_std':>8} {'sharpe':>7} {'hit':>6} {'last104':>8}")
    for fam in ("walk_forward", "gap311"):
        s = out[fam]
        print(f"{fam:>12} {s['mean_ic']:>9.4f} {s['ic_std']:>8.4f} {s['sharpe']:>7.2f} "
              f"{s['hit_rate']:>6.2f} {s['mean_ic_last104']:>8.4f}   "
              + " ".join(f"{v:+.4f}" for v in s["by_fold"].values()))
    print(f"\nto beat: survey {BEAT['survey_baseline']}, ridge {BEAT['model_a_ridge']}")


if __name__ == "__main__":
    main()
