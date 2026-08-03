"""Model F evaluation -- sign-family feature subspace ridge.

    python -m src.run_subspace

Runs both sign families ("pos", "neg") on the same 3wf+2gap fold budget as
B/D/C, then reports each family's own-correlation with Model A's prediction
on a shared held-out window -- the number that actually matters for the
portfolio decision (JOURNAL.md 2026-07-29: corr(B, D) = 0.835 is the thing
being tested against here, not raw IC).
"""
from __future__ import annotations

import json
import time

import numpy as np

from src.cv import gap_folds, walk_forward
from src.data import REPO, load_cache
from src.explog import record
from src.metrics import per_moon_ic, summarize
from src.models import ridge
from src.models.subspace import ALPHA, fit, predict

BEAT = {"survey_baseline": {"mean_ic": 0.0251, "sharpe": 1.19},
        "model_a_ridge": {"mean_ic": 0.0329, "sharpe": 1.25},
        "model_b_lgbm": {"mean_ic": 0.0429, "sharpe": 1.07},
        "model_d_twostage": {"mean_ic": 0.0479, "sharpe": 1.15}}


def evaluate(cache, folds, family: str, label: str) -> dict:
    all_moons, all_ics, per_fold, n_feat = [], [], {}, []
    for f in folds:
        t0 = time.time()
        feature_idx, beta = fit(cache, f.train, family)
        n_feat.append(len(feature_idx))
        pred, target, moon = predict(cache, f.val, feature_idx, beta)
        m, ic = per_moon_ic(pred, target, moon)
        per_fold[f.name] = float(ic.mean())
        all_moons.append(m)
        all_ics.append(ic)
        print(f"  {label}/{family}/{f.name}: n_features={len(feature_idx)} "
              f"ic={ic.mean():+.4f}  ({time.time() - t0:.0f}s)", flush=True)
    moons, ics = np.concatenate(all_moons), np.concatenate(all_ics)
    order = np.argsort(moons)
    s = summarize(moons[order], ics[order])
    s["by_fold"] = per_fold
    s["n_features_mean"] = float(np.mean(n_feat))
    return s


def correlation_vs_a(cache, fold) -> float:
    """Per-moon correlation between subspace-family predictions and Model A, averaged.

    Same methodology as the A/B/D matrix in JOURNAL.md 2026-07-29: fit each
    model on the fold's train moons, predict on val, correlate per moon.
    """
    out = {}
    A_ALPHA = 1e6  # JOURNAL.md 2026-07-28 exp_ridge: chosen alpha for Model A
    a_out = ridge.fit_expanding(cache, [fold.train], [A_ALPHA])
    a_beta = list(a_out.values())[0]
    a_pred, _, a_moon = ridge.predict(cache, fold.val, a_beta)

    for family in ("pos", "neg"):
        feature_idx, beta = fit(cache, fold.train, family)
        f_pred, _, f_moon = predict(cache, fold.val, feature_idx, beta)
        assert np.array_equal(a_moon, f_moon)
        corrs = []
        for m in np.unique(a_moon):
            mask = a_moon == m
            ap, fp = a_pred[mask], f_pred[mask]
            if ap.std() == 0 or fp.std() == 0:
                continue
            corrs.append(np.corrcoef(ap, fp)[0, 1])
        out[family] = float(np.mean(corrs))
    return out


def main() -> None:
    cache = load_cache()
    wf = walk_forward(cache.moon_values)[-3:]
    gf = gap_folds(cache.moon_values)[-2:]

    print(f"alpha={ALPHA}")
    for family in ("pos", "neg"):
        out = {"family": family, "alpha": ALPHA, "beat": BEAT,
               "walk_forward": evaluate(cache, wf, family, "wf"),
               "gap311": evaluate(cache, gf, family, "gap")}

        (REPO / "reports").mkdir(exist_ok=True)
        (REPO / "reports" / f"exp_subspace_{family}.json").write_text(json.dumps(out, indent=2))

        print(f"\n[{family}] {'family':>12} {'mean_ic':>9} {'ic_std':>8} {'sharpe':>7} "
              f"{'hit':>6} {'last104':>8} {'n_feat':>7}")
        for fam in ("walk_forward", "gap311"):
            s = out[fam]
            print(f"{fam:>12} {s['mean_ic']:>9.4f} {s['ic_std']:>8.4f} {s['sharpe']:>7.2f} "
                  f"{s['hit_rate']:>6.2f} {s['mean_ic_last104']:>8.4f} {s['n_features_mean']:>7.0f}   "
                  + " ".join(f"{v:+.4f}" for v in s["by_fold"].values()))

        record(
            f"exp_subspace_{family}",
            config={"model": "subspace", "family": family, "alpha": ALPHA},
            results={"walk_forward": out["walk_forward"], "gap311": out["gap311"]},
            conclusion="(điền tay sau khi xem số)",
        )

    print("\ncorrelation vs Model A (last walk-forward fold, per-moon avg):")
    corr = correlation_vs_a(cache, wf[-1])
    print(json.dumps(corr, indent=2))
    print(f"\nto beat: {BEAT}")


if __name__ == "__main__":
    main()
