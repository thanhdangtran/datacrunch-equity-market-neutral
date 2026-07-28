"""Survey the real data to check what PLAN.md currently assumes.

Everything here answers a question the plan already takes a position on, so the
output either confirms a section or forces an edit. Writes reports/survey.json.

    python -m src.survey
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.data import REPO, load_cache

REPORTS = REPO / "reports"
EMBARGO = 4


def per_moon_ic(cache, moons: np.ndarray) -> np.ndarray:
    """IC of every feature against the target, per moon. -> (n_moons, 1150)"""
    out = np.full((len(moons), len(cache.features)), np.nan, dtype=np.float32)
    for i, moon in enumerate(moons):
        sl = cache.rows(int(moon))
        y = cache.target[sl].astype(np.float32)
        yc = y - y.mean()
        y_norm = np.linalg.norm(yc)
        if y_norm == 0:
            continue
        x = cache.X[sl].astype(np.float32)
        x -= x.mean(axis=0, keepdims=True)
        x_norm = np.linalg.norm(x, axis=0)
        x_norm[x_norm == 0] = np.inf  # constant feature this moon -> IC 0
        out[i] = (x.T @ yc) / (x_norm * y_norm)
    return out


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    cache = load_cache()
    moons = cache.moon_values
    counts = np.diff(cache.moon_bounds)
    res: dict = {}

    print(f"moons {moons.min()}-{moons.max()} ({len(moons)}), "
          f"rows {len(cache.moon):,}, features {len(cache.features)}")

    # --- universe and target shape over time -------------------------------
    zero_frac = np.array([(cache.target[cache.rows(int(m))] == 0).mean() for m in moons])
    tgt_std = np.array([cache.target[cache.rows(int(m))].std() for m in moons])
    res["universe"] = {
        "stocks_min": int(counts.min()), "stocks_max": int(counts.max()),
        "stocks_mean": float(counts.mean()),
        "stocks_first50_mean": float(counts[:50].mean()),
        "stocks_last50_mean": float(counts[-50:].mean()),
        "zero_frac_overall": float(zero_frac.mean()),
        "zero_frac_first50": float(zero_frac[:50].mean()),
        "zero_frac_last50": float(zero_frac[-50:].mean()),
        "target_std_first50": float(tgt_std[:50].mean()),
        "target_std_last50": float(tgt_std[-50:].mean()),
    }
    # Theoretical noise floor of a single weekly Pearson.
    res["universe"]["ic_noise_std_expected"] = float(np.mean(1 / np.sqrt(counts)))

    # --- per-moon IC of every feature --------------------------------------
    t0 = time.time()
    ic = per_moon_ic(cache, moons)
    print(f"per-moon IC matrix {ic.shape} in {time.time() - t0:.1f}s")

    mean_ic = np.nanmean(ic, axis=0)
    ic_t = mean_ic / (np.nanstd(ic, axis=0) / np.sqrt(len(moons)))
    order = np.argsort(mean_ic)

    res["feature_ic"] = {
        "best_abs_mean_ic": float(np.abs(mean_ic).max()),
        "n_features_absic_gt_0.01": int((np.abs(mean_ic) > 0.01).sum()),
        "n_features_abs_t_gt_3": int((np.abs(ic_t) > 3).sum()),
        "top_negative": [
            {"feature": cache.features[i], "mean_ic": float(mean_ic[i]), "t": float(ic_t[i])}
            for i in order[:10]
        ],
        "top_positive": [
            {"feature": cache.features[i], "mean_ic": float(mean_ic[i]), "t": float(ic_t[i])}
            for i in order[::-1][:10]
        ],
    }

    # --- does the signal survive across eras? ------------------------------
    # This is the 2.7 question: we fit on old moons and are scored 311 moons later.
    half = len(moons) // 2
    early, late = np.nanmean(ic[:half], axis=0), np.nanmean(ic[half:], axis=0)
    q1, q4 = np.nanmean(ic[: len(moons) // 4], axis=0), np.nanmean(ic[-len(moons) // 4 :], axis=0)
    res["stability"] = {
        "corr_early_late_featureIC": float(np.corrcoef(early, late)[0, 1]),
        "corr_q1_q4_featureIC": float(np.corrcoef(q1, q4)[0, 1]),
        "top20_early_mean_ic_in_late": float(late[np.argsort(np.abs(early))[::-1][:20]].mean()),
        "top20_early_mean_ic_in_early": float(early[np.argsort(np.abs(early))[::-1][:20]].mean()),
    }

    # --- how many independent features are there really? -------------------
    rng = np.random.default_rng(0)
    sample_moons = rng.choice(moons, size=40, replace=False)
    blocks = [cache.X[cache.rows(int(m))].astype(np.float32) for m in sample_moons]
    Z = np.vstack(blocks)
    del blocks
    Z -= Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = np.inf
    Z /= sd
    corr = (Z.T @ Z) / len(Z)
    del Z
    ev = np.linalg.eigvalsh(corr)[::-1].clip(min=0)
    cum = np.cumsum(ev) / ev.sum()
    off = np.abs(corr[np.triu_indices_from(corr, k=1)])
    res["redundancy"] = {
        "sample_rows": int(sum(counts[np.searchsorted(moons, sample_moons)])),
        "pc_for_90pct_var": int(np.searchsorted(cum, 0.90) + 1),
        "pc_for_95pct_var": int(np.searchsorted(cum, 0.95) + 1),
        "pc_for_99pct_var": int(np.searchsorted(cum, 0.99) + 1),
        "top1_pc_var": float(cum[0]),
        "median_abs_offdiag_corr": float(np.median(off)),
        "frac_pairs_absr_gt_0.95": float((off > 0.95).mean()),
    }

    # --- cheapest possible baselines ---------------------------------------
    # Equal-weight the features with the strongest |mean IC|, sign-aligned.
    def baseline_ic(idx: np.ndarray, signs: np.ndarray) -> dict:
        vals = []
        for m in moons:
            sl = cache.rows(int(m))
            y = cache.target[sl].astype(np.float32)
            p = (cache.X[sl][:, idx].astype(np.float32) * signs).mean(axis=1)
            if p.std() == 0 or y.std() == 0:
                vals.append(0.0)
                continue
            vals.append(float(np.corrcoef(p, y)[0, 1]))
        v = np.array(vals)
        return {"mean_ic": float(v.mean()), "ic_std": float(v.std()),
                "sharpe": float(v.mean() / v.std()), "hit_rate": float((v > 0).mean()),
                "mean_ic_last104": float(v[-104:].mean())}

    best1 = int(np.argmax(np.abs(mean_ic)))
    res["baselines"] = {
        "single_best_feature": {
            "feature": cache.features[best1], "sign": int(np.sign(mean_ic[best1])),
            **baseline_ic(np.array([best1]), np.array([np.sign(mean_ic[best1])], np.float32)),
        },
    }
    for k in (20, 100):
        idx = np.argsort(np.abs(mean_ic))[::-1][:k]
        res["baselines"][f"top{k}_signed_mean"] = baseline_ic(
            idx, np.sign(mean_ic[idx]).astype(np.float32)
        )

    (REPORTS / "survey.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
