"""Experiment log.

Every scored run appends one row to reports/leaderboard.csv and one entry to
JOURNAL.md. The CSV is for sorting; the journal is for the part that actually
matters later -- what we concluded and why, which a number on its own never says.

    from src.explog import record
    record("exp_007_lgbm_small", config={...}, results={"walk_forward": {...}},
           conclusion="Sharpe fell; capacity was not the binding constraint.")
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "reports" / "leaderboard.csv"
JOURNAL = REPO / "JOURNAL.md"

FIELDS = ["date", "exp", "family", "mean_ic", "ic_std", "sharpe", "hit_rate",
          "mean_ic_last104", "n_moons", "by_fold", "config"]


def record(exp: str, config: dict, results: dict, conclusion: str,
           notes: str = "", journal: bool = True) -> None:
    """results maps a fold-family name ("walk_forward", "gap311") to a metrics dict.

    journal=False writes the CSV row only -- for backfilling runs whose write-up
    already exists.
    """
    CSV.parent.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()

    new = not CSV.exists()
    with CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for family, m in results.items():
            w.writerow({
                "date": today, "exp": exp, "family": family,
                "mean_ic": round(m.get("mean_ic", float("nan")), 5),
                "ic_std": round(m.get("ic_std", float("nan")), 5),
                "sharpe": round(m.get("sharpe", float("nan")), 3),
                "hit_rate": round(m.get("hit_rate", float("nan")), 3),
                "mean_ic_last104": round(m.get("mean_ic_last104", float("nan")), 5),
                "n_moons": m.get("n_moons", ""),
                "by_fold": json.dumps(m.get("by_fold", {})),
                "config": json.dumps(config, default=str),
            })

    if not journal:
        return

    lines = [f"\n## {today} — `{exp}`\n"]
    lines.append("| family | mean_ic | ic_std | sharpe | hit | last104 |")
    lines.append("|---|---|---|---|---|---|")
    for family, m in results.items():
        lines.append(
            f"| {family} | {m.get('mean_ic', 0):.4f} | {m.get('ic_std', 0):.4f} | "
            f"{m.get('sharpe', 0):.2f} | {m.get('hit_rate', 0):.2f} | "
            f"{m.get('mean_ic_last104', 0):.4f} |"
        )
    lines.append(f"\n**Config:** `{json.dumps(config, default=str)}`\n")
    lines.append(f"**Kết luận:** {conclusion}\n")
    if notes:
        lines.append(f"**Ghi chú:** {notes}\n")

    with JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
