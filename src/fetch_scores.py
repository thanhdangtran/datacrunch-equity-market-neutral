"""Read run status and scores from the hub.

    CRUNCH_API_KEY=...  python -m src.fetch_scores

The token inside each workspace (`.crunchdao/token`) is push-scoped: it resolves the
project but every read endpoint answers "Access Denied". Reading results needs a real
account API key, which the hub issues under account settings.

Without the key this still prints the competition metric definition, which is public.
"""
from __future__ import annotations

import json
import os

from crunch.api import Client

COMPETITION = "datacrunch-2"
USER_ID = 13215
PROJECTS = ["secure-ladybug", "lovely-fowl", "fantastic-snipe", "scornful-trout"]


def _attrs(obj) -> dict:
    return getattr(obj, "_attrs", {}) or {}


def main() -> None:
    key = os.getenv("CRUNCH_API_KEY")
    client = Client.from_env(show_progress=False)
    competition = client.competitions.get(COMPETITION)

    print("=== metrics (public) ===")
    for metric in competition.metrics.list():
        a = _attrs(metric)
        print(f"  {a.get('name')} | primary={a.get('primary')} "
              f"| rankOrder={a.get('rankOrder')} | weight={a.get('weight')} "
              f"| scorer={a.get('scorerFunction')} | reducer={a.get('reducerFunction')}")

    if not key:
        print("\nCRUNCH_API_KEY not set -- cannot read runs or scores.")
        print("Create one on the hub (account settings), then re-run with it exported.")
        return

    for name in PROJECTS:
        print(f"\n=== {name} ===")
        try:
            project = competition.projects.get_reference(None, (USER_ID, name))
        except Exception as exc:  # project may not exist yet
            print(f"  project lookup failed: {type(exc).__name__}: {exc}")
            continue

        try:
            runs = list(project.runs.list())
            failed = [r for r in runs if not r.success]
            print(f"  runs: {len(runs)} total, {len(failed)} failed")
            for run in failed:
                a = _attrs(run)
                print(f"    FAILED run {a.get('id')} ({str(a.get('createdAt'))[:10]}): "
                      f"{a.get('error')}")
        except Exception as exc:
            print(f"  runs: {type(exc).__name__}: {exc}")

        # The score lives on the prediction, not the run. `primaryMean` is the
        # competition metric; None means the target has not resolved yet (the
        # documented 6-week lag).
        try:
            predictions = list(project.predictions.list())
        except Exception as exc:
            print(f"  predictions: {type(exc).__name__}: {exc}")
            continue

        print(f"  {'pred':>7} {'created':>10} {'score':>10}  name")
        for prediction in predictions:
            a = _attrs(prediction)
            mean = a.get("primaryMean")
            shown = f"{mean:+.5f}" if isinstance(mean, (int, float)) else "pending"
            print(f"  {a.get('id'):>7} {str(a.get('createdAt'))[:10]:>10} {shown:>10}  "
                  f"{str(a.get('name'))[:60]}")


if __name__ == "__main__":
    main()
