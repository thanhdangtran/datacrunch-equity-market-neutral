"""Data loading for DataCrunch #2.

The raw parquet is 1_637_276 x 1150 float32 = 7.5 GB, but this machine only has
~2.5 GB of RAM free. So we convert once to an int8 memmap (1.9 GB on disk, mapped
not loaded) and everything downstream reads slices out of that.

The 7 feature levels {0, .17, .33, .5, .67, .83, 1.0} map exactly onto {0..6},
so int8 is lossless -- see PLAN.md 3.1.

Build the cache once:
    python -m src.data --build

Then:
    from src.data import load_cache
    d = load_cache()
    X = d.X[d.rows(500)]        # features for moon 500, shape (n_stocks, 1150)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "ws-lovely-fowl" / "data"
CACHE_DIR = REPO / "cache"

N_FEATURES = 1150
# The quantised levels, in order. Feature value -> index into this array.
LEVELS = np.array([0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0], dtype=np.float32)

# Columns read per pass. 1.64M rows x 50 float32 cols ~= 330 MB held by arrow,
# and we convert one column at a time on top of that, so peak stays well under
# the ~2.5 GB this machine has free.
COLUMN_CHUNK = 50


def _feature_names() -> list[str]:
    return [f"Feature_{i}" for i in range(1, N_FEATURES + 1)]


def build_cache(data_dir: Path = DATA_DIR, cache_dir: Path = CACHE_DIR) -> None:
    """Convert X/y parquet into an int8 memmap plus small metadata arrays."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path = data_dir / "X.reduced.parquet"
    y_path = data_dir / "y.reduced.parquet"

    n_rows = pq.ParquetFile(x_path).metadata.num_rows
    print(f"source: {x_path.name}  rows={n_rows:,}  features={N_FEATURES}")

    meta = pq.read_table(x_path, columns=["id", "moon"])
    ids = meta.column("id").to_numpy().astype(np.int32)
    moons = meta.column("moon").to_numpy().astype(np.int32)
    del meta

    # Everything downstream assumes rows are grouped by moon so a moon is one
    # contiguous slice. Verify rather than trust.
    if not np.all(np.diff(moons) >= 0):
        raise RuntimeError("rows are not sorted by moon; slicing assumptions break")

    y_tab = pq.read_table(y_path)
    y_ids = y_tab.column("id").to_numpy().astype(np.int32)
    y_moons = y_tab.column("moon").to_numpy().astype(np.int32)
    target = y_tab.column("target").to_numpy().astype(np.float32)
    del y_tab

    if not (np.array_equal(ids, y_ids) and np.array_equal(moons, y_moons)):
        raise RuntimeError("X and y rows are not aligned; would silently mislabel")

    names = _feature_names()
    X = np.lib.format.open_memmap(
        cache_dir / "X_int8.npy", mode="w+", dtype=np.int8, shape=(n_rows, N_FEATURES)
    )

    t0 = time.time()
    worst = 0.0
    for start in range(0, N_FEATURES, COLUMN_CHUNK):
        cols = names[start : start + COLUMN_CHUNK]
        block = pq.read_table(x_path, columns=cols)

        for k, col in enumerate(cols):
            v = block.column(col).to_numpy()
            # The 7 levels sit on i/6, so scaling by 6 and rounding recovers the
            # index directly -- no (rows x cols x 7) broadcast needed.
            codes = np.rint(v * 6.0).astype(np.int8)
            if codes.min() < 0 or codes.max() > 6:
                raise RuntimeError(f"{col}: code outside 0..6")
            residual = float(np.abs(v - LEVELS[codes]).max())
            if residual > 0.01:
                raise RuntimeError(f"{col}: value off the 7-level grid ({residual})")
            worst = max(worst, residual)
            X[:, start + k] = codes

        del block
        print(f"  cols {start + 1:>4}-{start + len(cols):<4} "
              f"({time.time() - t0:5.1f}s)", flush=True)

    print(f"max deviation from the 7-level grid: {worst:.2e}")

    X.flush()

    # Per-moon row ranges, so callers slice instead of scanning 1.6M rows.
    uniq, first = np.unique(moons, return_index=True)
    bounds = np.append(first, len(moons)).astype(np.int64)

    np.savez(
        cache_dir / "meta.npz",
        id=ids, moon=moons, target=target, moon_values=uniq, moon_bounds=bounds,
    )
    (cache_dir / "features.json").write_text(json.dumps(names))
    print(f"done in {time.time() - t0:.1f}s -> {cache_dir}")


@dataclass
class Cache:
    X: np.ndarray            # (n_rows, 1150) int8 memmap, values 0..6
    id: np.ndarray           # (n_rows,) int32
    moon: np.ndarray         # (n_rows,) int32
    target: np.ndarray       # (n_rows,) float32
    moon_values: np.ndarray  # (n_moons,) sorted unique moons
    moon_bounds: np.ndarray  # (n_moons + 1,) row offsets
    features: list[str]

    def rows(self, moon: int) -> slice:
        """Row slice for one moon."""
        i = int(np.searchsorted(self.moon_values, moon))
        if i >= len(self.moon_values) or self.moon_values[i] != moon:
            raise KeyError(f"moon {moon} not in cache")
        return slice(int(self.moon_bounds[i]), int(self.moon_bounds[i + 1]))

    def rows_between(self, lo: int, hi: int) -> slice:
        """Row slice covering moons in [lo, hi] inclusive."""
        i = int(np.searchsorted(self.moon_values, lo, side="left"))
        j = int(np.searchsorted(self.moon_values, hi, side="right"))
        return slice(int(self.moon_bounds[i]), int(self.moon_bounds[j]))


def load_cache(cache_dir: Path = CACHE_DIR) -> Cache:
    if not (cache_dir / "X_int8.npy").exists():
        raise FileNotFoundError(f"no cache at {cache_dir}; run: python -m src.data --build")
    m = np.load(cache_dir / "meta.npz")
    return Cache(
        X=np.load(cache_dir / "X_int8.npy", mmap_mode="r"),
        id=m["id"], moon=m["moon"], target=m["target"],
        moon_values=m["moon_values"], moon_bounds=m["moon_bounds"],
        features=json.loads((cache_dir / "features.json").read_text()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    if p.parse_args().build:
        build_cache()
    else:
        p.print_help()
