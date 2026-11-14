#!/usr/bin/env python3
"""Per-episode bowl positions -> <out>/bowls_<dataset>.parquet (one row per episode and bowl).

  PYTHONPATH=src python scripts/perception_bowls.py 431a71f7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hackjunction2026.perception.bowls import detect_bowls
from hackjunction2026.perception.dataset import Dataset
from hackjunction2026.perception.scene import Frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default="/home/intel/dev/captures/perception/A")
    args = ap.parse_args()
    ds = Dataset(args.dataset)
    out = Path(args.out)
    fr = Frame.load(out / f"table_frame_{ds.id}.json")
    rows = []
    for e in ds.episodes:
        first = {cam: ds.frame(e.index, cam, 0) for cam in ("top", "floor")}
        rows += detect_bowls(first, fr, e.index)
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", ds.id)
    df.to_parquet(out / f"bowls_{ds.id}.parquet", index=False)
    print(df.groupby("color")[["x_mm", "y_mm", "radius_mm"]].describe().round(0).T.to_string())
    print(len(df), "rows ->", out / f"bowls_{ds.id}.parquet")


if __name__ == "__main__":
    main()
