#!/usr/bin/env python3
"""
Build a sample series from TfL (Transport for London) journey-style open data.

Official open data portal: https://data.tfl.gov.uk/ (many datasets; URLs change).
This script expects a **local** CSV you downloaded (e.g. Oyster / journey extracts) or a direct URL
if you pass --url.

Typical workflow:
  1. Download a journey or taps CSV from TfL open data (large).
  2. Run: python download_tfl_journey_sample.py --path /path/to/tfl.csv --column journey_time --rows 80000

Output: sampler/datasets/tfl_journey_sample.csv with column `value`.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="TfL journey CSV → sample for time series")
    p.add_argument("--path", default="", help="Local CSV path (if not using --url)")
    p.add_argument("--url", default="", help="Optional HTTP URL to CSV (if small enough)")
    p.add_argument("--rows", type=int, default=50_000)
    p.add_argument(
        "--column",
        required=True,
        help="Numeric column name (e.g. journey duration, distance)",
    )
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "datasets" / "tfl_journey_sample.csv"),
    )
    args = p.parse_args()

    import pandas as pd

    if args.url:
        df = pd.read_csv(args.url, nrows=max(args.rows * 2, 100_000))
    elif args.path:
        df = pd.read_csv(args.path, nrows=max(args.rows * 2, 500_000))
    else:
        raise SystemExit("Provide --path to a local TfL CSV or --url")

    if args.column not in df.columns:
        raise SystemExit(f"Column {args.column!r} not found. Columns: {list(df.columns)[:40]}")
    s = pd.to_numeric(df[args.column], errors="coerce").dropna().iloc[: args.rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"value": s.values}).to_csv(out, index=False)
    print(f"Wrote {len(s)} rows to {out}")


if __name__ == "__main__":
    main()
