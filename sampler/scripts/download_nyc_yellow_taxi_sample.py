#!/usr/bin/env python3
"""
Download a sample from NYC TLC Yellow Taxi open data (Parquet) and save a single numeric column as CSV.

Source (public): NYC Taxi & Limousine Commission trip records (yellow taxi), monthly Parquet files.
Example URL pattern (update month/year as needed):
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet

Requires: pandas, pyarrow (for read_parquet).

Usage:
  python download_nyc_yellow_taxi_sample.py --url <parquet_url> --rows 100000 --column trip_distance \\
      --out ../datasets/nyc_yellow_taxi_trip_sample.csv

The output CSV has columns: index, value (for use as a univariate series in the Shiny app).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="NYC Yellow Taxi → sample CSV")
    p.add_argument(
        "--url",
        default="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
        help="HTTPS URL to a yellow Parquet file",
    )
    p.add_argument("--rows", type=int, default=50_000, help="Max rows to keep")
    p.add_argument(
        "--column",
        default="trip_distance",
        help="Numeric column to export as 'value' (e.g. trip_distance, total_amount)",
    )
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "datasets" / "nyc_yellow_taxi_trip_sample.csv"),
        help="Output CSV path",
    )
    args = p.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.url)
    if args.column not in df.columns:
        raise SystemExit(f"Column {args.column!r} not in dataset. Columns: {list(df.columns)[:30]}...")
    s = pd.to_numeric(df[args.column], errors="coerce").dropna().iloc[: args.rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"value": s.values}).to_csv(out, index=False)
    print(f"Wrote {len(s)} rows to {out}")


if __name__ == "__main__":
    main()
