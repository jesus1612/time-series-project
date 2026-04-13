# Sampler — datasets for ARIMA benchmarks

CSV time series used to evaluate linear vs parallel ARIMA (app benchmark tab and scripts).

This folder lives in the unified repository [time-series-project](https://github.com/jesus1612/time-series-project); paths below are relative to that repo root (your clone directory may have another name).

## Layout

- `datasets/` — generated CSV files (run `generate_datasets.py`).
- `generate_datasets.py` — builds or exports all series.
- `ANALISIS_CRUCE.md` — template / notes for speedup vs *N* experiments (fill after runs).

## Files

| File | Description |
|------|-------------|
| `airline_passengers.csv` | Classic monthly airline passengers (statsmodels `AirPassengers` if available). |
| `sunspot_yearly.csv` | Annual sunspots (`sunspots` dataset). |
| `daily_temperatures.csv` | Synthetic smooth seasonal + noise (~3650 rows). |
| `sp500_daily.csv` | Synthetic log-random-walk style “finance” series (~2500 rows). |
| `synthetic_arima_211.csv` | Simulated ARIMA(2,1,1)-like integrated AR noise (known structure). |
| `large_synthetic_10k.csv` | Long stationary-ish series for timing. |
| `large_synthetic_50k.csv` | Same, 50k rows. |
| `arima_eval_benchmark.csv` | **Recommended for ARIMA quality checks**: ~960 daily points, no NaN; integrated ARMA(1,1) on differences (informal ARIMA(1,1,1)-style DGP with φ≈0.65, θ≈0.40, σ=1, level 100). Good for RMSE/MAPE on hold-out and comparing linear vs Spark workflow. |

## Usage

From the repository root:

```bash
python sampler/generate_datasets.py
```

Requires: `numpy`, `pandas`; optional `statsmodels` for classic datasets.

## Policy

All sampler series are **complete** (no NaN), aligned with the app rule: missing values are not imputed.
