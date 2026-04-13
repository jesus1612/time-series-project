"""
ARIMA benchmark suite: TSLib linear, ParallelARIMAWorkflow, Spark+statsmodels, and
statsmodels local (in-process reference for evaluation only — not a production alternative).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tslib.benchmarks.arima_evaluation import (
    statsmodels_arima_fit_only,
    statsmodels_arima_forecast,
)
from tslib.metrics.evaluation import ForecastMetrics
from tslib.models.arima_model import ARIMAModel

DEFAULT_N_GRID = [100, 500, 1000, 2000, 5000, 10000]


def default_sampler_datasets_dir() -> Path:
    """TT/sampler/datasets relative to this package root (tslib-shiny-app)."""
    return Path(__file__).resolve().parents[3] / "sampler" / "datasets"


def _time_call(fn: Callable[[], Any], repeats: int = 2) -> float:
    times: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def _linear_fit_predict(train: np.ndarray, order: Tuple[int, int, int], horizon: int) -> np.ndarray:
    m = ARIMAModel(order=order, auto_select=False, validation=False, n_jobs=1)
    m.fit(train)
    out = m.predict(steps=horizon, return_conf_int=False)
    return np.asarray(out, dtype=float).ravel()


def _workflow_fit_predict(train: np.ndarray, horizon: int, verbose: bool = False) -> np.ndarray:
    from tslib.spark import ParallelARIMAWorkflow

    w = ParallelARIMAWorkflow(verbose=verbose)
    w.fit(train)
    pred = w.predict(steps=horizon, return_conf_int=False)
    return np.asarray(pred, dtype=float).ravel()


def _spark_statsmodels_forecast(
    spark: Any,
    train: np.ndarray,
    order: Tuple[int, int, int],
    horizon: int,
) -> np.ndarray:
    from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

    y = np.asarray(train, dtype=float)
    p, d, q = order
    schema = StructType(
        [
            StructField("step", IntegerType(), False),
            StructField("forecast", DoubleType(), False),
        ]
    )

    def run_batches(iterator):
        import pandas as pd
        from statsmodels.tsa.arima.model import ARIMA

        rows: List[Tuple[int, float]] = []
        for _ in iterator:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(y, order=(p, d, q)).fit()
                    fc = np.asarray(res.forecast(horizon), dtype=float).ravel()
                for i in range(min(horizon, len(fc))):
                    rows.append((i, float(fc[i])))
            except Exception:
                pass
        if not rows:
            yield pd.DataFrame(columns=["step", "forecast"])
        else:
            yield pd.DataFrame(rows, columns=["step", "forecast"])

    df = spark.range(1).coalesce(1).mapInPandas(run_batches, schema)
    collected = df.orderBy("step").collect()
    return [float(r.forecast) for r in collected]


class ARIMABenchmarkSuite:
    """Run performance and accuracy comparisons for the three ARIMA backends."""

    def __init__(self, sampler_dir: Optional[Path] = None):
        self.sampler_dir = Path(sampler_dir) if sampler_dir else default_sampler_datasets_dir()

    def run_performance_benchmark(
        self,
        n_obs_grid: Optional[List[int]] = None,
        repeats: int = 2,
        order: Tuple[int, int, int] = (1, 1, 1),
    ) -> Dict[str, Any]:
        """
        Wall-clock fit time only (forecast optional small). Returns times per method and N.
        """
        n_obs_grid = n_obs_grid or DEFAULT_N_GRID
        rng = np.random.default_rng(42)

        def synth(n: int) -> np.ndarray:
            eps = rng.standard_normal(n)
            y = np.zeros(n)
            for t in range(1, n):
                y[t] = 0.8 * y[t - 1] + eps[t]
            return y

        linear_times: Dict[int, float] = {}
        workflow_times: Dict[int, float] = {}
        spark_sm_times: Dict[int, float] = {}
        sm_local_times: Dict[int, float] = {}
        workflow_ok: Dict[int, bool] = {}
        spark_sm_ok: Dict[int, bool] = {}
        sm_local_ok: Dict[int, bool] = {}

        spark = None
        try:
            from tslib.spark.ensure import ensure_spark_session

            spark = ensure_spark_session(app_name="tslib_arima_bench", register_global=False)
        except Exception:
            pass

        for n in n_obs_grid:
            data = synth(n)

            linear_times[n] = _time_call(
                lambda: ARIMAModel(
                    order=order, auto_select=False, validation=False, n_jobs=1
                ).fit(data),
                repeats=repeats,
            )

            try:

                def _sm_loc():
                    statsmodels_arima_fit_only(data, order)

                sm_local_times[n] = _time_call(_sm_loc, repeats=repeats)
                sm_local_ok[n] = True
            except Exception:
                sm_local_times[n] = float("nan")
                sm_local_ok[n] = False

            if spark is not None:
                try:
                    from tslib.spark import ParallelARIMAWorkflow

                    def _wf():
                        w = ParallelARIMAWorkflow(verbose=False)
                        w.fit(data)

                    workflow_times[n] = _time_call(_wf, repeats=max(1, repeats - 1))
                    workflow_ok[n] = True
                except Exception:
                    workflow_times[n] = float("nan")
                    workflow_ok[n] = False

                try:

                    def _sm():
                        _spark_statsmodels_forecast(spark, data, order, min(5, max(1, n // 20)))

                    spark_sm_times[n] = _time_call(_sm, repeats=repeats)
                    spark_sm_ok[n] = True
                except Exception:
                    spark_sm_times[n] = float("nan")
                    spark_sm_ok[n] = False
            else:
                workflow_times[n] = float("nan")
                spark_sm_times[n] = float("nan")
                workflow_ok[n] = False
                spark_sm_ok[n] = False

        speedup_sm_local = {
            n: (linear_times[n] / sm_local_times[n])
            for n in n_obs_grid
            if sm_local_ok.get(n) and sm_local_times[n] > 0
        }

        speedup_w = {
            n: (linear_times[n] / workflow_times[n])
            for n in n_obs_grid
            if workflow_ok.get(n) and workflow_times[n] > 0
        }
        speedup_sm = {
            n: (linear_times[n] / spark_sm_times[n])
            for n in n_obs_grid
            if spark_sm_ok.get(n) and spark_sm_times[n] > 0
        }

        return {
            "n_obs_grid": n_obs_grid,
            "order": order,
            "linear_times": linear_times,
            "workflow_times": workflow_times,
            "spark_statsmodels_times": spark_sm_times,
            "statsmodels_local_times": sm_local_times,
            "workflow_ok": workflow_ok,
            "spark_statsmodels_ok": spark_sm_ok,
            "statsmodels_local_ok": sm_local_ok,
            "speedup_linear_vs_workflow": speedup_w,
            "speedup_linear_vs_spark_sm": speedup_sm,
            "speedup_linear_vs_statsmodels_local": speedup_sm_local,
            "crossover_workflow": self.find_crossover_point(speedup_w, threshold=1.0),
            "crossover_spark_sm": self.find_crossover_point(speedup_sm, threshold=1.0),
            "crossover_statsmodels_local": self.find_crossover_point(
                speedup_sm_local, threshold=1.0
            ),
        }

    @staticmethod
    def find_crossover_point(
        speedups_by_n: Dict[int, float],
        threshold: float = 1.0,
    ) -> Optional[int]:
        """Smallest N where speedup >= threshold; None if never."""
        best: Optional[int] = None
        for n in sorted(speedups_by_n):
            v = speedups_by_n[n]
            if np.isfinite(v) and v >= threshold:
                best = n
                break
        return best

    def run_accuracy_benchmark(
        self,
        csv_name: str = "synthetic_arima_211.csv",
        value_column: Optional[str] = None,
        order: Tuple[int, int, int] = (1, 1, 1),
        test_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        path = self.sampler_dir / csv_name
        if not path.exists():
            raise FileNotFoundError(f"Sampler file not found: {path}")

        df = pd.read_csv(path)
        if value_column is None:
            for cand in ("value", "y", "passengers", "close", "temperature", "sunspots"):
                if cand in df.columns:
                    value_column = cand
                    break
            if value_column is None:
                numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                value_column = numeric[-1] if numeric else "value"
        if value_column not in df.columns:
            raise ValueError(f"Column {value_column!r} not in {df.columns.tolist()}")

        y = pd.to_numeric(df[value_column], errors="coerce").values.astype(float)
        if np.any(np.isnan(y)):
            raise ValueError("Series contains NaN; sampler datasets must be complete.")

        n = len(y)
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_test
        train, actual = y[:n_train], y[n_train:]
        horizon = len(actual)

        metrics: Dict[str, Dict[str, float]] = {}

        pred_lin = _linear_fit_predict(train, order, horizon)
        metrics["tslib_linear"] = self._error_block(actual, pred_lin)

        try:
            pred_sm_loc = statsmodels_arima_forecast(train, order, horizon)
            m = min(len(np.asarray(actual)), len(np.asarray(pred_sm_loc).ravel()))
            metrics["statsmodels_local"] = self._error_block(
                np.asarray(actual)[:m],
                np.asarray(pred_sm_loc, dtype=float).ravel()[:m],
            )
        except Exception as e:
            metrics["statsmodels_local"] = {"error": str(e)}

        spark = None
        try:
            from tslib.spark.ensure import ensure_spark_session

            spark = ensure_spark_session(app_name="tslib_arima_bench_acc", register_global=False)
        except Exception:
            pass

        if spark is not None:
            try:
                pred_w = _workflow_fit_predict(train, horizon, verbose=False)
                metrics["parallel_workflow"] = self._error_block(actual, pred_w)
            except Exception as e:
                metrics["parallel_workflow"] = {"error": str(e)}
            try:
                pred_sm = np.array(
                    _spark_statsmodels_forecast(spark, train, order, horizon), dtype=float
                )
                if len(pred_sm) >= len(actual):
                    pred_sm = pred_sm[: len(actual)]
                metrics["spark_statsmodels"] = self._error_block(actual, pred_sm)
            except Exception as e:
                metrics["spark_statsmodels"] = {"error": str(e)}
        else:
            metrics["parallel_workflow"] = {"error": "Spark not available"}
            metrics["spark_statsmodels"] = {"error": "Spark not available"}

        return {
            "csv": csv_name,
            "n_total": n,
            "n_train": n_train,
            "n_test": n_test,
            "order": order,
            "metrics": metrics,
        }

    @staticmethod
    def _error_block(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        a = np.asarray(actual, dtype=float)
        p = np.asarray(predicted, dtype=float).ravel()
        m = min(len(a), len(p))
        if m == 0:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
        a, p = a[:m], p[:m]
        return {
            "rmse": float(ForecastMetrics.rmse(a, p)),
            "mae": float(ForecastMetrics.mae(a, p)),
            "mape": float(ForecastMetrics.mape(a, p)),
        }

    def build_performance_figure(self, perf: Dict[str, Any]) -> plt.Figure:
        """Two panels: fit times vs N, speedups vs N."""
        grid = perf["n_obs_grid"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax0, ax1 = axes

        t_lin = [perf["linear_times"].get(n, float("nan")) for n in grid]
        t_wf = [perf["workflow_times"].get(n, float("nan")) for n in grid]
        t_sm = [perf["spark_statsmodels_times"].get(n, float("nan")) for n in grid]
        t_sml = [perf.get("statsmodels_local_times", {}).get(n, float("nan")) for n in grid]

        ax0.plot(grid, t_lin, "o-", label="TSLib lineal (n_jobs=1)", color="#ff6b6b")
        ax0.plot(grid, t_wf, "s-", label="ParallelARIMAWorkflow", color="#10ac84")
        ax0.plot(grid, t_sm, "^-", label="Spark + statsmodels ARIMA", color="#54a0ff")
        ax0.plot(
            grid,
            t_sml,
            "d-",
            label="statsmodels local (ref. eval.)",
            color="#feca57",
        )
        ax0.set_xscale("log")
        ax0.set_xlabel("n_obs")
        ax0.set_ylabel("Tiempo de ajuste (s)")
        ax0.set_title("Rendimiento: tiempo de fit")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8)

        sw = perf.get("speedup_linear_vs_workflow") or {}
        ss = perf.get("speedup_linear_vs_spark_sm") or {}
        ssl = perf.get("speedup_linear_vs_statsmodels_local") or {}
        ax1.axhline(1.0, color="#8395a7", linestyle="--", alpha=0.7, label="Speedup = 1")
        if sw:
            ax1.plot(
                sorted(sw.keys()),
                [sw[k] for k in sorted(sw.keys())],
                "o-",
                label="Lineal / Workflow",
                color="#10ac84",
            )
        if ss:
            ax1.plot(
                sorted(ss.keys()),
                [ss[k] for k in sorted(ss.keys())],
                "s-",
                label="Lineal / Spark+SM",
                color="#54a0ff",
            )
        if ssl:
            ax1.plot(
                sorted(ssl.keys()),
                [ssl[k] for k in sorted(ssl.keys())],
                "d-",
                label="Lineal / SM local",
                color="#feca57",
            )
        ax1.set_xscale("log")
        ax1.set_xlabel("n_obs")
        ax1.set_ylabel("Speedup")
        ax1.set_title("Aceleración vs serie lineal")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        cw = perf.get("crossover_workflow")
        cs = perf.get("crossover_spark_sm")
        cl = perf.get("crossover_statsmodels_local")
        subtitle = []
        if cw is not None:
            subtitle.append(f"N* workflow (speedup≥1): {cw}")
        if cs is not None:
            subtitle.append(f"N* Spark+SM (speedup≥1): {cs}")
        if cl is not None:
            subtitle.append(f"N* SM local (speedup≥1): {cl}")
        fig.suptitle(" | ".join(subtitle) if subtitle else "Benchmark ARIMA (TSLib + refs.)", fontsize=10)

        fig.patch.set_facecolor("#1a1a1a")
        for ax in axes:
            ax.set_facecolor("#262626")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            ax.title.set_color("white")
            for s in ax.spines.values():
                s.set_color("#444444")
            leg = ax.get_legend()
            if leg:
                for t in leg.get_texts():
                    t.set_color("white")
                leg.get_frame().set_facecolor("#1a1a1a")

        plt.tight_layout()
        return fig

    def build_accuracy_figure(self, acc: Dict[str, Any]) -> plt.Figure:
        """Grouped bars for RMSE / MAE / MAPE by method."""
        metrics = acc.get("metrics") or {}
        methods = []
        rmse, mae, mape = [], [], []
        palette = ["#ff6b6b", "#10ac84", "#54a0ff", "#feca57"]
        for key, label in [
            ("tslib_linear", "Lineal TSLib"),
            ("parallel_workflow", "Workflow Spark"),
            ("spark_statsmodels", "Spark+statsmodels"),
            ("statsmodels_local", "statsmodels local (ref.)"),
        ]:
            block = metrics.get(key) or {}
            if "error" in block:
                continue
            methods.append(label)
            rmse.append(block.get("rmse", float("nan")))
            mae.append(block.get("mae", float("nan")))
            mape.append(block.get("mape", float("nan")))

        if not methods:
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.text(0.5, 0.5, "Sin métricas comparables", ha="center", va="center", color="gray")
            ax.axis("off")
            fig.patch.set_facecolor("#1a1a1a")
            return fig

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        titles = ["RMSE", "MAE", "MAPE (%)"]
        series = [rmse, mae, mape]
        x = np.arange(len(methods))
        for ax, vals, tit in zip(axes, series, titles):
            ax.bar(x, vals, color=palette[: len(methods)])
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
            ax.set_title(tit)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            f"Precisión holdout — {acc.get('csv')} (orden {acc.get('order')})",
            fontsize=10,
        )
        fig.patch.set_facecolor("#1a1a1a")
        for ax in axes:
            ax.set_facecolor("#262626")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            ax.yaxis.label.set_color("#cccccc")
        plt.tight_layout()
        return fig

