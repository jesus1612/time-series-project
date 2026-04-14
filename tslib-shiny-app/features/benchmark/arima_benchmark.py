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

# Five-tone palette: TSLib, ParallelARIMAWorkflow, Spark+statsmodels, statsmodels ref, neutrals
COLOR_TSLIB = "#E07A5F"
COLOR_WORKFLOW = "#2A9D8F"
COLOR_SPARK_SM = "#457B9D"
COLOR_SM_REF = "#E9C46A"
COLOR_NEUTRAL = "#94A3B8"

BENCHMARK_PALETTE = (COLOR_TSLIB, COLOR_WORKFLOW, COLOR_SPARK_SM, COLOR_SM_REF)

COLOR_BY_METRIC_KEY = {
    "tslib_linear": COLOR_TSLIB,
    "parallel_workflow": COLOR_WORKFLOW,
    "spark_statsmodels": COLOR_SPARK_SM,
    "statsmodels_local": COLOR_SM_REF,
}

# Short legend labels (plots + UI; avoid long parentheticals)
LABEL_TSLIB_LINEAR = "TSLib lineal"
LABEL_PARALLEL_WORKFLOW = "ParallelARIMAWorkflow"
LABEL_SPARK_STATSMODELS = "Spark · statsmodels"
LABEL_STATSMODELS_LOCAL = "statsmodels ref."


def default_sampler_datasets_dir() -> Path:
    """TT/sampler/datasets relative to this package root (tslib-shiny-app)."""
    return Path(__file__).resolve().parents[3] / "sampler" / "datasets"


def _validate_spark_session(spark: Any) -> Any:
    """
    Return spark if the JVM session can run a trivial job; else None.
    Prevents errors like 'NoneType' object has no attribute 'sc' from stale handles.
    """
    if spark is None:
        return None
    try:
        spark.range(1).count()
        return spark
    except Exception:
        return None


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


def _workflow_fit_predict(
    train: np.ndarray,
    horizon: int,
    verbose: bool = False,
    *,
    grid_mode: str = "auto_n",
    manual_max_p: Optional[int] = None,
    manual_max_q: Optional[int] = None,
) -> np.ndarray:
    from tslib.spark import ParallelARIMAWorkflow

    kw: Dict[str, Any] = {"verbose": verbose, "grid_mode": grid_mode}
    if grid_mode == "manual":
        kw["manual_max_p"] = manual_max_p
        kw["manual_max_q"] = manual_max_q
    w = ParallelARIMAWorkflow(**kw)
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
        grid_mode: str = "auto_n",
        manual_max_p: Optional[int] = None,
        manual_max_q: Optional[int] = None,
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

            spark = _validate_spark_session(
                ensure_spark_session(app_name="tslib_arima_bench", register_global=False)
            )
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
                        kw: Dict[str, Any] = {"verbose": False, "grid_mode": grid_mode}
                        if grid_mode == "manual":
                            kw["manual_max_p"] = manual_max_p
                            kw["manual_max_q"] = manual_max_q
                        w = ParallelARIMAWorkflow(**kw)
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
        grid_mode: str = "auto_n",
        manual_max_p: Optional[int] = None,
        manual_max_q: Optional[int] = None,
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

            spark = _validate_spark_session(
                ensure_spark_session(app_name="tslib_arima_bench_acc", register_global=False)
            )
        except Exception:
            pass

        if spark is not None:
            try:
                pred_w = _workflow_fit_predict(
                    train,
                    horizon,
                    verbose=False,
                    grid_mode=grid_mode,
                    manual_max_p=manual_max_p,
                    manual_max_q=manual_max_q,
                )
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

    @staticmethod
    def _style_dark_axis(ax: plt.Axes) -> None:
        ax.set_facecolor("#262626")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("white")
        for s in ax.spines.values():
            s.set_color("#444444")

    def build_performance_time_figure(self, perf: Dict[str, Any]) -> plt.Figure:
        """
        Fit times vs N with log-scaled Y so ParallelARIMAWorkflow does not flatten other series.
        """
        grid = perf["n_obs_grid"]
        fig, ax0 = plt.subplots(figsize=(11, 4.5))
        t_lin = [perf["linear_times"].get(n, float("nan")) for n in grid]
        t_wf = [perf["workflow_times"].get(n, float("nan")) for n in grid]
        t_sm = [perf["spark_statsmodels_times"].get(n, float("nan")) for n in grid]
        t_sml = [perf.get("statsmodels_local_times", {}).get(n, float("nan")) for n in grid]

        ax0.plot(grid, t_lin, "o-", label=LABEL_TSLIB_LINEAR, color=COLOR_TSLIB, markersize=6)
        ax0.plot(grid, t_wf, "s-", label=LABEL_PARALLEL_WORKFLOW, color=COLOR_WORKFLOW, markersize=6)
        ax0.plot(grid, t_sm, "^-", label=LABEL_SPARK_STATSMODELS, color=COLOR_SPARK_SM, markersize=6)
        ax0.plot(grid, t_sml, "d-", label=LABEL_STATSMODELS_LOCAL, color=COLOR_SM_REF, markersize=6)
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        positive = [x for x in t_lin + t_wf + t_sm + t_sml if np.isfinite(x) and x > 0]
        if positive:
            ax0.set_ylim(bottom=max(1e-4, min(positive) * 0.5))
        ax0.set_xlabel("n_obs (muestras sintéticas)")
        ax0.set_ylabel("Tiempo de ajuste (s, escala log)")
        ax0.set_title("Tiempo de ajuste vs n — eje Y log · verde = ParallelARIMAWorkflow")
        ax0.grid(True, alpha=0.3)
        ax0.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=True,
            facecolor="#1a1a1a",
            edgecolor="#555555",
            labelcolor="white",
        )
        fig.patch.set_facecolor("#1a1a1a")
        self._style_dark_axis(ax0)
        leg = ax0.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_color("white")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.28)
        return fig

    def build_performance_speedup_figure(self, perf: Dict[str, Any]) -> plt.Figure:
        """Speedup vs TSLib lineal (ratio >1 means faster than linear TSLib)."""
        fig, ax1 = plt.subplots(figsize=(11, 4.2))
        sw = perf.get("speedup_linear_vs_workflow") or {}
        ss = perf.get("speedup_linear_vs_spark_sm") or {}
        ssl = perf.get("speedup_linear_vs_statsmodels_local") or {}
        ax1.axhline(1.0, color=COLOR_NEUTRAL, linestyle="--", alpha=0.7, label="Referencia 1×")
        if sw:
            ks = sorted(sw.keys())
            ax1.plot(
                ks,
                [sw[k] for k in ks],
                "o-",
                label=f"÷ {LABEL_PARALLEL_WORKFLOW}",
                color=COLOR_WORKFLOW,
            )
        if ss:
            ks = sorted(ss.keys())
            ax1.plot(
                ks,
                [ss[k] for k in ks],
                "s-",
                label=f"÷ {LABEL_SPARK_STATSMODELS}",
                color=COLOR_SPARK_SM,
            )
        if ssl:
            ks = sorted(ssl.keys())
            ax1.plot(
                ks,
                [ssl[k] for k in ks],
                "d-",
                label=f"÷ {LABEL_STATSMODELS_LOCAL}",
                color=COLOR_SM_REF,
            )
        ax1.set_xscale("log")
        ax1.set_xlabel("n_obs")
        ax1.set_ylabel("Speedup (t_lineal / t_otro)")
        ax1.set_title("Aceleración respecto a TSLib lineal")
        ax1.grid(True, alpha=0.3)
        ax1.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            frameon=True,
            facecolor="#1a1a1a",
            edgecolor="#555555",
            labelcolor="white",
        )
        cw = perf.get("crossover_workflow")
        cs = perf.get("crossover_spark_sm")
        cl = perf.get("crossover_statsmodels_local")
        lines = []
        if cw is not None:
            lines.append(f"N* workflow≥1: {cw}")
        if cs is not None:
            lines.append(f"N* Spark+SM≥1: {cs}")
        if cl is not None:
            lines.append(f"N* SM local≥1: {cl}")
        if lines:
            fig.suptitle(" | ".join(lines), fontsize=9, color="#aaaaaa", y=1.02)
        fig.patch.set_facecolor("#1a1a1a")
        self._style_dark_axis(ax1)
        leg = ax1.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_color("white")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        return fig

    def build_performance_figure(self, perf: Dict[str, Any]) -> plt.Figure:
        """Backward-compatible: single figure with time + speedup (prefer split figures in UI)."""
        return self.build_performance_time_figure(perf)

    def build_exploratory_diagnostics_figure(
        self,
        train: np.ndarray,
        order: Tuple[int, int, int] = (1, 1, 1),
    ) -> plt.Figure:
        """
        ACF/PACF of training series + residuals and Q-Q after TSLib linear fit (same order as benchmark).
        """
        from scipy import stats as scipy_stats
        from tslib.core.acf_pacf import ACFCalculator, PACFCalculator

        train = np.asarray(train, dtype=float).ravel()
        max_lag = max(5, min(40, len(train) // 4))
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))

        acf_calc = ACFCalculator(max_lags=max_lag, n_jobs=1)
        lags_a, acf_v = acf_calc.calculate(train)
        axes[0, 0].vlines(lags_a, 0, acf_v, color=COLOR_WORKFLOW, linewidth=1.2)
        axes[0, 0].axhline(0, color="#888", lw=0.8)
        axes[0, 0].set_title("ACF (serie de entrenamiento)")
        axes[0, 0].set_xlabel("Retardo")

        pacf_calc = PACFCalculator(max_lags=max_lag, n_jobs=1)
        lags_p, pacf_v = pacf_calc.calculate(train)
        axes[0, 1].vlines(lags_p, 0, pacf_v, color=COLOR_SPARK_SM, linewidth=1.2)
        axes[0, 1].axhline(0, color="#888", lw=0.8)
        axes[0, 1].set_title("PACF (serie de entrenamiento)")
        axes[0, 1].set_xlabel("Retardo")

        m = ARIMAModel(order=order, auto_select=False, validation=False, n_jobs=1)
        m.fit(train)
        res = m.get_residuals()
        axes[1, 0].plot(res, color=COLOR_SM_REF, lw=0.9)
        axes[1, 0].axhline(0, color=COLOR_TSLIB, ls="--", lw=0.8)
        axes[1, 0].set_title(f"Residuos TSLib lineal ARIMA{order}")
        axes[1, 0].set_xlabel("t")

        if len(res) > 8 and np.std(res) > 1e-12:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scipy_stats.probplot(res, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title("Q-Q residuos (normalidad aprox.)")
            axes[1, 1].get_lines()[0].set_markerfacecolor(COLOR_WORKFLOW)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Muestra corta para Q-Q",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                color="gray",
            )

        fig.suptitle(
            f"Diagnóstico exploratorio (train, orden fijo {order})",
            fontsize=11,
            color="white",
        )
        fig.patch.set_facecolor("#1a1a1a")
        for ax in axes.flat:
            self._style_dark_axis(ax)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def build_error_by_horizon_figure(
        self,
        csv_name: str,
        order: Tuple[int, int, int] = (1, 1, 1),
        test_ratio: float = 0.2,
        grid_mode: str = "auto_n",
        manual_max_p: Optional[int] = None,
        manual_max_q: Optional[int] = None,
    ) -> Optional[plt.Figure]:
        """
        |error| per out-of-sample step: ParallelARIMAWorkflow (when Spark is available) vs
        TSLib lineal, statsmodels local, and optionally Spark+statsmodels — same routes as
        run_accuracy_benchmark.
        """
        path = self.sampler_dir / csv_name
        if not path.exists():
            return None
        df = pd.read_csv(path)
        value_column = None
        for cand in ("value", "y", "passengers", "close", "temperature", "sunspots"):
            if cand in df.columns:
                value_column = cand
                break
        if value_column is None:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            value_column = numeric[-1] if numeric else None
        if not value_column:
            return None
        y = pd.to_numeric(df[value_column], errors="coerce").values.astype(float)
        n = len(y)
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_test
        train, actual = y[:n_train], y[n_train:]
        horizon = len(actual)
        actual = np.asarray(actual, dtype=float)

        series: List[Tuple[str, np.ndarray, str, float]] = []
        try:
            pred_lin = _linear_fit_predict(train, order, horizon)
            series.append((LABEL_TSLIB_LINEAR, np.asarray(pred_lin, dtype=float).ravel(), COLOR_TSLIB, 1.1))
        except Exception:
            pred_lin = None

        try:
            pred_sm = statsmodels_arima_forecast(train, order, horizon)
            series.append(
                (
                    LABEL_STATSMODELS_LOCAL,
                    np.asarray(pred_sm, dtype=float).ravel(),
                    COLOR_SM_REF,
                    1.1,
                )
            )
        except Exception:
            pass

        spark = None
        try:
            from tslib.spark.ensure import ensure_spark_session

            spark = _validate_spark_session(
                ensure_spark_session(app_name="tslib_arima_bench_horizon", register_global=False)
            )
        except Exception:
            pass

        if spark is not None:
            try:
                pred_w = _workflow_fit_predict(
                    train,
                    horizon,
                    verbose=False,
                    grid_mode=grid_mode,
                    manual_max_p=manual_max_p,
                    manual_max_q=manual_max_q,
                )
                series.append(
                    (
                        LABEL_PARALLEL_WORKFLOW,
                        np.asarray(pred_w, dtype=float).ravel(),
                        COLOR_WORKFLOW,
                        2.0,
                    )
                )
            except Exception:
                pass
            try:
                pred_ss = np.array(
                    _spark_statsmodels_forecast(spark, train, order, horizon), dtype=float
                ).ravel()
                if len(pred_ss) > 0:
                    series.append((LABEL_SPARK_STATSMODELS, pred_ss, COLOR_SPARK_SM, 1.1))
            except Exception:
                pass

        if not series:
            return None

        lengths = [len(np.asarray(p[1]).ravel()) for p in series]
        m = min(int(len(actual)), min(lengths))
        if m <= 0:
            return None
        actual_m = actual[:m]
        x = np.arange(1, m + 1)

        fig, ax = plt.subplots(figsize=(12, 4.2))
        for label, pred, color, lw in series:
            p = np.asarray(pred, dtype=float).ravel()[:m]
            err = np.abs(actual_m - p)
            z = 4 if label == LABEL_PARALLEL_WORKFLOW else 2
            ax.plot(x, err, label=label, color=color, linewidth=lw, zorder=z, alpha=0.92)

        ax.set_xlabel("Horizonte (pasos fuera de muestra)")
        ax.set_ylabel("|error|")
        ax.set_title(
            f"|Error| por horizonte — {csv_name} · énfasis en {LABEL_PARALLEL_WORKFLOW} (línea gruesa)"
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            fontsize=8,
            facecolor="#1a1a1a",
            edgecolor="#555555",
            labelcolor="white",
        )
        ax.grid(True, axis="y", alpha=0.25)
        fig.patch.set_facecolor("#1a1a1a")
        self._style_dark_axis(ax)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        return fig

    def build_accuracy_figure(self, acc: Dict[str, Any]) -> plt.Figure:
        """Grouped bars for RMSE / MAE / MAPE by method."""
        metrics = acc.get("metrics") or {}
        methods = []
        rmse, mae, mape = [], [], []
        bar_colors: List[str] = []
        for key, label in [
            ("tslib_linear", LABEL_TSLIB_LINEAR),
            ("parallel_workflow", LABEL_PARALLEL_WORKFLOW),
            ("spark_statsmodels", LABEL_SPARK_STATSMODELS),
            ("statsmodels_local", LABEL_STATSMODELS_LOCAL),
        ]:
            block = metrics.get(key) or {}
            if "error" in block:
                continue
            methods.append(label)
            bar_colors.append(COLOR_BY_METRIC_KEY.get(key, COLOR_NEUTRAL))
            rmse.append(block.get("rmse", float("nan")))
            mae.append(block.get("mae", float("nan")))
            mape.append(block.get("mape", float("nan")))

        if not methods:
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.text(0.5, 0.5, "Sin métricas comparables", ha="center", va="center", color="gray")
            ax.axis("off")
            fig.patch.set_facecolor("#1a1a1a")
            return fig

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        titles = ["RMSE", "MAE", "MAPE (%)"]
        series = [rmse, mae, mape]
        x = np.arange(len(methods))
        for ax, vals, tit in zip(axes, series, titles):
            ax.bar(x, vals, color=bar_colors)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=22, ha="right", fontsize=7)
            ax.set_title(tit)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            f"Holdout · {acc.get('csv')} · ARIMA{acc.get('order')}",
            fontsize=10,
            color="white",
        )
        fig.patch.set_facecolor("#1a1a1a")
        for ax in axes:
            ax.set_facecolor("#262626")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            ax.yaxis.label.set_color("#cccccc")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(bottom=0.28)
        return fig

