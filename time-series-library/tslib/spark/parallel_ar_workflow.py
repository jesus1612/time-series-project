"""
Parallel AR Workflow (classic AR(p) on stationary working data)

Mirrors the parallel ARIMA methodology (stationarity → sliding grid → Spark MLE per
window → global selection → validation → diagnostics → final refit) but the model
family is **AR(p)** only using ``ARProcess`` (MLE). Preprocessing (log + differencing)
matches Step 1 of ``ParallelARIMAWorkflow`` so the series passed to AR is stationary.

Parallel steps (Spark): sliding-window fits, optional full-sample loop (driver),
fixed-window backtest, residual diagnostics map.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.arima import ARProcess
from ..core.stationarity import StationarityAnalyzer
from ..core.arima_order_suggestion import suggest_p_q_orders
from ..metrics.evaluation import InformationCriteria, ForecastMetrics, ResidualAnalyzer
from ..preprocessing.transformations import LogTransformer
from ..utils.checks import check_spark_availability
from .core import get_optimized_spark_config
from .ensure import ensure_spark_session

try:
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


class ParallelARWorkflow:
    """
    Full parallel workflow for classic AR(p) with Spark-accelerated grid search.

    Attributes
    ----------
    order_ : tuple
        Selected order ``(p,)`` (compatible with app / TSLib AR).
    working_data_ : ndarray
        Series used for the final AR fit (after log / differencing).
    differencing_order_ : int
        Number of differences applied in Step 1 (for documentation).
    parameters_ : dict
        phi, c, sigma2 from the final ``ARProcess``.
    """

    def __init__(
        self,
        spark_session=None,
        spark_config: Optional[Dict[str, str]] = None,
        master: Optional[str] = None,
        app_name: Optional[str] = None,
        verbose: bool = True,
        grid_mode: str = "auto_n",
        manual_max_p: Optional[int] = None,
        d_max: int = 2,
        acf_pacf_alpha: float = 0.05,
        acf_pacf_max_lag: Optional[int] = None,
        full_sample_reconcile: bool = True,
        full_sample_reconcile_top_k: int = 12,
    ):
        if not check_spark_availability():
            from .ensure import DISTRIBUTED_REQUIRES_SPARK

            raise ImportError(DISTRIBUTED_REQUIRES_SPARK)

        self.verbose = verbose
        self.fitted_ = False

        if spark_session is None:
            cfg = dict(spark_config if spark_config is not None else get_optimized_spark_config(1000))
            cfg["spark.sql.execution.arrow.pyspark.enabled"] = "false"
            self.spark = ensure_spark_session(
                spark_session=None,
                spark_config=cfg,
                master=master,
                app_name=app_name or "TSLib-ParallelARWorkflow",
                register_global=True,
            )
            self._owns_spark = True
        else:
            self.spark = ensure_spark_session(
                spark_session=spark_session,
                spark_config=None,
                master=None,
                app_name=None,
                register_global=True,
            )
            self._owns_spark = False
            try:
                self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
            except Exception:
                pass

        self._stationarity_analyzer = StationarityAnalyzer()
        self._log_transformer = None
        self._residual_analyzer = ResidualAnalyzer()

        self.data_ = None
        self.working_data_ = None
        self.differencing_order_: int = 0
        self.spark_timing_: Dict[str, float] = {}
        self.order_: Optional[Tuple[int, ...]] = None
        self.parameters_: Optional[Dict[str, Any]] = None
        self._final_model: Optional[ARProcess] = None
        self.results_: Dict[str, Any] = {
            "step1_differencing": None,
            "step2_3_grid": None,
            "step4_5_sliding_fitting": None,
            "step6_model_selection": None,
            "step6_full_sample_reconciliation": None,
            "step7_8_validation": None,
            "step9_diagnostics": None,
            "step10_adjustment": None,
            "config": None,
        }

        self._config: Dict[str, Any] = {
            "max_p": None,
            "num_sliding_windows": None,
            "num_fixed_windows": None,
            "overlap_sliding": None,
            "significance_level": 0.05,
        }

        if grid_mode not in ("auto_n", "acf_pacf", "manual"):
            raise ValueError("grid_mode must be 'auto_n', 'acf_pacf', or 'manual'")
        self._grid_mode = grid_mode
        self._manual_max_p = manual_max_p
        self._d_max = int(max(0, d_max))
        self._acf_pacf_alpha = float(acf_pacf_alpha)
        self._acf_pacf_max_lag = acf_pacf_max_lag
        self._full_sample_reconcile = bool(full_sample_reconcile)
        self._full_sample_reconcile_top_k = max(1, int(full_sample_reconcile_top_k))

    # --- Step 1: same stationarity loop as ParallelARIMAWorkflow -----------------
    def _calculate_variance_growth(self, data: np.ndarray) -> float:
        n = len(data)
        if n < 20:
            return 1.0
        half = n // 2
        var_first = float(np.var(data[:half]))
        var_second = float(np.var(data[half:]))
        if var_first < 1e-12:
            return 100.0
        return var_second / var_first

    def _determine_differencing_order(
        self, data: np.ndarray
    ) -> Tuple[int, bool, Optional[LogTransformer]]:
        variance_growth = self._calculate_variance_growth(data)
        log_needed = variance_growth > 1.5
        if log_needed:
            transformer = LogTransformer()
            transformer.fit(data)
            transformed_data = transformer.transform(data)
        else:
            transformer = None
            transformed_data = data.copy()

        if len(transformed_data) < 4:
            stationarity_results = self._stationarity_analyzer.analyze(transformed_data)
            d = int(stationarity_results["suggested_differencing_order"])
            self.results_["step1_differencing"] = {
                "d": d,
                "log_transform_needed": log_needed,
                "variance_growth": variance_growth,
                "stationarity_results": {**stationarity_results, "iterations": []},
            }
            return d, log_needed, transformer

        y = transformed_data.astype(float)
        iterations: List[Dict[str, Any]] = []
        current_d = 0
        last_adf = None
        last_kpss = None
        stationary_reached = False
        insufficient_after_diff = False

        while True:
            last_adf = self._stationarity_analyzer.adf_test.test(y)
            last_kpss = self._stationarity_analyzer.kpss_test.test(y)
            is_stat = self._stationarity_analyzer._determine_stationarity(last_adf, last_kpss)
            iterations.append(
                {
                    "d": current_d,
                    "n_obs": int(len(y)),
                    "adf_stationary": bool(last_adf["is_stationary"]),
                    "kpss_stationary": bool(last_kpss["is_stationary"]),
                }
            )
            if is_stat:
                stationary_reached = True
                break
            if current_d >= self._d_max:
                break
            y_new = np.diff(y)
            if len(y_new) < 4:
                insufficient_after_diff = True
                break
            y = y_new
            current_d += 1

        if stationary_reached:
            d = current_d
        elif insufficient_after_diff:
            d = min(current_d + 1, self._d_max)
        else:
            d = self._d_max

        stationarity_results = {
            "adf_test": last_adf,
            "kpss_test": last_kpss,
            "is_stationary": stationary_reached,
            "suggested_differencing_order": d,
            "iterations": iterations,
        }
        self.results_["step1_differencing"] = {
            "d": d,
            "log_transform_needed": log_needed,
            "variance_growth": variance_growth,
            "stationarity_results": stationarity_results,
        }
        return d, log_needed, transformer

    def _determine_parameter_ranges(self, n_obs: int) -> Dict[str, int]:
        if n_obs < 500:
            cfg = {
                "max_p": 3,
                "num_sliding_windows": max(3, min(5, n_obs // 100)),
                "num_fixed_windows": max(3, n_obs // 150),
                "overlap_sliding": 0.3,
            }
        elif n_obs < 2000:
            cfg = {
                "max_p": 5,
                "num_sliding_windows": max(5, min(10, n_obs // 150)),
                "num_fixed_windows": max(4, n_obs // 200),
                "overlap_sliding": 0.2,
            }
        else:
            cfg = {
                "max_p": 5,
                "num_sliding_windows": max(10, min(20, n_obs // 150)),
                "num_fixed_windows": max(5, n_obs // 250),
                "overlap_sliding": 0.15,
            }
        self._config.update(cfg)
        return cfg

    def _apply_grid_mode(
        self, config: Dict[str, Any], working_data: np.ndarray, d_prep: int, n_obs: int
    ) -> Dict[str, Any]:
        if self._grid_mode == "manual":
            if self._manual_max_p is None:
                raise ValueError("grid_mode='manual' requires manual_max_p.")
            out = {**config, "max_p": int(self._manual_max_p)}
            return out
        if self._grid_mode == "acf_pacf":
            cap_p = int(config["max_p"])
            mp, _mq, meta = suggest_p_q_orders(
                working_data,
                d_prep,
                max_lag=self._acf_pacf_max_lag,
                alpha=self._acf_pacf_alpha,
                max_p_bound=cap_p,
                max_q_bound=cap_p,
            )
            self.results_["acf_pacf_identification"] = meta
            return {**config, "max_p": mp}
        return config

    def _p_candidates(self, max_p: int) -> List[int]:
        return [p for p in range(1, max_p + 1)]

    def _create_sliding_windows(
        self, data: np.ndarray, num_windows: int, overlap: float
    ) -> List[Dict[str, Any]]:
        n = len(data)
        if n < 1:
            return []
        num_windows = max(1, int(num_windows))
        denom = num_windows * (1 - overlap) + overlap
        window_size = int(n / denom) if denom > 0 else n
        window_size = max(1, min(window_size, n))
        min_window_size = min(n, max(10, n // max(num_windows + 5, 1)))
        window_size = max(window_size, min_window_size)
        window_size = min(window_size, n)
        step_size = max(1, int(window_size * (1 - overlap)))
        windows: List[Dict[str, Any]] = []
        window_id = 0
        for start_idx in range(0, n - window_size + 1, step_size):
            end_idx = min(start_idx + window_size, n)
            windows.append(
                {
                    "window_id": window_id,
                    "data": data[start_idx:end_idx],
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "size": end_idx - start_idx,
                }
            )
            window_id += 1
            if window_id >= num_windows:
                break
        if not windows:
            arr = np.asarray(data, dtype=float)
            windows.append(
                {
                    "window_id": 0,
                    "data": arr.copy(),
                    "start_idx": 0,
                    "end_idx": n,
                    "size": n,
                }
            )
        return windows

    def _fit_sliding_spark(self, windows: List[Dict[str, Any]], p_list: List[int]) -> pd.DataFrame:
        import time as _time

        tasks_list = []
        for w in windows:
            for p in p_list:
                tasks_list.append(
                    {
                        "window_id": w["window_id"],
                        "window_data": w["data"].tolist(),
                        "window_size": len(w["data"]),
                        "p": p,
                    }
                )
        if not tasks_list:
            raise ValueError("Parallel AR: no sliding tasks.")

        tasks_pandas = pd.DataFrame(tasks_list)
        schema = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("window_data", ArrayType(DoubleType()), True),
                StructField("window_size", IntegerType(), True),
                StructField("p", IntegerType(), True),
            ]
        )
        tasks_df = self.spark.createDataFrame(tasks_pandas, schema=schema)
        num_partitions = max(1, min(len(tasks_pandas), self.spark.sparkContext.defaultParallelism))
        tasks_df = tasks_df.repartition(num_partitions)

        td0 = _time.perf_counter()
        tasks_df = tasks_df.cache()
        try:
            tasks_df.count()
        except Exception:
            pass
        self.spark_timing_["tasks_dataframe_distribute_s"] = float(_time.perf_counter() - td0)

        output_schema = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("p", IntegerType(), True),
                StructField("success", BooleanType(), True),
                StructField("aicc", DoubleType(), True),
                StructField("aic", DoubleType(), True),
                StructField("bic", DoubleType(), True),
                StructField("log_likelihood", DoubleType(), True),
                StructField("phi", ArrayType(DoubleType()), True),
                StructField("c", DoubleType(), True),
                StructField("sigma2", DoubleType(), True),
                StructField("error", StringType(), True),
            ]
        )

        def fit_map(iterator):
            import numpy as np
            import pandas as pd
            import warnings

            try:
                from tslib.core.arima import ARProcess
                from tslib.metrics.evaluation import InformationCriteria
            except ImportError:
                import os
                import sys

                for path in sys.path:
                    if os.path.exists(os.path.join(path, "tslib")):
                        if path not in sys.path:
                            sys.path.insert(0, path)
                        break
                from tslib.core.arima import ARProcess
                from tslib.metrics.evaluation import InformationCriteria

            rows = []
            for pdf in iterator:
                for _, row in pdf.iterrows():
                    wid = int(row["window_id"])
                    p = int(row["p"])
                    wd = np.asarray(row["window_data"], dtype=float)
                    try:
                        min_obs = max(p + 10, 30)
                        if len(wd) < min_obs:
                            raise ValueError(f"short window {len(wd)}")
                        model = ARProcess(order=p, trend="c", n_jobs=1)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(wd)
                        fp = model._fitted_params
                        ll = float(fp["log_likelihood"])
                        n_obs = len(wd)
                        n_params = p + 1
                        aicc = InformationCriteria.aicc(ll, n_params, n_obs)
                        phi = [float(fp["parameters"].get(f"phi_{i+1}", 0.0)) for i in range(p)]
                        c = float(model.constant if hasattr(model, "constant") else 0.0)
                        sig2 = float(fp["parameters"].get("sigma2", 0.0))
                        rows.append(
                            {
                                "window_id": wid,
                                "p": p,
                                "success": True,
                                "aicc": float(aicc),
                                "aic": float(fp["aic"]),
                                "bic": float(fp["bic"]),
                                "log_likelihood": ll,
                                "phi": phi,
                                "c": c,
                                "sigma2": sig2,
                                "error": "",
                            }
                        )
                    except Exception as e:
                        rows.append(
                            {
                                "window_id": wid,
                                "p": p,
                                "success": False,
                                "aicc": np.inf,
                                "aic": np.inf,
                                "bic": np.inf,
                                "log_likelihood": np.nan,
                                "phi": [],
                                "c": 0.0,
                                "sigma2": 0.0,
                                "error": str(e)[:200],
                            }
                        )
            if rows:
                yield pd.DataFrame(rows)

        if self.verbose:
            print("  🚀 Parallel AR sliding fit (Spark mapInPandas)...")
            t0 = _time.time()
        out = tasks_df.mapInPandas(fit_map, schema=output_schema).toPandas()
        if self.verbose:
            print(f"  ⏱️  Sliding fit done in {_time.time() - t0:.2f}s")
        self.results_["step4_5_sliding_fitting"] = {
            "results_df": out,
            "num_windows": len(windows),
            "p_list": p_list,
        }
        return out

    def _select_global_p(self, results_df: pd.DataFrame) -> Tuple[int, float]:
        ok = results_df[results_df["success"] == True].copy()
        if len(ok) == 0:
            raise ValueError("No successful AR fits in sliding step.")
        ok["rank"] = ok.groupby("window_id")["aicc"].rank(method="min")
        ms = ok.groupby("p").agg({"rank": ["mean", "std", "count"], "aicc": ["mean", "std"]}).reset_index()
        ms.columns = ["p", "rank_mean", "rank_std", "count", "aicc_mean", "aicc_std"]
        ms["score"] = ms["rank_mean"] + 0.5 * ms["rank_std"].fillna(0)
        best = ms.loc[ms["score"].idxmin()]
        p_best = int(best["p"])
        nw = results_df["window_id"].nunique()
        appearance = best["count"] / nw if nw else 0.0
        consistency = 1.0 / (1.0 + best["rank_std"]) if best["rank_std"] > 0 else 1.0
        conf = 0.6 * appearance + 0.4 * consistency
        self.results_["step6_model_selection"] = {
            "best_p": p_best,
            "confidence": conf,
            "model_scores": ms,
        }
        return p_best, float(conf)

    def _reconcile_full_sample(
        self, working_data: np.ndarray, sliding_p: int, conf: float
    ) -> Tuple[int, float]:
        step6 = self.results_.get("step6_model_selection") or {}
        ms = step6.get("model_scores")
        if ms is None or len(ms) == 0:
            self.results_["step6_full_sample_reconciliation"] = {"skipped": True}
            return sliding_p, conf

        data = np.asarray(working_data, dtype=float).ravel()
        k = min(self._full_sample_reconcile_top_k, len(ms))
        top = ms.nsmallest(k, "score")
        candidates = list(dict.fromkeys([int(r["p"]) for _, r in top.iterrows()]))
        if sliding_p not in candidates:
            candidates.insert(0, sliding_p)

        aic_by_p: Dict[int, float] = {}
        for p in candidates:
            try:
                m = ARProcess(order=p, trend="c", n_jobs=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(data)
                aic_by_p[p] = float(m._fitted_params["aic"])
            except Exception:
                continue
        if not aic_by_p:
            return sliding_p, conf
        best_p = min(aic_by_p.keys(), key=lambda x: aic_by_p[x])
        self.results_["step6_full_sample_reconciliation"] = {
            "skipped": False,
            "aic_by_p": aic_by_p,
            "selected_p": best_p,
        }
        return best_p, conf

    def _create_fixed_windows(
        self, data: np.ndarray, num_windows: int, min_train_pct: float = 0.7
    ) -> List[Dict[str, Any]]:
        n = len(data)
        p_ord = self.order_[0] if self.order_ else 1
        window_size = n // num_windows
        min_required = p_ord + 10
        min_window_size = int(min_required / min_train_pct) + 10
        window_size = max(window_size, min_window_size)
        windows: List[Dict[str, Any]] = []
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, n)
            if end_idx - start_idx < min_window_size:
                break
            window_data = data[start_idx:end_idx]
            train_size = int(len(window_data) * min_train_pct)
            train_data = window_data[:train_size]
            test_data = window_data[train_size:]
            if len(train_data) < min_required:
                continue
            windows.append(
                {
                    "window_id": i,
                    "train_data": train_data,
                    "test_data": test_data,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                }
            )
        return windows

    def _backtest(self, windows: List[Dict[str, Any]], p: int) -> pd.DataFrame:
        if not windows:
            return pd.DataFrame()

        tasks_list = []
        for w in windows:
            tasks_list.append(
                {
                    "window_id": int(w["window_id"]),
                    "train_data": np.asarray(w["train_data"], dtype=float).tolist(),
                    "test_data": np.asarray(w["test_data"], dtype=float).tolist(),
                    "train_size": int(w["train_size"]),
                    "test_size": int(w["test_size"]),
                    "p": p,
                }
            )
        tasks_pandas = pd.DataFrame(tasks_list)
        schema_in = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("train_data", ArrayType(DoubleType()), True),
                StructField("test_data", ArrayType(DoubleType()), True),
                StructField("train_size", IntegerType(), True),
                StructField("test_size", IntegerType(), True),
                StructField("p", IntegerType(), True),
            ]
        )
        tasks_df = self.spark.createDataFrame(tasks_pandas, schema=schema_in)
        npart = max(1, min(len(tasks_pandas), self.spark.sparkContext.defaultParallelism))
        tasks_df = tasks_df.repartition(npart)

        output_schema = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("p", IntegerType(), True),
                StructField("success", BooleanType(), True),
                StructField("mae", DoubleType(), True),
                StructField("rmse", DoubleType(), True),
                StructField("mape", DoubleType(), True),
                StructField("error", StringType(), True),
            ]
        )

        def bt_map(iterator):
            import numpy as np
            import pandas as pd
            import warnings

            try:
                from tslib.core.arima import ARProcess
                from tslib.metrics.evaluation import ForecastMetrics
            except ImportError:
                import os
                import sys

                for path in sys.path:
                    if os.path.exists(os.path.join(path, "tslib")):
                        if path not in sys.path:
                            sys.path.insert(0, path)
                        break
                from tslib.core.arima import ARProcess
                from tslib.metrics.evaluation import ForecastMetrics

            rows_out = []
            for pdf in iterator:
                for _, row in pdf.iterrows():
                    wid = int(row["window_id"])
                    pp = int(row["p"])
                    train_data = np.asarray(row["train_data"], dtype=float)
                    test_data = np.asarray(row["test_data"], dtype=float)
                    try:
                        model = ARProcess(order=pp, trend="c", n_jobs=1)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(train_data)
                        pred = model.predict(steps=len(test_data))
                        mae = float(ForecastMetrics.mae(test_data, pred))
                        rmse = float(ForecastMetrics.rmse(test_data, pred))
                        mape = float(ForecastMetrics.mape(test_data, pred))
                        rows_out.append(
                            {
                                "window_id": wid,
                                "p": pp,
                                "success": True,
                                "mae": mae,
                                "rmse": rmse,
                                "mape": mape,
                                "error": "",
                            }
                        )
                    except Exception as e:
                        rows_out.append(
                            {
                                "window_id": wid,
                                "p": pp,
                                "success": False,
                                "mae": np.nan,
                                "rmse": np.nan,
                                "mape": np.nan,
                                "error": str(e)[:500],
                            }
                        )
            if rows_out:
                yield pd.DataFrame(rows_out)

        results_df = tasks_df.mapInPandas(bt_map, schema=output_schema).toPandas()
        self.results_["step7_8_validation"] = {
            "results_df": results_df,
            "num_windows": len(windows),
        }
        return results_df

    def _diagnose(self, windows: List[Dict[str, Any]], p: int) -> pd.DataFrame:
        sig_level = float(self._config["significance_level"])
        tasks_list = [
            {
                "window_id": int(w["window_id"]),
                "train_data": np.asarray(w["train_data"], dtype=float).tolist(),
                "p": p,
            }
            for w in windows
        ]
        if not tasks_list:
            return pd.DataFrame()
        tasks_pandas = pd.DataFrame(tasks_list)
        schema_in = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("train_data", ArrayType(DoubleType()), True),
                StructField("p", IntegerType(), True),
            ]
        )
        tasks_df = self.spark.createDataFrame(tasks_pandas, schema=schema_in)
        npart = max(1, min(len(tasks_pandas), self.spark.sparkContext.defaultParallelism))
        tasks_df = tasks_df.repartition(npart)

        output_schema = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("p", IntegerType(), True),
                StructField("ljung_box_statistic", DoubleType(), True),
                StructField("ljung_box_p_value", DoubleType(), True),
                StructField("ljung_box_pass", BooleanType(), True),
                StructField("acf_significant_peaks", IntegerType(), True),
                StructField("acf_pass", BooleanType(), True),
                StructField("overall_pass", BooleanType(), True),
                StructField("success", BooleanType(), True),
                StructField("error", StringType(), True),
            ]
        )

        def diag_map(iterator):
            import numpy as np
            import pandas as pd
            import warnings

            try:
                from tslib.core.arima import ARProcess
                from tslib.metrics.evaluation import ResidualAnalyzer
            except ImportError:
                import os
                import sys

                for path in sys.path:
                    if os.path.exists(os.path.join(path, "tslib")):
                        if path not in sys.path:
                            sys.path.insert(0, path)
                        break
                from tslib.core.arima import ARProcess
                from tslib.metrics.evaluation import ResidualAnalyzer

            rows_out = []
            for pdf in iterator:
                for _, row in pdf.iterrows():
                    wid = int(row["window_id"])
                    pp = int(row["p"])
                    train_data = np.asarray(row["train_data"], dtype=float)
                    try:
                        model = ARProcess(order=pp, trend="c", n_jobs=1)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(train_data)
                        residuals = model.get_residuals()
                        fitted_values = model.get_fitted_values()
                        analyzer = ResidualAnalyzer()
                        ra = analyzer.analyze(residuals, fitted_values)
                        acf_test = ra["autocorrelation_tests"]
                        lb = ra["ljung_box_test"]
                        lb_pass = float(lb["p_value"]) > sig_level
                        acf_autocorrs = acf_test["autocorrelations"]
                        n = len(residuals)
                        conf_bound = 1.96 / np.sqrt(max(n, 1))
                        sig_peaks = sum(1 for ac in acf_autocorrs if abs(ac) > conf_bound)
                        acf_ok = sig_peaks <= 2
                        rows_out.append(
                            {
                                "window_id": wid,
                                "p": pp,
                                "ljung_box_statistic": float(lb["statistic"]),
                                "ljung_box_p_value": float(lb["p_value"]),
                                "ljung_box_pass": bool(lb_pass),
                                "acf_significant_peaks": int(sig_peaks),
                                "acf_pass": bool(acf_ok),
                                "overall_pass": bool(lb_pass and acf_ok),
                                "success": True,
                                "error": "",
                            }
                        )
                    except Exception as e:
                        rows_out.append(
                            {
                                "window_id": wid,
                                "p": pp,
                                "ljung_box_statistic": np.nan,
                                "ljung_box_p_value": np.nan,
                                "ljung_box_pass": False,
                                "acf_significant_peaks": -1,
                                "acf_pass": False,
                                "overall_pass": False,
                                "success": False,
                                "error": str(e)[:500],
                            }
                        )
            if rows_out:
                yield pd.DataFrame(rows_out)

        diagnostics_df = tasks_df.mapInPandas(diag_map, schema=output_schema).toPandas()
        self.results_["step9_diagnostics"] = {"diagnostics_df": diagnostics_df}
        return diagnostics_df

    def _check_failures(self, diagnostics_df: pd.DataFrame) -> Tuple[bool, List[int]]:
        if diagnostics_df is None or len(diagnostics_df) == 0:
            return False, []
        failed = diagnostics_df[diagnostics_df["overall_pass"] == False]["window_id"].tolist()
        if len(failed) < 2:
            return False, failed
        for i in range(len(failed) - 1):
            if failed[i + 1] - failed[i] == 1:
                return True, failed
        return False, failed

    def _local_adjust(self, data: np.ndarray, p: int) -> int:
        try:
            base = ARProcess(order=p, trend="c", n_jobs=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base.fit(data)
            bp = base._fitted_params
            n_obs = len(data)
            base_aicc = InformationCriteria.aicc(bp["log_likelihood"], p + 1, n_obs)
        except Exception:
            return p
        best_p, best_aicc = p, base_aicc
        for cand in (p + 1, p - 1):
            if cand < 1:
                continue
            try:
                m = ARProcess(order=cand, trend="c", n_jobs=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(data)
                fp = m._fitted_params
                aicc = InformationCriteria.aicc(fp["log_likelihood"], cand + 1, n_obs)
                imp = (base_aicc - aicc) / base_aicc * 100
                if aicc < best_aicc and imp > 10:
                    best_aicc = aicc
                    best_p = cand
            except Exception:
                continue
        self.results_["step10_adjustment"] = {"from_p": p, "to_p": best_p}
        return best_p

    def fit(self, data: Union[np.ndarray, pd.Series, list]) -> "ParallelARWorkflow":
        if isinstance(data, pd.Series):
            data = data.values
        else:
            data = np.asarray(data, dtype=float)
        self.data_ = data
        n = len(data)
        self.spark_timing_ = {}

        tw = time.perf_counter()
        try:
            self.spark.range(1).count()
        except Exception:
            pass
        self.spark_timing_["executor_warmup_s"] = float(time.perf_counter() - tw)

        d_prep, log_needed, transformer = self._determine_differencing_order(data)
        self.differencing_order_ = int(d_prep)
        self._log_transformer = transformer
        if log_needed and transformer:
            working_data = transformer.transform(data)
        else:
            working_data = data.copy()
        if d_prep > 0:
            y = working_data.astype(float)
            for _ in range(d_prep):
                y = np.diff(y)
            working_data = y

        cfg = self._determine_parameter_ranges(n)
        cfg = self._apply_grid_mode(cfg, working_data, d_prep, n)
        self._config.update(cfg)
        self.results_["config"] = dict(self._config, grid_mode=self._grid_mode, d_max=self._d_max)

        p_list = self._p_candidates(int(cfg["max_p"]))
        self.results_["step2_3_grid"] = {"max_p": cfg["max_p"], "p_list": p_list}

        sliding_windows = self._create_sliding_windows(
            working_data, cfg["num_sliding_windows"], cfg["overlap_sliding"]
        )
        slide_df = self._fit_sliding_spark(sliding_windows, p_list)
        best_p, conf = self._select_global_p(slide_df)
        if self._full_sample_reconcile:
            best_p, conf = self._reconcile_full_sample(working_data, best_p, conf)
        self.order_ = (best_p,)

        fixed_w = self._create_fixed_windows(working_data, cfg["num_fixed_windows"])
        self._backtest(fixed_w, best_p)
        diag_df = self._diagnose(fixed_w, best_p)
        need_adj, _ = self._check_failures(diag_df)
        if need_adj:
            best_p = self._local_adjust(working_data, best_p)
            self.order_ = (best_p,)

        p_final = self.order_[0]
        self._final_model = ARProcess(order=p_final, trend="c", n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._final_model.fit(working_data)
        fp = self._final_model._fitted_params
        self.parameters_ = {
            "phi": [fp["parameters"].get(f"phi_{i+1}", 0.0) for i in range(p_final)],
            "theta": [],
            "c": float(self._final_model.constant if hasattr(self._final_model, "constant") else 0.0),
            "sigma2": float(fp["parameters"].get("sigma2", 0.0)),
            "log_likelihood": fp["log_likelihood"],
            "aic": fp["aic"],
            "bic": fp["bic"],
        }
        self.working_data_ = np.asarray(working_data, dtype=float).copy()
        self.fitted_ = True
        self.results_["spark_timing"] = dict(self.spark_timing_)
        return self

    def predict(
        self,
        steps: int = 1,
        return_conf_int: bool = False,
        alpha: float = 0.05,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        if not self.fitted_ or self._final_model is None:
            raise ValueError("Workflow must be fitted before prediction.")
        if steps <= 0:
            raise ValueError("Steps must be positive")
        if return_conf_int:
            pred, ci = self._final_model.predict(steps=steps, return_conf_int=True)
            if self._log_transformer is not None:
                pred = self._log_transformer.inverse_transform(pred)
                ci = (
                    self._log_transformer.inverse_transform(ci[0]),
                    self._log_transformer.inverse_transform(ci[1]),
                )
            return pred, ci
        pred = self._final_model.predict(steps=steps, return_conf_int=False)
        if self._log_transformer is not None:
            pred = self._log_transformer.inverse_transform(pred)
        return pred

    def __del__(self):
        try:
            if getattr(self, "_owns_spark", False) and getattr(self, "spark", None) is not None:
                self.spark.stop()
        except Exception:
            pass
