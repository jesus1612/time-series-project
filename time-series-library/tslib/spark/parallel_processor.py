"""
GenericParallelProcessor — Spark-based parallel processor for all TSLib model types.

Extends the ARIMA infrastructure in parallel_arima.py to support AR, MA, ARMA,
and ARIMA models with a single, unified API.

Usage example (requires PySpark)::

    from tslib.spark.parallel_processor import GenericParallelProcessor

    processor = GenericParallelProcessor(model_type='AR', spark=spark, n_jobs=4)
    results_df = processor.fit_multiple(
        df=spark_df,
        group_col='series_id',
        value_col='y',
        order=2,         # For AR: p; For MA: q; For ARMA/ARIMA: (p,d,q)
        steps=10,
    )
"""

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType,
        DoubleType, ArrayType
    )
    import pandas as pd
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    'AR':    ('tslib.models.ar_model',   'ARModel'),
    'MA':    ('tslib.models.ma_model',   'MAModel'),
    'ARMA':  ('tslib.models.arma_model', 'ARMAModel'),
    'ARIMA': ('tslib.models.arima_model','ARIMAModel'),
}

def _build_model(model_type: str, order: Any, n_jobs: int = 1):
    """Instantiate the correct TSLib model class with the given order."""
    model_type = model_type.upper()
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type='{model_type}'. "
            f"Choose from: {list(_MODEL_REGISTRY)}"
        )
    module_path, class_name = _MODEL_REGISTRY[model_type]
    import importlib
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, class_name)

    if model_type == 'AR':
        # ARModel/ARProcess expect scalar p; callers may pass (p,) from shared tuple API
        p = int(order[0]) if isinstance(order, (tuple, list)) else int(order)
        return ModelClass(order=p, auto_select=False, n_jobs=n_jobs)
    elif model_type == 'MA':
        q = int(order[0]) if isinstance(order, (tuple, list)) else int(order)
        return ModelClass(order=q, auto_select=False, n_jobs=n_jobs)
    elif model_type == 'ARMA':
        p, q = order if hasattr(order, '__len__') else (order, 0)
        return ModelClass(order=(p, q), auto_select=False, n_jobs=n_jobs)
    else:  # ARIMA
        p, d, q = order if hasattr(order, '__len__') else (order, 0, 0)
        return ModelClass(order=(p, d, q), auto_select=False, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# applyInPandas group functions (Spark 3+)
# ---------------------------------------------------------------------------

def _fit_predict_group(
    pdf: 'pd.DataFrame',
    value_col: str,
    model_type: str,
    order: Any,
    steps: int,
    n_jobs: int,
) -> 'pd.DataFrame':
    """
    Fit a model on a single group and return forecast rows.

    Called inside Spark groupBy.applyInPandas — one iterator batch per group.
    """
    import warnings
    import pandas as pd

    group_id = pdf['_group_id'].iloc[0] if '_group_id' in pdf.columns else 'unknown'
    y = pdf[value_col].dropna().values
    captured: List[str] = []

    try:
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            model = _build_model(model_type, order, n_jobs=n_jobs)
            model.fit(y)
            captured = [str(w.message) for w in wrec]
        forecasts = model.predict(steps=steps)
        status = 'ok'
    except Exception as exc:
        forecasts = np.full(steps, np.nan)
        status = str(exc)[:200]

    notes_str = " | ".join(dict.fromkeys(captured))[:2000] if captured else ""

    result = pd.DataFrame({
        'series_id':  [group_id] * steps,
        'step':       list(range(1, steps + 1)),
        'forecast':   forecasts.tolist(),
        'status':     [status] * steps,
        'engine_warnings': [notes_str] * steps,
    })
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GenericParallelProcessor:
    """
    Unified Spark-based parallel processor for AR, MA, ARMA, and ARIMA models.

    Parameters
    ----------
    model_type : str
        One of 'AR', 'MA', 'ARMA', 'ARIMA' (case-insensitive).
    spark : SparkSession, optional
        Active Spark session. If None, attempts to get or create one.
    n_jobs : int
        Parallelism within each model (passed to the model constructor).
        -1 = all cores, 1 = sequential.

    Examples
    --------
    >>> proc = GenericParallelProcessor(model_type='AR', spark=spark)
    >>> results = proc.fit_multiple(df, group_col='id', value_col='y',
    ...                             order=2, steps=5)
    """

    def __init__(
        self,
        model_type: str = 'ARIMA',
        spark: Optional[Any] = None,
        n_jobs: int = 1,
        spark_config: Optional[Dict[str, str]] = None,
        master: str = "local[*]",
        app_name: str = "TSLib-GenericParallel",
    ):
        if not _SPARK_AVAILABLE:
            raise ImportError(
                "PySpark is required for GenericParallelProcessor. "
                "Install with: pip install pyspark"
            )
        from .ensure import ensure_spark_session

        self.model_type = model_type.upper()
        self.n_jobs = n_jobs
        self.spark = ensure_spark_session(
            spark_session=spark,
            spark_config=spark_config,
            master=master,
            app_name=app_name,
            register_global=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_multiple(
        self,
        df: 'DataFrame',
        group_col: str,
        value_col: str,
        order: Any,
        steps: int = 10,
    ) -> 'DataFrame':
        """
        Fit a model on each group in *df* and return combined forecasts.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Long-format DataFrame with at least *group_col* and *value_col*.
        group_col : str
            Column identifying each time series.
        value_col : str
            Column containing the time series values (numeric).
        order : int or tuple
            Model order: int for AR/MA, (p,q) for ARMA, (p,d,q) for ARIMA.
        steps : int
            Forecast horizon.

        Returns
        -------
        pyspark.sql.DataFrame
            Columns: series_id, step, forecast, status
        """
        import pandas as pd

        # Rename group column to a safe internal name
        working_df = df.withColumnRenamed(group_col, '_group_id')

        # Collect schema for the output
        output_schema = StructType([
            StructField('series_id', StringType(),  True),
            StructField('step',      IntegerType(), True),
            StructField('forecast',  DoubleType(),  True),
            StructField('status',    StringType(),  True),
            StructField('engine_warnings', StringType(), True),
        ])

        # Capture closure variables
        _model_type = self.model_type
        _order      = order
        _steps      = steps
        _n_jobs     = self.n_jobs
        _value_col  = value_col

        def _apply_groups(iterator: Iterator[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            for pdf in iterator:
                yield _fit_predict_group(
                    pdf, _value_col, _model_type, _order, _steps, _n_jobs
                )

        result = working_df.groupBy('_group_id').applyInPandas(_apply_groups, schema=output_schema)
        return result

    def fit_and_collect(
        self,
        df: 'DataFrame',
        group_col: str,
        value_col: str,
        order: Any,
        steps: int = 10,
    ) -> List[Dict]:
        """
        Like fit_multiple() but collects results to the driver as a list of dicts.

        Convenient for small numbers of series or testing.
        """
        result_df = self.fit_multiple(df, group_col, value_col, order, steps)
        rows = result_df.collect()
        return [row.asDict() for row in rows]

    # ------------------------------------------------------------------
    # Sequential fallback (no Spark, for local testing)
    # ------------------------------------------------------------------

    @staticmethod
    def fit_multiple_sequential(
        series_dict: Dict[str, np.ndarray],
        model_type: str,
        order: Any,
        steps: int = 10,
        n_jobs: int = 1,
    ) -> Dict[str, Dict]:
        """
        Sequential fallback disabled by Spark-only policy.

        Parameters
        ----------
        series_dict : dict
            Mapping of series_id → 1-D numpy array.
        model_type : str
            'AR', 'MA', 'ARMA', or 'ARIMA'.
        order : int or tuple
            Model order.
        steps : int
            Forecast horizon.
        n_jobs : int
            Within-model parallelism.

        Returns
        -------
        dict
            Mapping of series_id → {'forecast': array, 'status': str}
        """
        raise RuntimeError(
            "fit_multiple_sequential está deshabilitado por política Spark-only. "
            "Usa GenericParallelProcessor.fit_multiple con un SparkSession activo."
        )
