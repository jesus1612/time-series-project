"""
Parallel ARIMA Workflow with Spark

Complete implementation of 11-step parallel ARIMA process using Spark for
distributed computation. This class is independent and follows SOLID principles.

Process Overview:
1. Determine differencing order 'd' (non-parallel)
2-3. Define parameter ranges and generate combinations
4-5. Parallel fitting with sliding windows
6. Global model selection (sliding-window ranks + AICc)
6b. Optional: re-rank top candidates by AIC on full series (closer to standard ARIMA)
7-8. Validation with fixed windows
9. Residual diagnostics
10. Local adjustment if needed
11. Final forecast
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union, Literal
from scipy import stats
import warnings

# Import Spark components
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import col, lit, struct, collect_list, udf
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType,
        DoubleType, ArrayType, FloatType, BooleanType,
    )
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkSession = None
    DataFrame = None

# Import TSLib components
from ..core.stationarity import StationarityAnalyzer
from ..core.arima import ARIMAProcess
from ..core.optimization import MLEOptimizer
from ..preprocessing.transformations import LogTransformer
from ..metrics.evaluation import InformationCriteria, ForecastMetrics, ResidualAnalyzer
from ..utils.checks import check_spark_availability
from .core import get_optimized_spark_config
from .ensure import ensure_spark_session


class ParallelARIMAWorkflow:
    """
    Complete parallel ARIMA workflow using Spark
    
    This class implements a comprehensive parallel ARIMA process (experimental)
    using Spark: sliding-window grid search, optional **full-sample AIC**
    reconciliation of the order (to align with standard one-sample ARIMA selection),
    validation, diagnostics, and a final MLE fit on the full series.
    
    The class is completely independent from ARIMAModel and can be used
    as an alternative parallel approach.
    
    Parameters:
    -----------
    spark_session : SparkSession, optional
        Existing Spark session to use. If None, will create one.
    spark_config : dict, optional
        Configuration for Spark session if creating new one
    verbose : bool
        Whether to print progress information
        
    Attributes:
    -----------
    fitted_ : bool
        Whether the workflow has been fitted
    order_ : tuple
        Final ARIMA order (p, d, q)
    parameters_ : dict
        Final model parameters (phi, theta, c, sigma2)
    results_ : dict
        Complete results from all steps
        
    Example:
    --------
    >>> from tslib.spark import ParallelARIMAWorkflow
    >>> import numpy as np
    >>> 
    >>> # Generate sample data
    >>> data = np.cumsum(np.random.randn(1000))
    >>> 
    >>> # Fit parallel workflow
    >>> workflow = ParallelARIMAWorkflow(verbose=True)
    >>> workflow.fit(data)
    >>> 
    >>> # Make predictions
    >>> forecast = workflow.predict(steps=10)
    >>> 
    >>> # Get detailed results
    >>> results = workflow.get_results()
    >>> print(workflow.summary())
    """
    
    def __init__(self, 
                 spark_session: Optional[SparkSession] = None,
                 spark_config: Optional[Dict[str, str]] = None,
                 master: Optional[str] = None,
                 app_name: Optional[str] = None,
                 verbose: bool = True,
                 grid_mode: Literal["auto_n", "acf_pacf", "manual"] = "auto_n",
                 manual_max_p: Optional[int] = None,
                 manual_max_q: Optional[int] = None,
                 d_max: int = 2,
                 acf_pacf_alpha: float = 0.05,
                 acf_pacf_max_lag: Optional[int] = None,
                 full_sample_reconcile: bool = True,
                 full_sample_reconcile_top_k: int = 15):
        """Initialize parallel ARIMA workflow"""
        
        if not check_spark_availability():
            from .ensure import DISTRIBUTED_REQUIRES_SPARK

            raise ImportError(DISTRIBUTED_REQUIRES_SPARK)
        
        self.verbose = verbose
        self.fitted_ = False
        
        # Initialize or validate Spark session (required for distributed workflow)
        if spark_session is None:
            cfg = dict(spark_config if spark_config is not None else get_optimized_spark_config(1000))
            cfg['spark.sql.execution.arrow.pyspark.enabled'] = 'false'
            self.spark = ensure_spark_session(
                spark_session=None,
                spark_config=cfg,
                master=master,
                app_name=app_name or 'TSLib-ParallelARIMAWorkflow',
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
        
        # Initialize components
        self._stationarity_analyzer = StationarityAnalyzer()
        self._log_transformer = None
        self._residual_analyzer = ResidualAnalyzer()
        
        # Results storage
        self.data_ = None
        self.working_data_ = None  # Same scale as final MLE (e.g. log); for fair benchmarks vs lineal ARIMA
        self.order_ = None
        self.parameters_ = None
        self.results_ = {
            'step1_differencing': None,
            'step2_3_combinations': None,
            'step4_5_sliding_fitting': None,
            'step6_model_selection': None,
            'step6_full_sample_reconciliation': None,
            'step7_8_validation': None,
            'step9_diagnostics': None,
            'step10_adjustment': None,
            'config': None
        }
        
        # Configuration parameters (will be set based on data size)
        self._config = {
            'max_p': None,
            'max_q': None,
            'num_sliding_windows': None,
            'num_fixed_windows': None,
            'overlap_sliding': None,
            'significance_level': 0.05
        }
        
        if grid_mode not in ("auto_n", "acf_pacf", "manual"):
            raise ValueError("grid_mode must be 'auto_n', 'acf_pacf', or 'manual'")
        self._grid_mode = grid_mode
        self._manual_max_p = manual_max_p
        self._manual_max_q = manual_max_q
        self._d_max = int(max(0, d_max))
        self._acf_pacf_alpha = float(acf_pacf_alpha)
        self._acf_pacf_max_lag = acf_pacf_max_lag
        self._full_sample_reconcile = bool(full_sample_reconcile)
        self._full_sample_reconcile_top_k = max(1, int(full_sample_reconcile_top_k))

        if self.verbose:
            print("✓ ParallelARIMAWorkflow initialized")
            print(f"  Spark session: {'Created' if self._owns_spark else 'Provided'}")
            print(f"  grid_mode: {self._grid_mode}, d_max: {self._d_max}")
            print(
                f"  full_sample_reconcile: {self._full_sample_reconcile} "
                f"(top_k={self._full_sample_reconcile_top_k})"
            )
    
    def __del__(self):
        """Cleanup Spark resources if we own the session"""
        if hasattr(self, '_owns_spark') and self._owns_spark and hasattr(self, 'spark'):
            try:
                self.spark.stop()
            except:
                pass
    
    # =========================================================================
    # STEP 1: Determine Differencing Order (Non-Parallel)
    # =========================================================================
    
    def _determine_differencing_order(self, data: np.ndarray) -> Tuple[int, bool, Optional[LogTransformer]]:
        """
        Determine differencing order 'd' using ADF/KPSS tests
        
        Also detects variance growth and applies log transformation if needed.
        
        Parameters:
        -----------
        data : np.ndarray
            Original time series data
            
        Returns:
        --------
        d : int
            Differencing order
        log_needed : bool
            Whether log transformation is needed
        transformer : LogTransformer or None
            Fitted log transformer if needed
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 1: Determining Differencing Order")
            print("="*70)
        
        # Check for variance growth
        variance_growth = self._calculate_variance_growth(data)
        log_needed = variance_growth > 1.5  # Threshold for log transform
        
        if log_needed:
            if self.verbose:
                print(f"  Variance growth detected: {variance_growth:.2f}")
                print("  Applying log transformation...")
            
            transformer = LogTransformer()
            transformer.fit(data)
            transformed_data = transformer.transform(data)
        else:
            if self.verbose:
                print(f"  Variance growth: {variance_growth:.2f} (stable)")
            transformer = None
            transformed_data = data.copy()
        
        # Explicit stationarity loop (diagram "¿estacionaria?") up to d_max
        if len(transformed_data) < 4:
            if self.verbose:
                print("  Short series: single-pass stationarity analyze()")
            stationarity_results = self._stationarity_analyzer.analyze(transformed_data)
            d = int(stationarity_results["suggested_differencing_order"])
            self.results_["step1_differencing"] = {
                "d": d,
                "log_transform_needed": log_needed,
                "variance_growth": variance_growth,
                "stationarity_results": {**stationarity_results, "iterations": []},
            }
            return d, log_needed, transformer

        if self.verbose:
            print("  Running stationarity loop (ADF/KPSS) up to d_max=%d..." % self._d_max)

        y = transformed_data.astype(float)
        iterations: List[Dict[str, Any]] = []
        current_d = 0
        last_adf: Optional[Dict[str, Any]] = None
        last_kpss: Optional[Dict[str, Any]] = None
        stationary_reached = False
        insufficient_after_diff = False

        while True:
            last_adf = self._stationarity_analyzer.adf_test.test(y)
            last_kpss = self._stationarity_analyzer.kpss_test.test(y)
            is_stat = self._stationarity_analyzer._determine_stationarity(
                last_adf, last_kpss
            )
            iterations.append({
                "d": current_d,
                "n_obs": int(len(y)),
                "adf_stationary": bool(last_adf["is_stationary"]),
                "kpss_stationary": bool(last_kpss["is_stationary"]),
                "adf_statistic": float(last_adf["test_statistic"]),
                "kpss_statistic": float(last_kpss["test_statistic"]),
            })
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

        if self.verbose:
            print(f"  ADF test stationary: {last_adf['is_stationary']}")
            print(f"  KPSS test stationary: {last_kpss['is_stationary']}")
            print(f"  ✓ Differencing iterations: {len(iterations)}")
            print(f"  ✓ Suggested differencing order: d = {d}")

        self.results_["step1_differencing"] = {
            "d": d,
            "log_transform_needed": log_needed,
            "variance_growth": variance_growth,
            "stationarity_results": stationarity_results,
        }

        return d, log_needed, transformer
    
    def _calculate_variance_growth(self, data: np.ndarray) -> float:
        """
        Calculate variance growth to detect heteroscedasticity
        
        Splits data into first and second half, compares variances.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        growth_ratio : float
            Ratio of second half variance to first half variance
        """
        n = len(data)
        mid = n // 2
        
        first_half = data[:mid]
        second_half = data[mid:]
        
        var_first = np.var(first_half)
        var_second = np.var(second_half)
        
        # Avoid division by zero
        if var_first < 1e-10:
            return 1.0
        
        return var_second / var_first
    
    # =========================================================================
    # STEPS 2-3: Define Parameter Ranges and Generate Combinations
    # =========================================================================
    
    def _determine_parameter_ranges(self, n_obs: int) -> Dict[str, int]:
        """
        Determine parameter ranges based on dataset size
        
        Automatic configuration without depending on system memory:
        - Small (<500): max_p=3, max_q=3, 3-5 sliding windows
        - Medium (500-2000): max_p=5, max_q=5, 5-10 sliding windows
        - Large (>2000): max_p=5, max_q=5, 10-20 sliding windows
        
        Parameters:
        -----------
        n_obs : int
            Number of observations in dataset
            
        Returns:
        --------
        config : dict
            Configuration parameters
        """
        if n_obs < 500:
            # Small dataset
            config = {
                'max_p': 3,
                'max_q': 3,
                'num_sliding_windows': max(3, min(5, n_obs // 100)),
                'num_fixed_windows': max(3, n_obs // 150),
                'overlap_sliding': 0.3
            }
        elif n_obs < 2000:
            # Medium dataset
            config = {
                'max_p': 5,
                'max_q': 5,
                'num_sliding_windows': max(5, min(10, n_obs // 150)),
                'num_fixed_windows': max(4, n_obs // 200),
                'overlap_sliding': 0.2
            }
        else:
            # Large dataset
            config = {
                'max_p': 5,
                'max_q': 5,
                'num_sliding_windows': max(10, min(20, n_obs // 150)),
                'num_fixed_windows': max(5, n_obs // 250),
                'overlap_sliding': 0.15
            }
        
        if self.verbose:
            print("\n" + "="*70)
            print("STEPS 2-3: Parameter Configuration")
            print("="*70)
            print(f"  Dataset size: {n_obs} observations")
            print(f"  Category: {'Small' if n_obs < 500 else 'Medium' if n_obs < 2000 else 'Large'}")
            print(f"  Max p: {config['max_p']}")
            print(f"  Max q: {config['max_q']}")
            print(f"  Sliding windows: {config['num_sliding_windows']}")
            print(f"  Fixed windows: {config['num_fixed_windows']}")
            print(f"  Overlap: {config['overlap_sliding']}")
        
        # Store configuration
        self._config.update(config)
        self.results_['config'] = config
        
        return config
    
    def _apply_grid_mode_to_config(
        self,
        config: Dict[str, Any],
        working_data: np.ndarray,
        d: int,
        n_obs: int,
    ) -> Dict[str, Any]:
        """
        Adjust max_p / max_q from grid_mode: auto_n (unchanged), acf_pacf, or manual.
        """
        from ..core.arima_order_suggestion import suggest_p_q_orders

        if self._grid_mode == "manual":
            if self._manual_max_p is None or self._manual_max_q is None:
                raise ValueError(
                    "grid_mode='manual' requires manual_max_p and manual_max_q."
                )
            out = {
                **config,
                "max_p": int(self._manual_max_p),
                "max_q": int(self._manual_max_q),
            }
            if self.verbose:
                print(f"  Grid mode manual: max_p={out['max_p']}, max_q={out['max_q']}")
            return out

        if self._grid_mode == "acf_pacf":
            cap_p = int(config["max_p"])
            cap_q = int(config["max_q"])
            mp, mq, meta = suggest_p_q_orders(
                working_data,
                d,
                max_lag=self._acf_pacf_max_lag,
                alpha=self._acf_pacf_alpha,
                max_p_bound=cap_p,
                max_q_bound=cap_q,
            )
            self.results_["acf_pacf_identification"] = meta
            out = {**config, "max_p": mp, "max_q": mq}
            if self.verbose:
                print(
                    f"  Grid mode acf_pacf: max_p={mp}, max_q={mq} "
                    f"(auto_n caps {cap_p}, {cap_q})"
                )
            return out

        return config
    
    def _generate_parameter_combinations(self, d: int, max_p: int, max_q: int) -> List[Tuple[int, int, int]]:
        """
        Generate all (p, d, q) combinations to test
        
        Parameters:
        -----------
        d : int
            Fixed differencing order
        max_p : int
            Maximum AR order
        max_q : int
            Maximum MA order
            
        Returns:
        --------
        combinations : list of tuples
            List of (p, d, q) combinations
        """
        combinations = []
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                # Skip (0, d, 0) - must have at least AR or MA component
                if p == 0 and q == 0:
                    continue
                
                combinations.append((p, d, q))
        
        if self.verbose:
            print(f"  ✓ Generated {len(combinations)} parameter combinations")
        
        # Store results
        self.results_['step2_3_combinations'] = {
            'd': d,
            'max_p': max_p,
            'max_q': max_q,
            'combinations': combinations,
            'num_combinations': len(combinations)
        }
        
        return combinations
    
    # =========================================================================
    # STEP 4: Create Sliding Windows
    # =========================================================================
    
    def _create_sliding_windows(self, 
                               data: np.ndarray, 
                               num_windows: int, 
                               overlap: float) -> List[Dict[str, Any]]:
        """
        Create sliding windows for parallel fitting
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        num_windows : int
            Number of sliding windows to create
        overlap : float
            Overlap between windows (0.0 to 1.0)
            
        Returns:
        --------
        windows : list of dicts
            Each dict contains: window_id, data, start_idx, end_idx
        """
        n = len(data)
        if n < 1:
            return []

        num_windows = max(1, int(num_windows))

        # Calculate window size (with overlap); never exceed series length
        denom = num_windows * (1 - overlap) + overlap
        window_size = int(n / denom) if denom > 0 else n
        window_size = max(1, min(window_size, n))

        # Minimum window scales with n so short series still get >= 1 window
        # (fixed floor of 50 caused n < 50 -> no windows -> empty Spark DataFrame)
        min_window_size = min(n, max(10, n // max(num_windows + 5, 1)))
        window_size = max(window_size, min_window_size)
        window_size = min(window_size, n)

        # Calculate step size
        step_size = int(window_size * (1 - overlap))
        step_size = max(step_size, 1)

        windows = []
        window_id = 0

        for start_idx in range(0, n - window_size + 1, step_size):
            end_idx = start_idx + window_size

            if end_idx > n:
                end_idx = n

            windows.append({
                'window_id': window_id,
                'data': data[start_idx:end_idx],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': end_idx - start_idx
            })

            window_id += 1

            if window_id >= num_windows:
                break

        # Guarantee at least one window (e.g. very short series or tight geometry)
        if not windows:
            arr = np.asarray(data, dtype=float)
            windows.append({
                'window_id': 0,
                'data': arr.copy(),
                'start_idx': 0,
                'end_idx': n,
                'size': n,
            })

        if self.verbose:
            print("\n" + "="*70)
            print("STEP 4: Creating Sliding Windows")
            print("="*70)
            print(f"  Number of windows: {len(windows)}")
            print(f"  Window size: {window_size}")
            print(f"  Overlap: {overlap*100:.0f}%")
            print(f"  Step size: {step_size}")
        
        return windows
    
    # =========================================================================
    # STEP 5: Parallel Fitting with Sliding Windows (Spark)
    # =========================================================================
    
    def _fit_single_model(self, 
                         data: np.ndarray, 
                         p: int, 
                         d: int, 
                         q: int) -> Dict[str, Any]:
        """
        Fit a single ARIMA model to data
        
        This is a helper function used by Spark UDF.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        p, d, q : int
            ARIMA orders
            
        Returns:
        --------
        results : dict
            Fitting results including AICc, parameters, success flag
        """
        try:
            # Check minimum data requirement
            min_obs = max(p + d + q + 10, 30)
            if len(data) < min_obs:
                return {
                    'success': False,
                    'error': f'Insufficient data: {len(data)} < {min_obs}',
                    'p': p, 'd': d, 'q': q,
                    'aicc': np.inf
                }
            
            # Fit ARIMA model
            model = ARIMAProcess(
                ar_order=p,
                diff_order=d,
                ma_order=q,
                trend='c',
                n_jobs=1  # No nested parallelization
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(data)
            
            # Extract parameters
            params = model._fitted_params
            
            # Calculate AICc
            n_params = p + q + 1  # AR + MA + variance
            n_obs = len(data) - d  # Effective observations after differencing
            
            aicc = InformationCriteria.aicc(
                log_likelihood=params['log_likelihood'],
                n_params=n_params,
                n_obs=n_obs
            )
            
            # Extract phi, theta, c
            phi = [params['parameters'].get(f'phi_{i+1}', 0.0) for i in range(p)]
            theta = [params['parameters'].get(f'theta_{i+1}', 0.0) for i in range(q)]
            c = model.constant if hasattr(model, 'constant') else 0.0
            sigma2 = params['parameters'].get('sigma2', 0.0)
            
            return {
                'success': True,
                'p': p, 'd': d, 'q': q,
                'aicc': aicc,
                'aic': params['aic'],
                'bic': params['bic'],
                'log_likelihood': params['log_likelihood'],
                'phi': phi,
                'theta': theta,
                'c': c,
                'sigma2': sigma2,
                'n_obs': n_obs
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'p': p, 'd': d, 'q': q,
                'aicc': np.inf
            }
    
    def _fit_models_parallel_sliding(self, 
                                     windows: List[Dict[str, Any]], 
                                     combinations: List[Tuple[int, int, int]]) -> pd.DataFrame:
        """
        Fit all model combinations across all windows using Spark
        
        This is the main parallelization step.
        
        Parameters:
        -----------
        windows : list of dicts
            Sliding windows
        combinations : list of tuples
            (p, d, q) combinations to test
            
        Returns:
        --------
        results_df : pandas DataFrame
            Results for all windows and combinations
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 5: Parallel Fitting with Spark")
            print("="*70)
            print(f"  Windows: {len(windows)}")
            print(f"  Combinations: {len(combinations)}")
            print(f"  Total fits: {len(windows) * len(combinations)}")
            print("  Starting parallel computation...")
        
        # Create pandas DataFrame with tasks (better serialization with applyInPandas)
        tasks_list = []
        for window in windows:
            for p, d, q in combinations:
                tasks_list.append({
                    'window_id': window['window_id'],
                    'window_data': window['data'].tolist(),  # Convert to list for serialization
                    'window_size': len(window['data']),
                    'p': p,
                    'd': d,
                    'q': q
                })

        if not tasks_list:
            raise ValueError(
                "Parallel ARIMA sliding fit has no tasks (empty windows or no (p,d,q) combinations). "
                f"windows={len(windows)}, combinations={len(combinations)}."
            )

        tasks_pandas = pd.DataFrame(tasks_list)
        
        # Convert to Spark DataFrame with explicit schema
        from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType
        
        schema = StructType([
            StructField("window_id", IntegerType(), True),
            StructField("window_data", ArrayType(DoubleType()), True),
            StructField("window_size", IntegerType(), True),
            StructField("p", IntegerType(), True),
            StructField("d", IntegerType(), True),
            StructField("q", IntegerType(), True)
        ])
        
        tasks_df = self.spark.createDataFrame(tasks_pandas, schema=schema)
        
        # Repartition for better parallelization (one partition per CPU core)
        # This ensures all tasks are distributed across workers
        num_partitions = max(1, min(len(tasks_pandas), 
                                    self.spark.sparkContext.defaultParallelism))
        tasks_df = tasks_df.repartition(num_partitions)
        
        if self.verbose:
            sc = self.spark.sparkContext
            actual_partitions = tasks_df.rdd.getNumPartitions()
            print(f"  📊 Parallelism Info:")
            print(f"     Cores disponibles: {sc.defaultParallelism}")
            print(f"     Partitions creadas: {actual_partitions}")
            print(f"     Total tareas: {len(tasks_pandas)}")
            print(f"     Tareas por partition: ~{len(tasks_pandas) / actual_partitions:.1f}")
        
        # Define output schema
        from pyspark.sql.types import BooleanType, StringType
        
        output_schema = StructType([
            StructField("window_id", IntegerType(), True),
            StructField("p", IntegerType(), True),
            StructField("d", IntegerType(), True),
            StructField("q", IntegerType(), True),
            StructField("success", BooleanType(), True),
            StructField("aicc", DoubleType(), True),
            StructField("aic", DoubleType(), True),
            StructField("bic", DoubleType(), True),
            StructField("log_likelihood", DoubleType(), True),
            StructField("phi", ArrayType(DoubleType()), True),
            StructField("theta", ArrayType(DoubleType()), True),
            StructField("c", DoubleType(), True),
            StructField("sigma2", DoubleType(), True),
            StructField("error", StringType(), True)
        ])
        
        # Apply function using mapInPandas (processes each row individually)
        # This avoids grouping and processes all combinations in parallel
        def fit_models_map(iterator):
            """Process each row (window, combination pair) individually"""
            import pandas as pd
            import numpy as np
            import warnings
            
            # Import once per partition
            try:
                from tslib.core.arima import ARIMAProcess
                from tslib.metrics.evaluation import InformationCriteria
            except ImportError:
                import sys
                import os
                # Try to find tslib in path
                for path in sys.path:
                    tslib_path = os.path.join(path, 'tslib')
                    if os.path.exists(tslib_path):
                        if path not in sys.path:
                            sys.path.insert(0, path)
                        break
                from tslib.core.arima import ARIMAProcess
                from tslib.metrics.evaluation import InformationCriteria
            
            results = []
            
            for pdf in iterator:
                for _, row in pdf.iterrows():
                    try:
                        window_id = int(row['window_id'])
                        window_data = np.array(row['window_data'], dtype=float)
                        p, d, q = int(row['p']), int(row['d']), int(row['q'])
                        
                        # Check minimum data requirement
                        min_obs = max(p + d + q + 10, 30)
                        if len(window_data) < min_obs:
                            results.append({
                                'window_id': window_id,
                                'p': p, 'd': d, 'q': q,
                                'success': False,
                                'aicc': np.inf,
                                'aic': np.inf,
                                'bic': np.inf,
                                'log_likelihood': np.nan,
                                'phi': [],
                                'theta': [],
                                'c': 0.0,
                                'sigma2': 0.0,
                                'error': f'Insufficient data: {len(window_data)} < {min_obs}'
                            })
                            continue
                        
                        # Fit ARIMA model
                        model = ARIMAProcess(
                            ar_order=p,
                            diff_order=d,
                            ma_order=q,
                            trend='c',
                            n_jobs=1
                        )
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(window_data)
                        
                        # Extract parameters
                        params = model._fitted_params
                        
                        # Calculate AICc
                        n_params = p + q + 1
                        n_obs = len(window_data) - d
                        
                        aicc = InformationCriteria.aicc(
                            log_likelihood=params['log_likelihood'],
                            n_params=n_params,
                            n_obs=n_obs
                        )
                        
                        # Extract phi, theta, c
                        phi = [float(params['parameters'].get(f'phi_{i+1}', 0.0)) for i in range(p)]
                        theta = [float(params['parameters'].get(f'theta_{i+1}', 0.0)) for i in range(q)]
                        c = float(model.constant if hasattr(model, 'constant') else 0.0)
                        sigma2 = float(params['parameters'].get('sigma2', 0.0))
                        
                        results.append({
                            'window_id': window_id,
                            'p': p, 'd': d, 'q': q,
                            'success': True,
                            'aicc': float(aicc),
                            'aic': float(params['aic']),
                            'bic': float(params['bic']),
                            'log_likelihood': float(params['log_likelihood']),
                            'phi': phi,
                            'theta': theta,
                            'c': c,
                            'sigma2': sigma2,
                            'error': ''
                        })
                        
                    except Exception as e:
                        results.append({
                            'window_id': int(row['window_id']),
                            'p': int(row['p']), 'd': int(row['d']), 'q': int(row['q']),
                            'success': False,
                            'aicc': np.inf,
                            'aic': np.inf,
                            'bic': np.inf,
                            'log_likelihood': np.nan,
                            'phi': [],
                            'theta': [],
                            'c': 0.0,
                            'sigma2': 0.0,
                            'error': str(e)[:200]
                        })
            
            if results:
                yield pd.DataFrame(results)
        if self.verbose:
            import time
            print(f"  🚀 Iniciando ejecución paralela...")
            start_time = time.time()

        # Apply function using mapInPandas (processes each partition)
        results_spark = tasks_df.mapInPandas(fit_models_map, schema=output_schema)
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"  ⏱️  Ejecución completada en {elapsed:.2f}s")
            print(f"     Throughput: {len(tasks_pandas) / elapsed:.1f} tareas/segundo")
        # Convert to pandas (Arrow is already disabled for NumPy compatibility)
        try:
            results_df = results_spark.toPandas()
        except Exception as e:
            # Fallback: collect rows manually
            if self.verbose:
                print(f"  Warning: Pandas conversion failed, using fallback method: {str(e)[:50]}")
            results_rows = results_spark.collect()
            results_list = []
            for row in results_rows:
                results_list.append({
                    'window_id': row.window_id,
                    'p': row.p,
                    'd': row.d,
                    'q': row.q,
                    'success': row.success,
                    'aicc': float(row.aicc) if row.aicc is not None else np.inf,
                    'aic': float(row.aic) if row.aic is not None else np.inf,
                    'bic': float(row.bic) if row.bic is not None else np.inf,
                    'log_likelihood': float(row.log_likelihood) if row.log_likelihood is not None else np.nan,
                    'phi': list(row.phi) if row.phi else [],
                    'theta': list(row.theta) if row.theta else [],
                    'c': float(row.c) if row.c is not None else 0.0,
                    'sigma2': float(row.sigma2) if row.sigma2 is not None else 0.0,
                    'error': row.error if row.error else ''
                })
            results_df = pd.DataFrame(results_list)
        
        # Count successes
        num_success = results_df['success'].sum()
        num_total = len(results_df)
        
        if self.verbose:
            print(f"  ✓ Completed: {num_success}/{num_total} models fitted successfully")
            print(f"  Success rate: {100*num_success/num_total:.1f}%")
        
        # Store results
        self.results_['step4_5_sliding_fitting'] = {
            'results_df': results_df,
            'num_windows': len(windows),
            'num_combinations': len(combinations),
            'num_success': int(num_success),
            'num_total': int(num_total)
        }
        
        return results_df
    
    # =========================================================================
    # STEP 6: Global Model Selection
    # =========================================================================
    
    def _select_global_model(self, results_df: pd.DataFrame) -> Tuple[Tuple[int, int, int], float]:
        """
        Select best global model using ranking system
        
        Ranks models by AICc in each window, then aggregates rankings
        across windows with penalty for inconsistency.
        
        Parameters:
        -----------
        results_df : pandas DataFrame
            Results from parallel fitting
            
        Returns:
        --------
        best_order : tuple
            Best (p, d, q) order
        confidence_score : float
            Confidence score (0-1)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 6: Global Model Selection")
            print("="*70)
        
        # Filter successful fits
        success_df = results_df[results_df['success'] == True].copy()
        
        if len(success_df) == 0:
            raise ValueError("No successful model fits found")
        
        # Rank models by AICc within each window
        success_df['rank'] = success_df.groupby('window_id')['aicc'].rank(method='min')
        
        # Calculate aggregate statistics for each (p, d, q) combination
        model_scores = success_df.groupby(['p', 'd', 'q']).agg({
            'rank': ['mean', 'std', 'count'],
            'aicc': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        model_scores.columns = ['p', 'd', 'q', 'rank_mean', 'rank_std', 'count', 'aicc_mean', 'aicc_std']
        
        # Calculate composite score
        # Lower is better: mean rank + penalty for inconsistency
        model_scores['score'] = (
            model_scores['rank_mean'] + 
            0.5 * model_scores['rank_std'].fillna(0)  # Penalty for inconsistency
        )
        
        # Find best model
        best_idx = model_scores['score'].idxmin()
        best_row = model_scores.loc[best_idx]
        
        best_order = (int(best_row['p']), int(best_row['d']), int(best_row['q']))
        
        # Calculate confidence score
        # Based on consistency across windows
        num_windows = results_df['window_id'].nunique()
        appearance_rate = best_row['count'] / num_windows
        consistency = 1.0 / (1.0 + best_row['rank_std']) if best_row['rank_std'] > 0 else 1.0
        confidence_score = 0.6 * appearance_rate + 0.4 * consistency
        
        if self.verbose:
            print(f"  ✓ Selected model: ARIMA{best_order}")
            print(f"  Average rank: {best_row['rank_mean']:.2f}")
            print(f"  Rank std: {best_row['rank_std']:.2f}")
            print(f"  Appeared in: {int(best_row['count'])}/{num_windows} windows")
            print(f"  Mean AICc: {best_row['aicc_mean']:.2f}")
            print(f"  Confidence score: {confidence_score:.2f}")
            
            # Show top 5 models
            print("\n  Top 5 models:")
            top_5 = model_scores.nsmallest(5, 'score')
            for idx, row in top_5.iterrows():
                print(f"    ARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])}): "
                      f"score={row['score']:.2f}, rank={row['rank_mean']:.2f}, "
                      f"aicc={row['aicc_mean']:.2f}")
        
        # Store results
        self.results_['step6_model_selection'] = {
            'best_order': best_order,
            'confidence_score': confidence_score,
            'model_scores': model_scores,
            'best_metrics': {
                'rank_mean': float(best_row['rank_mean']),
                'rank_std': float(best_row['rank_std']),
                'aicc_mean': float(best_row['aicc_mean']),
                'aicc_std': float(best_row['aicc_std']),
                'appearance_count': int(best_row['count']),
                'num_windows': num_windows
            }
        }
        
        return best_order, confidence_score
    
    def _reconcile_order_full_sample_aic(
        self,
        working_data: np.ndarray,
        sliding_best_order: Tuple[int, int, int],
        sliding_confidence: float,
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        Re-rank top candidates from step 6 by AIC on the full working series.

        Experimental workflows pick (p,d,q) from sliding-window AICc ranks; standard
        ARIMA practice selects the order using the same sample as the final MLE fit.
        This step evaluates the top-K orders from step 6 on ``working_data`` and
        chooses the minimum-AIC fit, aligning selection with the final estimation sample.
        """
        step6 = self.results_.get("step6_model_selection") or {}
        model_scores = step6.get("model_scores")
        if model_scores is None or len(model_scores) == 0:
            self.results_["step6_full_sample_reconciliation"] = {
                "enabled": True,
                "skipped": True,
                "reason": "no model_scores",
                "order_sliding": sliding_best_order,
                "order_full_sample": sliding_best_order,
            }
            return sliding_best_order, sliding_confidence

        data = np.asarray(working_data, dtype=float).ravel()
        k = min(self._full_sample_reconcile_top_k, len(model_scores))
        top = model_scores.nsmallest(k, "score")

        seen = set()
        candidates: List[Tuple[int, int, int]] = []
        for _, row in top.iterrows():
            t = (int(row["p"]), int(row["d"]), int(row["q"]))
            if t not in seen:
                seen.add(t)
                candidates.append(t)

        if sliding_best_order not in seen:
            candidates.insert(0, sliding_best_order)

        aic_by_order: Dict[Tuple[int, int, int], float] = {}
        bic_by_order: Dict[Tuple[int, int, int], float] = {}
        failed: List[Tuple[int, int, int]] = []

        for t in candidates:
            p, d, q = t
            try:
                model = ARIMAProcess(
                    ar_order=p,
                    diff_order=d,
                    ma_order=q,
                    trend="c",
                    n_jobs=1,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(data)
                fp = model._fitted_params
                aic = fp.get("aic")
                bic = fp.get("bic")
                if aic is None or not np.isfinite(aic):
                    failed.append(t)
                    continue
                aic_by_order[t] = float(aic)
                if bic is not None and np.isfinite(bic):
                    bic_by_order[t] = float(bic)
            except Exception:
                failed.append(t)
                continue

        if not aic_by_order:
            if self.verbose:
                print(
                    "\n  ⚠ Full-sample AIC reconciliation: all candidate fits failed; "
                    "keeping sliding-window order."
                )
            self.results_["step6_full_sample_reconciliation"] = {
                "enabled": True,
                "skipped": False,
                "failed_all": True,
                "candidates_tried": candidates,
                "failed": failed,
                "order_sliding": sliding_best_order,
                "order_full_sample": sliding_best_order,
                "aic_by_order": {},
            }
            return sliding_best_order, sliding_confidence

        best_t = min(aic_by_order.keys(), key=lambda x: aic_by_order[x])
        best_aic = aic_by_order[best_t]
        changed = best_t != sliding_best_order

        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 6b: Full-sample order reconciliation (min AIC on full series)")
            print("=" * 70)
            print(f"  Candidates from sliding ranks (top {k}): {len(candidates)} unique orders")
            for t in sorted(aic_by_order.keys(), key=lambda x: aic_by_order[x]):
                mark = " ← selected" if t == best_t else ""
                print(f"    ARIMA{t}: AIC={aic_by_order[t]:.2f}{mark}")
            if changed:
                print(
                    f"  Order updated: ARIMA{sliding_best_order} (sliding) → ARIMA{best_t} (full-sample AIC)"
                )
            else:
                print(f"  Order unchanged: ARIMA{best_t} (best full-sample AIC matches sliding pick)")

        self.results_["step6_full_sample_reconciliation"] = {
            "enabled": True,
            "skipped": False,
            "top_k": k,
            "candidates_evaluated": list(aic_by_order.keys()),
            "candidates_tried": candidates,
            "failed_fits": failed,
            "aic_by_order": {str(k): v for k, v in aic_by_order.items()},
            "bic_by_order": {str(k): v for k, v in bic_by_order.items()},
            "order_sliding": sliding_best_order,
            "order_full_sample": best_t,
            "best_full_sample_aic": best_aic,
            "changed_from_sliding": changed,
        }

        return best_t, sliding_confidence
    
    # =========================================================================
    # STEP 7: Create Fixed Windows for Validation
    # =========================================================================
    
    def _create_fixed_windows(self, 
                             data: np.ndarray, 
                             num_windows: int,
                             min_train_pct: float = 0.7) -> List[Dict[str, Any]]:
        """
        Create fixed (non-overlapping) windows for validation
        
        Each window is split into train/test sets for backtesting.
        Validates that 70% of data > max(p,q) for current order.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        num_windows : int
            Number of fixed windows
        min_train_pct : float
            Minimum percentage for training set
            
        Returns:
        --------
        windows : list of dicts
            Each dict contains: window_id, train_data, test_data, indices
        """
        n = len(data)
        window_size = n // num_windows
        
        # Ensure minimum window size based on current order
        p, d, q = self.order_ if self.order_ else (1, 0, 1)
        min_required = max(p, q) + 10
        min_window_size = int(min_required / min_train_pct) + 10
        
        window_size = max(window_size, min_window_size)
        
        windows = []
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, n)
            
            if end_idx - start_idx < min_window_size:
                break  # Skip if window too small
            
            window_data = data[start_idx:end_idx]
            
            # Split into train/test
            train_size = int(len(window_data) * min_train_pct)
            train_data = window_data[:train_size]
            test_data = window_data[train_size:]
            
            # Validate train size
            if len(train_data) < min_required:
                continue  # Skip if insufficient training data
            
            windows.append({
                'window_id': i,
                'train_data': train_data,
                'test_data': test_data,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'train_size': len(train_data),
                'test_size': len(test_data)
            })
        
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 7: Creating Fixed Windows for Validation")
            print("="*70)
            print(f"  Number of windows: {len(windows)}")
            print(f"  Window size: ~{window_size}")
            print(f"  Train/Test split: {min_train_pct*100:.0f}% / {(1-min_train_pct)*100:.0f}%")
        
        return windows
    
    # =========================================================================
    # STEP 8: Backtesting with Fixed Windows
    # =========================================================================
    
    def _backtest_fixed_windows(self,
                                windows: List[Dict[str, Any]],
                                order: Tuple[int, int, int]) -> pd.DataFrame:
        """
        Perform backtesting on fixed windows (Spark mapInPandas per window).

        Refits model on each window's training set and evaluates on test set.
        Calculates MAE, RMSE, MAPE metrics.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 8: Backtesting with Fixed Windows (Spark)")
            print("="*70)
            print(f"  Model: ARIMA{order}")
            print(f"  Windows: {len(windows)}")

        p, d, q = order
        if not windows:
            results_df = pd.DataFrame(
                columns=[
                    "window_id", "p", "d", "q", "success", "mae", "rmse", "mape",
                    "phi", "theta", "c", "sigma2", "train_size", "test_size", "error",
                ]
            )
            self.results_["step7_8_validation"] = {
                "results_df": results_df,
                "num_windows": 0,
                "num_success": 0,
                "metrics": {"avg_mae": np.nan, "avg_rmse": np.nan, "avg_mape": np.nan},
            }
            return results_df

        tasks_list: List[Dict[str, Any]] = []
        for window in windows:
            tasks_list.append(
                {
                    "window_id": int(window["window_id"]),
                    "train_data": np.asarray(window["train_data"], dtype=float).tolist(),
                    "test_data": np.asarray(window["test_data"], dtype=float).tolist(),
                    "train_size": int(window["train_size"]),
                    "test_size": int(window["test_size"]),
                    "p": p,
                    "d": d,
                    "q": q,
                }
            )

        tasks_pandas = pd.DataFrame(tasks_list)
        from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, ArrayType, StringType, BooleanType

        schema_in = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("train_data", ArrayType(DoubleType()), True),
                StructField("test_data", ArrayType(DoubleType()), True),
                StructField("train_size", IntegerType(), True),
                StructField("test_size", IntegerType(), True),
                StructField("p", IntegerType(), True),
                StructField("d", IntegerType(), True),
                StructField("q", IntegerType(), True),
            ]
        )
        tasks_df = self.spark.createDataFrame(tasks_pandas, schema=schema_in)
        num_partitions = max(
            1, min(len(tasks_pandas), self.spark.sparkContext.defaultParallelism)
        )
        tasks_df = tasks_df.repartition(num_partitions)

        output_schema = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("p", IntegerType(), True),
                StructField("d", IntegerType(), True),
                StructField("q", IntegerType(), True),
                StructField("success", BooleanType(), True),
                StructField("mae", DoubleType(), True),
                StructField("rmse", DoubleType(), True),
                StructField("mape", DoubleType(), True),
                StructField("phi", ArrayType(DoubleType()), True),
                StructField("theta", ArrayType(DoubleType()), True),
                StructField("c", DoubleType(), True),
                StructField("sigma2", DoubleType(), True),
                StructField("train_size", IntegerType(), True),
                StructField("test_size", IntegerType(), True),
                StructField("error", StringType(), True),
            ]
        )

        def backtest_map(iterator):
            import pandas as pd
            import numpy as np
            import warnings

            try:
                from tslib.core.arima import ARIMAProcess
                from tslib.metrics.evaluation import ForecastMetrics
            except ImportError:
                import os
                import sys

                for path in sys.path:
                    if os.path.exists(os.path.join(path, "tslib")):
                        if path not in sys.path:
                            sys.path.insert(0, path)
                        break
                from tslib.core.arima import ARIMAProcess
                from tslib.metrics.evaluation import ForecastMetrics

            rows_out: List[Dict[str, Any]] = []
            for pdf in iterator:
                for _, row in pdf.iterrows():
                    wid = int(row["window_id"])
                    pp, dd, qq = int(row["p"]), int(row["d"]), int(row["q"])
                    train_data = np.asarray(row["train_data"], dtype=float)
                    test_data = np.asarray(row["test_data"], dtype=float)
                    ts_tr = int(row["train_size"])
                    ts_te = int(row["test_size"])
                    try:
                        model = ARIMAProcess(
                            ar_order=pp,
                            diff_order=dd,
                            ma_order=qq,
                            trend="c",
                            n_jobs=1,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(train_data)
                        params = model._fitted_params
                        phi = [float(params["parameters"].get(f"phi_{i+1}", 0.0)) for i in range(pp)]
                        theta = [float(params["parameters"].get(f"theta_{i+1}", 0.0)) for i in range(qq)]
                        c = float(model.constant if hasattr(model, "constant") else 0.0)
                        sigma2 = float(params["parameters"].get("sigma2", 0.0))
                        test_size = len(test_data)
                        predictions = model.predict(steps=test_size)
                        mae = float(ForecastMetrics.mae(test_data, predictions))
                        rmse = float(ForecastMetrics.rmse(test_data, predictions))
                        mape = float(ForecastMetrics.mape(test_data, predictions))
                        rows_out.append(
                            {
                                "window_id": wid,
                                "p": pp,
                                "d": dd,
                                "q": qq,
                                "success": True,
                                "mae": mae,
                                "rmse": rmse,
                                "mape": mape,
                                "phi": phi,
                                "theta": theta,
                                "c": c,
                                "sigma2": sigma2,
                                "train_size": ts_tr,
                                "test_size": ts_te,
                                "error": "",
                            }
                        )
                    except Exception as e:
                        rows_out.append(
                            {
                                "window_id": wid,
                                "p": pp,
                                "d": dd,
                                "q": qq,
                                "success": False,
                                "mae": np.nan,
                                "rmse": np.nan,
                                "mape": np.nan,
                                "phi": [],
                                "theta": [],
                                "c": np.nan,
                                "sigma2": np.nan,
                                "train_size": ts_tr,
                                "test_size": ts_te,
                                "error": str(e)[:500],
                            }
                        )
            if rows_out:
                yield pd.DataFrame(rows_out)

        if self.verbose:
            import time

            print("  🚀 Backtesting en Spark (mapInPandas)...")
            t0 = time.time()

        results_spark = tasks_df.mapInPandas(backtest_map, schema=output_schema)
        results_df = results_spark.toPandas()

        if self.verbose:
            import time

            print(f"  ⏱️  Backtesting completado en {time.time() - t0:.2f}s")

        if "success" in results_df.columns:
            results_df["success"] = results_df["success"].astype(bool)
        results_df = results_df.sort_values("window_id").reset_index(drop=True)

        successful = results_df[results_df["success"]]
        if len(successful) > 0:
            avg_mae = successful["mae"].mean()
            avg_rmse = successful["rmse"].mean()
            avg_mape = successful["mape"].mean()
            if self.verbose:
                print(f"  ✓ Completed: {len(successful)}/{len(results_df)} windows")
                print(f"  Average MAE: {avg_mae:.4f}")
                print(f"  Average RMSE: {avg_rmse:.4f}")
                print(f"  Average MAPE: {avg_mape:.2f}%")
        else:
            if self.verbose:
                print("  ✗ All windows failed")

        self.results_["step7_8_validation"] = {
            "results_df": results_df,
            "num_windows": len(windows),
            "num_success": int(results_df["success"].sum()) if "success" in results_df.columns else 0,
            "metrics": {
                "avg_mae": float(successful["mae"].mean()) if len(successful) > 0 else np.nan,
                "avg_rmse": float(successful["rmse"].mean()) if len(successful) > 0 else np.nan,
                "avg_mape": float(successful["mape"].mean()) if len(successful) > 0 else np.nan,
            },
        }

        return results_df
    
    # =========================================================================
    # STEP 9: Residual Diagnostics
    # =========================================================================
    
    def _diagnose_residuals(
        self,
        fixed_windows: List[Dict[str, Any]],
        order: Tuple[int, int, int],
    ) -> pd.DataFrame:
        """
        Residual diagnostics per fixed window (Spark mapInPandas).

        Analyzes residuals, ACF of residuals, and Ljung-Box test.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 9: Residual Diagnostics (Spark)")
            print("="*70)

        p, d, q = order
        sig_level = float(self._config["significance_level"])

        if not fixed_windows:
            diagnostics_df = pd.DataFrame(
                columns=[
                    "window_id", "p", "d", "q",
                    "ljung_box_statistic", "ljung_box_p_value", "ljung_box_pass",
                    "acf_significant_peaks", "acf_pass", "overall_pass",
                    "residuals_mean", "residuals_std", "success", "error",
                ]
            )
            self.results_["step9_diagnostics"] = {
                "diagnostics_df": diagnostics_df,
                "num_windows": 0,
                "pass_rates": {"overall": 0.0, "ljung_box": 0.0, "acf": 0.0},
            }
            return diagnostics_df

        tasks_list: List[Dict[str, Any]] = []
        for window in fixed_windows:
            tasks_list.append(
                {
                    "window_id": int(window["window_id"]),
                    "train_data": np.asarray(window["train_data"], dtype=float).tolist(),
                    "p": p,
                    "d": d,
                    "q": q,
                }
            )

        tasks_pandas = pd.DataFrame(tasks_list)
        from pyspark.sql.types import (
            StructType,
            StructField,
            IntegerType,
            DoubleType,
            StringType,
            BooleanType,
            ArrayType,
        )

        schema_in = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("train_data", ArrayType(DoubleType()), True),
                StructField("p", IntegerType(), True),
                StructField("d", IntegerType(), True),
                StructField("q", IntegerType(), True),
            ]
        )
        tasks_df = self.spark.createDataFrame(tasks_pandas, schema=schema_in)
        npart = max(1, min(len(tasks_pandas), self.spark.sparkContext.defaultParallelism))
        tasks_df = tasks_df.repartition(npart)

        output_schema = StructType(
            [
                StructField("window_id", IntegerType(), True),
                StructField("p", IntegerType(), True),
                StructField("d", IntegerType(), True),
                StructField("q", IntegerType(), True),
                StructField("ljung_box_statistic", DoubleType(), True),
                StructField("ljung_box_p_value", DoubleType(), True),
                StructField("ljung_box_pass", BooleanType(), True),
                StructField("acf_significant_peaks", IntegerType(), True),
                StructField("acf_pass", BooleanType(), True),
                StructField("overall_pass", BooleanType(), True),
                StructField("residuals_mean", DoubleType(), True),
                StructField("residuals_std", DoubleType(), True),
                StructField("success", BooleanType(), True),
                StructField("error", StringType(), True),
            ]
        )

        def diagnose_map(iterator):
            import warnings

            import numpy as np
            import pandas as pd

            try:
                from tslib.core.arima import ARIMAProcess
                from tslib.metrics.evaluation import ResidualAnalyzer
            except ImportError:
                import os
                import sys

                for path in sys.path:
                    if os.path.exists(os.path.join(path, "tslib")):
                        if path not in sys.path:
                            sys.path.insert(0, path)
                        break
                from tslib.core.arima import ARIMAProcess
                from tslib.metrics.evaluation import ResidualAnalyzer

            rows_out: List[Dict[str, Any]] = []
            for pdf in iterator:
                for _, row in pdf.iterrows():
                    wid = int(row["window_id"])
                    pp, dd, qq = int(row["p"]), int(row["d"]), int(row["q"])
                    train_data = np.asarray(row["train_data"], dtype=float)
                    try:
                        model = ARIMAProcess(
                            ar_order=pp,
                            diff_order=dd,
                            ma_order=qq,
                            trend="c",
                            n_jobs=1,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(train_data)
                        residuals = model.get_residuals()
                        fitted_values = model.get_fitted_values()
                        analyzer = ResidualAnalyzer()
                        residual_analysis = analyzer.analyze(residuals, fitted_values)
                        acf_test = residual_analysis["autocorrelation_tests"]
                        ljung_box = residual_analysis["ljung_box_test"]
                        ljung_box_pass = float(ljung_box["p_value"]) > sig_level
                        acf_autocorrs = acf_test["autocorrelations"]
                        n = len(residuals)
                        conf_bound = 1.96 / np.sqrt(max(n, 1))
                        significant_acf_peaks = sum(
                            1 for ac in acf_autocorrs if abs(ac) > conf_bound
                        )
                        acf_pass = significant_acf_peaks <= 2
                        rows_out.append(
                            {
                                "window_id": wid,
                                "p": pp,
                                "d": dd,
                                "q": qq,
                                "ljung_box_statistic": float(ljung_box["statistic"]),
                                "ljung_box_p_value": float(ljung_box["p_value"]),
                                "ljung_box_pass": bool(ljung_box_pass),
                                "acf_significant_peaks": int(significant_acf_peaks),
                                "acf_pass": bool(acf_pass),
                                "overall_pass": bool(ljung_box_pass and acf_pass),
                                "residuals_mean": float(residual_analysis["basic_stats"]["mean"]),
                                "residuals_std": float(residual_analysis["basic_stats"]["std"]),
                                "success": True,
                                "error": "",
                            }
                        )
                    except Exception as e:
                        rows_out.append(
                            {
                                "window_id": wid,
                                "p": pp,
                                "d": dd,
                                "q": qq,
                                "ljung_box_statistic": np.nan,
                                "ljung_box_p_value": np.nan,
                                "ljung_box_pass": False,
                                "acf_significant_peaks": -1,
                                "acf_pass": False,
                                "overall_pass": False,
                                "residuals_mean": np.nan,
                                "residuals_std": np.nan,
                                "success": False,
                                "error": str(e)[:500],
                            }
                        )
            if rows_out:
                yield pd.DataFrame(rows_out)

        if self.verbose:
            import time

            print("  🚀 Diagnóstico de residuos en Spark (mapInPandas)...")
            t0 = time.time()

        diagnostics_df = tasks_df.mapInPandas(diagnose_map, schema=output_schema).toPandas()

        if self.verbose:
            import time

            print(f"  ⏱️  Diagnóstico completado en {time.time() - t0:.2f}s")

        for col in ("success", "ljung_box_pass", "acf_pass", "overall_pass"):
            if col in diagnostics_df.columns:
                diagnostics_df[col] = diagnostics_df[col].astype(bool)
        diagnostics_df = diagnostics_df.sort_values("window_id").reset_index(drop=True)

        overall_pass_rate = 0.0
        ljung_box_pass_rate = 0.0
        acf_pass_rate = 0.0
        successful = diagnostics_df[diagnostics_df["success"]]
        if len(successful) > 0:
            overall_pass_rate = successful["overall_pass"].sum() / len(successful)
            ljung_box_pass_rate = successful["ljung_box_pass"].sum() / len(successful)
            acf_pass_rate = successful["acf_pass"].sum() / len(successful)
            if self.verbose:
                print(f"  ✓ Completed: {len(successful)}/{len(diagnostics_df)} windows")
                print(f"  Overall pass rate: {overall_pass_rate*100:.1f}%")
                print(f"  Ljung-Box pass rate: {ljung_box_pass_rate*100:.1f}%")
                print(f"  ACF pass rate: {acf_pass_rate*100:.1f}%")

        self.results_["step9_diagnostics"] = {
            "diagnostics_df": diagnostics_df,
            "num_windows": len(fixed_windows),
            "pass_rates": {
                "overall": float(overall_pass_rate),
                "ljung_box": float(ljung_box_pass_rate),
                "acf": float(acf_pass_rate),
            },
        }

        return diagnostics_df
    
    # =========================================================================
    # STEP 10: Local Adjustment if Needed
    # =========================================================================
    
    def _check_diagnostic_failures(self, diagnostics_df: pd.DataFrame) -> Tuple[bool, List[int]]:
        """
        Check for diagnostic failures requiring local adjustment
        
        Detects if ACF has significant peaks OR Ljung-Box fails in 2+ consecutive windows.
        
        Parameters:
        -----------
        diagnostics_df : pandas DataFrame
            Diagnostic results
            
        Returns:
        --------
        needs_adjustment : bool
            Whether local adjustment is needed
        failed_windows : list
            Window IDs that failed diagnostics
        """
        # Find failed windows
        failed_windows = diagnostics_df[diagnostics_df['overall_pass'] == False]['window_id'].tolist()
        
        if len(failed_windows) < 2:
            return False, failed_windows
        
        # Check for consecutive failures
        has_consecutive = False
        for i in range(len(failed_windows) - 1):
            if failed_windows[i+1] - failed_windows[i] == 1:
                has_consecutive = True
                break
        
        return has_consecutive, failed_windows
    
    def _try_local_adjustment(self, 
                             data: np.ndarray, 
                             current_order: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Try local adjustment: (p+1,q), (p,q+1), then (p-1,q) and (p,q-1) when valid.
        
        Only adjusts if AICc improves by >10% vs the baseline model.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        current_order : tuple
            Current (p, d, q) order
            
        Returns:
        --------
        final_order : tuple
            Final (p, d, q) order (may be same as input)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 10: Local Adjustment")
            print("="*70)
            print(f"  Current model: ARIMA{current_order}")
            print("  Testing local adjustments...")
        
        p, d, q = current_order
        
        # Fit current model to get baseline AICc
        try:
            base_model = ARIMAProcess(p, d, q, trend='c', n_jobs=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_model.fit(data)
            
            base_params = base_model._fitted_params
            n_obs = len(data) - d
            n_params = p + q + 1
            base_aicc = InformationCriteria.aicc(
                base_params['log_likelihood'],
                n_params,
                n_obs
            )
        except:
            if self.verbose:
                print("  ✗ Failed to fit current model")
            return current_order
        
        if self.verbose:
            print(f"  Current AICc: {base_aicc:.2f}")
        
        # Try adjustments: +1 in each dimension, then -1 (narrative-aligned mini-grid)
        candidates = [
            (p + 1, d, q),
            (p, d, q + 1),
        ]
        if p > 0:
            candidates.append((p - 1, d, q))
        if q > 0:
            candidates.append((p, d, q - 1))

        best_order = current_order
        best_aicc = base_aicc
        
        for test_p, test_d, test_q in candidates:
            if test_p == 0 and test_q == 0:
                continue
            try:
                test_model = ARIMAProcess(test_p, test_d, test_q, trend='c', n_jobs=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_model.fit(data)
                
                test_params = test_model._fitted_params
                test_n_params = test_p + test_q + 1
                test_aicc = InformationCriteria.aicc(
                    test_params['log_likelihood'],
                    test_n_params,
                    n_obs
                )
                
                # Check if improvement is significant (>10%)
                improvement_pct = (base_aicc - test_aicc) / base_aicc * 100
                
                if self.verbose:
                    print(f"  ARIMA({test_p},{test_d},{test_q}): AICc={test_aicc:.2f}, "
                          f"improvement={improvement_pct:.1f}%")
                
                if test_aicc < best_aicc and improvement_pct > 10:
                    best_aicc = test_aicc
                    best_order = (test_p, test_d, test_q)
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ARIMA({test_p},{test_d},{test_q}): Failed - {str(e)[:50]}")
                continue
        
        if best_order != current_order:
            improvement_pct = (base_aicc - best_aicc) / base_aicc * 100
            if self.verbose:
                print(f"  ✓ Adjusted to ARIMA{best_order}")
                print(f"  AICc improvement: {improvement_pct:.1f}%")
        else:
            if self.verbose:
                print(f"  ✓ Keeping ARIMA{current_order} (no significant improvement)")
        
        # Store results
        self.results_['step10_adjustment'] = {
            'current_order': current_order,
            'final_order': best_order,
            'adjusted': best_order != current_order,
            'base_aicc': float(base_aicc),
            'final_aicc': float(best_aicc),
            'improvement_pct': float((base_aicc - best_aicc) / base_aicc * 100) if best_aicc < base_aicc else 0.0,
            'candidates_tried': [tuple(c) for c in candidates],
        }
        
        return best_order
    
    # =========================================================================
    # MAIN FIT METHOD: Orchestrates Steps 1-10
    # =========================================================================
    
    def fit(self, data: Union[np.ndarray, pd.Series, list]) -> 'ParallelARIMAWorkflow':
        """
        Fit the parallel ARIMA workflow
        
        Executes all 10 steps of the parallel process:
        1. Determine differencing order
        2-3. Generate parameter combinations
        4-5. Parallel fitting with sliding windows
        6. Global model selection
        7-8. Validation with fixed windows
        9. Residual diagnostics
        10. Local adjustment if needed
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        self : ParallelARIMAWorkflow
            Fitted workflow
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        else:
            data = np.asarray(data, dtype=float)
        
        self.data_ = data
        n = len(data)
        
        if self.verbose:
            _sc = self.spark.sparkContext
            print("\n" + "="*70)
            print("PARALLEL ARIMA WORKFLOW - STARTING")
            print("="*70)
            print(f"Data: {n} observations")
            print(f"🔧 Spark Config:")
            print(f"   Master: {_sc.master}")
            print(f"   Cores: {_sc.defaultParallelism}")
            print(f"   App: {_sc.appName}")
        
        # STEP 1: Determine differencing order
        d, log_needed, transformer = self._determine_differencing_order(data)
        self._log_transformer = transformer
        
        if log_needed and transformer:
            working_data = transformer.transform(data)
        else:
            working_data = data.copy()
        
        # STEPS 2-3: Determine parameter ranges and generate combinations
        config = self._determine_parameter_ranges(n)
        config = self._apply_grid_mode_to_config(config, working_data, d, n)
        self._config.update(config)
        self._config["grid_mode"] = self._grid_mode
        self._config["d_max"] = self._d_max
        self.results_["config"] = {
            **self._config,
        }
        combinations = self._generate_parameter_combinations(
            d, config["max_p"], config["max_q"]
        )
        
        # STEP 4: Create sliding windows
        sliding_windows = self._create_sliding_windows(
            working_data,
            config['num_sliding_windows'],
            config['overlap_sliding']
        )
        
        # STEP 5: Parallel fitting with Spark
        sliding_results = self._fit_models_parallel_sliding(sliding_windows, combinations)
        
        # STEP 6: Select global model (sliding windows + AICc ranks)
        best_order, confidence = self._select_global_model(sliding_results)
        # STEP 6b: Reconcile with full-sample AIC on top-K candidates (closer to textbook ARIMA)
        if self._full_sample_reconcile:
            best_order, confidence = self._reconcile_order_full_sample_aic(
                working_data, best_order, confidence
            )
        self.order_ = best_order
        
        # STEP 7: Create fixed windows for validation
        fixed_windows = self._create_fixed_windows(
            working_data,
            config['num_fixed_windows']
        )
        
        # STEP 8: Backtesting
        validation_results = self._backtest_fixed_windows(fixed_windows, best_order)
        
        # STEP 9: Residual diagnostics
        diagnostics = self._diagnose_residuals(fixed_windows, best_order)
        
        # STEP 10: Check if local adjustment is needed
        needs_adjustment, failed_windows = self._check_diagnostic_failures(diagnostics)
        
        if needs_adjustment:
            if self.verbose:
                print(f"\n  ⚠ Diagnostic failures detected in {len(failed_windows)} windows")
                print("  Attempting local adjustment...")
            final_order = self._try_local_adjustment(working_data, best_order)
            self.order_ = final_order
        else:
            if self.verbose:
                print("\n  ✓ Diagnostics passed - no adjustment needed")
            final_order = best_order
        
        # Fit final model on all data
        if self.verbose:
            print("\n" + "="*70)
            print("FITTING FINAL MODEL")
            print("="*70)
            print(f"  Model: ARIMA{final_order}")
            print("  Fitting on full dataset...")
        
        p, d, q = final_order
        self._final_model = ARIMAProcess(
            ar_order=p,
            diff_order=d,
            ma_order=q,
            trend='c',
            n_jobs=1
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._final_model.fit(working_data)
        
        # Extract final parameters
        params = self._final_model._fitted_params
        self.parameters_ = {
            'phi': [params['parameters'].get(f'phi_{i+1}', 0.0) for i in range(p)],
            'theta': [params['parameters'].get(f'theta_{i+1}', 0.0) for i in range(q)],
            'c': self._final_model.constant if hasattr(self._final_model, 'constant') else 0.0,
            'sigma2': params['parameters'].get('sigma2', 0.0),
            'log_likelihood': params['log_likelihood'],
            'aic': params['aic'],
            'bic': params['bic']
        }
        
        self.fitted_ = True
        self.working_data_ = np.asarray(working_data, dtype=float).copy()
        
        if self.verbose:
            print(f"  ✓ Model fitted successfully")
            print("\n" + "="*70)
            print("PARALLEL ARIMA WORKFLOW - COMPLETED")
            print("="*70)
            print(f"  Final model: ARIMA{final_order}")
            print(f"  AIC: {params['aic']:.2f}")
            print(f"  BIC: {params['bic']:.2f}")
        
        return self
    
    # =========================================================================
    # STEP 11: Forecast (Non-Parallel)
    # =========================================================================
    
    def predict(self, 
                steps: int = 1, 
                return_conf_int: bool = False,
                alpha: float = 0.05) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate forecast from fitted model
        
        Uses most recent data with fitted parameters. Non-parallelized.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals
            
        Returns:
        --------
        predictions : np.ndarray
            Forecast values
        conf_int : tuple, optional
            (lower, upper) confidence intervals if return_conf_int=True
        """
        if not self.fitted_:
            raise ValueError("Workflow must be fitted before prediction")
        
        if steps <= 0:
            raise ValueError("Steps must be positive")
        
        if self.verbose:
            print(f"\nGenerating {steps}-step forecast...")
        
        # Generate predictions
        if return_conf_int:
            predictions, conf_int = self._final_model.predict(
                steps=steps,
                return_conf_int=True
            )
            
            # Transform back if log transform was applied
            if self._log_transformer is not None:
                predictions = self._log_transformer.inverse_transform(predictions)
                conf_int = (
                    self._log_transformer.inverse_transform(conf_int[0]),
                    self._log_transformer.inverse_transform(conf_int[1])
                )
            
            return predictions, conf_int
        else:
            predictions = self._final_model.predict(steps=steps)
            
            # Transform back if log transform was applied
            if self._log_transformer is not None:
                predictions = self._log_transformer.inverse_transform(predictions)
            
            return predictions
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get comprehensive results from all workflow steps
        
        Returns:
        --------
        results : dict
            Complete results including all intermediate steps
        """
        if not self.fitted_:
            raise ValueError("Workflow must be fitted before accessing results")
        
        return {
            'order': self.order_,
            'parameters': self.parameters_,
            'config': self._config,
            'step_results': self.results_
        }
    
    def summary(self) -> str:
        """
        Generate comprehensive summary of workflow
        
        Returns:
        --------
        summary : str
            Formatted summary string
        """
        if not self.fitted_:
            raise ValueError("Workflow must be fitted before summary")
        
        summary = []
        summary.append("="*70)
        summary.append("PARALLEL ARIMA WORKFLOW SUMMARY")
        summary.append("="*70)
        summary.append("")
        
        # Final model
        summary.append(f"Final Model: ARIMA{self.order_}")
        summary.append("")
        
        # Parameters
        summary.append("Parameters:")
        p, d, q = self.order_
        if p > 0:
            summary.append(f"  AR coefficients (φ): {[f'{x:.4f}' for x in self.parameters_['phi']]}")
        if q > 0:
            summary.append(f"  MA coefficients (θ): {[f'{x:.4f}' for x in self.parameters_['theta']]}")
        summary.append(f"  Constant (c): {self.parameters_['c']:.4f}")
        summary.append(f"  Variance (σ²): {self.parameters_['sigma2']:.4f}")
        summary.append("")
        
        # Model statistics
        summary.append("Model Statistics:")
        summary.append(f"  Log-Likelihood: {self.parameters_['log_likelihood']:.4f}")
        summary.append(f"  AIC: {self.parameters_['aic']:.4f}")
        summary.append(f"  BIC: {self.parameters_['bic']:.4f}")
        summary.append("")
        
        # Configuration
        summary.append("Configuration:")
        summary.append(f"  Dataset size: {len(self.data_)} observations")
        summary.append(f"  grid_mode: {self._grid_mode}")
        summary.append(f"  Max p: {self._config['max_p']}")
        summary.append(f"  Max q: {self._config['max_q']}")
        summary.append(f"  Sliding windows: {self._config['num_sliding_windows']}")
        summary.append(f"  Fixed windows: {self._config['num_fixed_windows']}")
        summary.append("")
        
        # Model selection
        if self.results_['step6_model_selection']:
            sel = self.results_['step6_model_selection']
            summary.append("Model Selection:")
            summary.append(f"  Selected from: {self.results_['step2_3_combinations']['num_combinations']} combinations")
            summary.append(f"  Confidence score: {sel['confidence_score']:.2f}")
            summary.append(f"  Appeared in: {sel['best_metrics']['appearance_count']}/{sel['best_metrics']['num_windows']} windows")
            summary.append("")
        
        rec = self.results_.get("step6_full_sample_reconciliation")
        if rec and not rec.get("skipped") and rec.get("enabled"):
            summary.append("Full-sample AIC reconciliation (step 6b):")
            if rec.get("changed_from_sliding"):
                summary.append(
                    f"  Order: ARIMA{tuple(rec['order_sliding'])} (sliding) → "
                    f"ARIMA{tuple(rec['order_full_sample'])} (min AIC on full series)"
                )
            else:
                summary.append(
                    f"  Order: ARIMA{tuple(rec['order_full_sample'])} "
                    "(matches sliding pick; best full-sample AIC among candidates)"
                )
            if rec.get("best_full_sample_aic") is not None:
                summary.append(f"  Best full-sample AIC: {rec['best_full_sample_aic']:.4f}")
            summary.append("")
        
        # Validation metrics
        if self.results_['step7_8_validation']:
            val = self.results_['step7_8_validation']
            if 'metrics' in val:
                summary.append("Validation Metrics (Backtesting):")
                summary.append(f"  MAE: {val['metrics']['avg_mae']:.4f}")
                summary.append(f"  RMSE: {val['metrics']['avg_rmse']:.4f}")
                summary.append(f"  MAPE: {val['metrics']['avg_mape']:.2f}%")
                summary.append("")
        
        # Diagnostics
        if self.results_['step9_diagnostics']:
            diag = self.results_['step9_diagnostics']
            summary.append("Residual Diagnostics:")
            summary.append(f"  Overall pass rate: {diag['pass_rates']['overall']*100:.1f}%")
            summary.append(f"  Ljung-Box pass rate: {diag['pass_rates']['ljung_box']*100:.1f}%")
            summary.append(f"  ACF pass rate: {diag['pass_rates']['acf']*100:.1f}%")
            summary.append("")
        
        # Adjustment
        if self.results_['step10_adjustment']:
            adj = self.results_['step10_adjustment']
            if adj['adjusted']:
                summary.append("Local Adjustment:")
                summary.append(f"  Original: ARIMA{adj['current_order']}")
                summary.append(f"  Adjusted: ARIMA{adj['final_order']}")
                summary.append(f"  AICc improvement: {adj['improvement_pct']:.1f}%")
                summary.append("")
        
        summary.append("="*70)
        
        return "\n".join(summary)

