import time
import os
import sys
from typing import Dict, List, Tuple
import numpy as np

from tslib.models.arima_model import ARIMAModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_ar1(n: int, phi: float = 0.7, seed: int = 42) -> np.ndarray:
    """Stationary AR(1) series."""
    rng = np.random.default_rng(seed)
    y   = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + eps[t]
    return y


def _time_fit(model_cls, model_kwargs: dict, data: np.ndarray, repeats: int = 3) -> float:
    """Return best (min) wall-clock fit time over *repeats* runs (seconds)."""
    times = []
    for _ in range(repeats):
        m = model_cls(**model_kwargs)
        t0 = time.perf_counter()
        m.fit(data)
        times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

N_OBS_GRID = [100, 500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000]
N_JOBS_CASES = {'sequential': 1, 'parallel_all': -1}

# Benchmark tab focuses on ARIMA only (several orders); wizard still offers AR / MA / ARMA.
MODEL_CONFIGS = {
    "ARIMA(1,1,1)": (ARIMAModel, {"order": (1, 1, 1), "auto_select": False, "n_jobs": 1}),
    "ARIMA(2,1,1)": (ARIMAModel, {"order": (2, 1, 1), "auto_select": False, "n_jobs": 1}),
    "ARIMA(1,1,2)": (ARIMAModel, {"order": (1, 1, 2), "auto_select": False, "n_jobs": 1}),
}


class BenchmarkRunner:
    """Collects timing data and computes the elbow threshold."""

    def __init__(self, n_obs_grid: List[int] = None, repeats: int = 3, seed: int = 42):
        self.n_obs_grid = n_obs_grid or N_OBS_GRID
        self.repeats    = repeats
        self.seed       = seed
        self.results: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.models = list(MODEL_CONFIGS.keys())

    def run(self) -> None:
        """Run all benchmarks and store results."""
        for model_name, (cls, base_kwargs) in MODEL_CONFIGS.items():
            self.results[model_name] = {}
            for n in self.n_obs_grid:
                data = _generate_ar1(n, seed=self.seed)

                self.results[model_name][n] = {}
                for strategy, n_jobs in N_JOBS_CASES.items():
                    kwargs = {**base_kwargs}
                    # ARIMA already has n_jobs; others need it added
                    kwargs['n_jobs'] = n_jobs
                    
                    try:
                        t = _time_fit(cls, kwargs, data, self.repeats)
                    except Exception as exc:
                        t = float('nan')
                        # Log error internally or print
                    self.results[model_name][n][strategy] = t

    def speedups(self) -> Dict[str, Dict[int, float]]:
        """Return sequential / parallel speedup ratio."""
        out = {}
        for model_name, by_n in self.results.items():
            out[model_name] = {}
            for n, by_strat in by_n.items():
                seq = by_strat.get('sequential', float('nan'))
                par = by_strat.get('parallel_all', float('nan'))
                out[model_name][n] = seq / par if par and par > 0 else float('nan')
        return out

    def elbow_threshold(self) -> Dict[str, int]:
        """
        For each model, return the smallest n where parallel speedup > 1.10
        (i.e., parallel is at least 10% faster).
        """
        thresholds = {}
        for model_name, by_n in self.speedups().items():
            threshold = None
            for n in sorted(by_n):
                if by_n[n] > 1.10:
                    threshold = n
                    break
            thresholds[model_name] = threshold
        return thresholds
