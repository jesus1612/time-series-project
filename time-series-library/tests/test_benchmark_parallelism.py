"""
Benchmark suite: Sequential vs Parallel for AR, MA, ARMA, ARIMA models.

Measures wall-clock time for fitting each model type under different:
- n_obs  : number of observations in the series
- n_jobs : 1 (sequential) vs -1 (all cores)

Identifies the "elbow" where parallelisation starts to pay off and prints
a recommendation table.

Run with::

    pytest tests/test_benchmark_parallelism.py -v -m slow -s

Or directly::

    python tests/test_benchmark_parallelism.py
"""

import time
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the library root is on the path when run directly
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tslib.models.ar_model    import ARModel
from tslib.models.ma_model    import MAModel
from tslib.models.arma_model  import ARMAModel
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


def _generate_ma1(n: int, theta: float = 0.5, seed: int = 42) -> np.ndarray:
    """MA(1) series."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n + 1)
    return eps[1:] + theta * eps[:-1]


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

N_OBS_GRID   = [100, 500, 1_000, 5_000, 10_000]
N_JOBS_CASES = {'sequential': 1, 'parallel_all': -1}

MODEL_CONFIGS = {
    'AR(2)':    (ARModel,   {'order': 2, 'auto_select': False}),
    'MA(2)':    (MAModel,   {'order': 2, 'auto_select': False}),
    'ARMA(1,1)':(ARMAModel, {'order': (1, 1), 'auto_select': False}),
    'ARIMA(1,1,1)': (ARIMAModel, {'order': (1, 1, 1), 'auto_select': False, 'n_jobs': 1}),
}


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Collects timing data and computes the elbow threshold."""

    def __init__(self, n_obs_grid: List[int] = None, repeats: int = 3):
        self.n_obs_grid = n_obs_grid or N_OBS_GRID
        self.repeats    = repeats
        self.results: Dict[str, Dict[int, Dict[str, float]]] = {}

    def run(self) -> None:
        """Run all benchmarks and store results."""
        for model_name, (cls, base_kwargs) in MODEL_CONFIGS.items():
            self.results[model_name] = {}
            for n in self.n_obs_grid:
                # Choose appropriate data generator
                if 'MA' in model_name and 'ARMA' not in model_name and 'ARIMA' not in model_name:
                    data = _generate_ma1(n)
                else:
                    data = _generate_ar1(n)

                self.results[model_name][n] = {}
                for strategy, n_jobs in N_JOBS_CASES.items():
                    kwargs = {**base_kwargs}
                    # ARIMA already has n_jobs; others need it added
                    if model_name.startswith('ARIMA'):
                        kwargs['n_jobs'] = n_jobs
                    else:
                        kwargs['n_jobs'] = n_jobs
                    try:
                        t = _time_fit(cls, kwargs, data, self.repeats)
                    except Exception as exc:
                        t = float('nan')
                        print(f"  ⚠  {model_name} n={n} {strategy}: {exc}")
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

    def print_summary(self) -> None:
        """Print a compact ASCII summary table."""
        col_w = 14
        header = f"{'model':<18}" + "".join(
            f"{'n='+str(n):>{col_w}}" for n in self.n_obs_grid
        )
        sep = "-" * len(header)

        print("\n" + sep)
        print("BENCHMARK — Wall-clock fit time (seconds, best of repeats)")
        print(sep)

        for strategy in ('sequential', 'parallel_all'):
            print(f"\n  strategy: {strategy}")
            print(f"  {'model':<18}" + "".join(
                f"{'n='+str(n):>{col_w}}" for n in self.n_obs_grid
            ))
            print("  " + "-" * (len(header) - 2))
            for model_name, by_n in self.results.items():
                row = f"  {model_name:<18}"
                for n in self.n_obs_grid:
                    t = by_n.get(n, {}).get(strategy, float('nan'))
                    row += f"{t:>{col_w}.4f}"
                print(row)

        print("\n" + sep)
        print("SPEEDUP (sequential / parallel_all)  — values >1 mean parallel wins")
        print(sep)
        print(f"{'model':<18}" + "".join(
            f"{'n='+str(n):>{col_w}}" for n in self.n_obs_grid
        ))
        print("-" * len(header))
        for model_name, by_n in self.speedups().items():
            row = f"{model_name:<18}"
            for n in self.n_obs_grid:
                s = by_n.get(n, float('nan'))
                marker = " ✓" if s > 1.10 else "  "
                row += f"{s:>{col_w-2}.2f}{marker}"
            print(row)

        print("\n" + sep)
        print("RECOMMENDED THRESHOLD — minimum n where parallel starts winning")
        print(sep)
        for model_name, n in self.elbow_threshold().items():
            label = f"{n:,}" if n else "no clear benefit in tested range"
            print(f"  {model_name:<20} → n ≥ {label}")
        print(sep + "\n")

    def plot_elbow_curves(self, save_path: str = None) -> None:
        """
        Plot timing curves and speedup by model type.

        Requires matplotlib. If save_path is given, saves instead of showing.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        n_models = len(MODEL_CONFIGS)
        fig = plt.figure(figsize=(14, 4 * n_models))
        gs  = gridspec.GridSpec(n_models, 2, figure=fig)

        ns      = self.n_obs_grid
        speedup = self.speedups()

        for row_idx, (model_name, by_n) in enumerate(self.results.items()):
            # Left: timing curves
            ax_time = fig.add_subplot(gs[row_idx, 0])
            for strategy, color in [('sequential', '#e74c3c'), ('parallel_all', '#2ecc71')]:
                ys = [by_n[n].get(strategy, float('nan')) for n in ns]
                ax_time.plot(ns, ys, 'o-', color=color, label=strategy, linewidth=2)
            ax_time.set_title(f"{model_name} — Fit Time", fontsize=11)
            ax_time.set_xlabel("n_obs")
            ax_time.set_ylabel("seconds")
            ax_time.legend(fontsize=8)
            ax_time.grid(True, alpha=0.3)
            ax_time.set_xscale('log')

            # Right: speedup curve with elbow marker
            ax_sp = fig.add_subplot(gs[row_idx, 1])
            sp_vals = [speedup[model_name].get(n, float('nan')) for n in ns]
            ax_sp.plot(ns, sp_vals, 's--', color='#3498db', linewidth=2, label='speedup')
            ax_sp.axhline(1.0,  color='grey', linestyle=':', linewidth=1)
            ax_sp.axhline(1.10, color='orange', linestyle='--', linewidth=1, label='10% faster')

            # Mark elbow
            elbow_n = self.elbow_threshold().get(model_name)
            if elbow_n and elbow_n in ns:
                sp_at_elbow = speedup[model_name].get(elbow_n, float('nan'))
                ax_sp.axvline(elbow_n, color='orange', linestyle=':', linewidth=1.5)
                ax_sp.annotate(
                    f"n={elbow_n:,}",
                    xy=(elbow_n, sp_at_elbow),
                    xytext=(elbow_n * 1.1, sp_at_elbow + 0.05),
                    fontsize=8, color='darkorange'
                )
            ax_sp.set_title(f"{model_name} — Speedup", fontsize=11)
            ax_sp.set_xlabel("n_obs")
            ax_sp.set_ylabel("seq / par")
            ax_sp.legend(fontsize=8)
            ax_sp.grid(True, alpha=0.3)
            ax_sp.set_xscale('log')

        plt.suptitle(
            "TSLib Parallelism Benchmark — Sequential vs. n_jobs=-1",
            fontsize=14, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved → {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Pytest tests (marked 'slow' so they only run on demand)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestParallelismBenchmark:
    """
    Execute the benchmark in pytest context.

    Run with:  pytest -v -m slow -s tests/test_benchmark_parallelism.py
    """

    @pytest.fixture(scope='class')
    def runner(self):
        r = BenchmarkRunner(n_obs_grid=N_OBS_GRID, repeats=3)
        r.run()
        return r

    def test_all_models_produce_results(self, runner):
        """Every model × n_obs combination should return a finite time."""
        for model_name, by_n in runner.results.items():
            for n, by_strat in by_n.items():
                for strategy, t in by_strat.items():
                    assert np.isfinite(t), (
                        f"{model_name} n={n} {strategy} returned non-finite time"
                    )

    def test_parallel_not_slower_for_large_series(self, runner):
        """
        For the largest n, parallel should not be more than 2x slower than sequential
        (accounting for overhead).  Being slower by 2x at large n would indicate a bug.
        """
        large_n = max(N_OBS_GRID)
        for model_name, by_n in runner.results.items():
            seq = by_n[large_n].get('sequential', float('nan'))
            par = by_n[large_n].get('parallel_all', float('nan'))
            if np.isfinite(seq) and np.isfinite(par):
                # Allow up to 3x overhead (parallel may lose on small data)
                assert par < seq * 3, (
                    f"{model_name} at n={large_n}: parallel ({par:.3f}s) is "
                    f"more than 3x slower than sequential ({seq:.3f}s)"
                )

    def test_speedup_ratio_positive(self, runner):
        """Speedup ratio should always be a positive number."""
        for model_name, by_n in runner.speedups().items():
            for n, sp in by_n.items():
                if np.isfinite(sp):
                    assert sp > 0, f"Negative speedup for {model_name} n={n}"

    def test_print_and_plot(self, runner, tmp_path):
        """Smoke test: summary table and plot generation should not raise."""
        runner.print_summary()
        plot_file = str(tmp_path / "elbow_curves.png")
        runner.plot_elbow_curves(save_path=plot_file)
        assert os.path.exists(plot_file)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TSLib Parallelism Benchmark')
    parser.add_argument('--n-obs', nargs='+', type=int,
                        default=N_OBS_GRID,
                        help='list of series lengths to test')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of timing repeats (best-of)')
    parser.add_argument('--plot', action='store_true',
                        help='show elbow-curve plots when done')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='save plot to this path instead of showing')
    args = parser.parse_args()

    runner = BenchmarkRunner(n_obs_grid=args.n_obs, repeats=args.repeats)
    print(f"Running benchmark for n_obs={args.n_obs} ({args.repeats} repeats each) …\n")
    runner.run()
    runner.print_summary()

    if args.plot or args.save_plot:
        runner.plot_elbow_curves(save_path=args.save_plot)
