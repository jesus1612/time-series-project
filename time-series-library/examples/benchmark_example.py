"""
Benchmark example: Sequential vs Parallel for AR, MA, ARMA, ARIMA.

Demonstrates the n_jobs parameter and shows at which series length
parallelism starts to pay off.

Run with::

    python examples/benchmark_example.py
    python examples/benchmark_example.py --plot --save-plot docs/images/elbow_curves.png
"""

import sys
import os

# Make sure library is importable from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_benchmark_parallelism import BenchmarkRunner

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TSLib Parallelism Benchmark Example')
    parser.add_argument('--n-obs', nargs='+', type=int,
                        default=[100, 500, 1_000, 5_000],
                        help='series lengths to benchmark')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--plot', action='store_true', help='show plots')
    parser.add_argument('--save-plot', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  TSLib Parallelism Benchmark")
    print("  Comparing n_jobs=1 (sequential) vs n_jobs=-1 (all cores)")
    print("=" * 60)

    runner = BenchmarkRunner(n_obs_grid=args.n_obs, repeats=args.repeats)
    runner.run()
    runner.print_summary()

    thresholds = runner.elbow_threshold()
    print("\n📌 Quick reference:")
    for model, n in thresholds.items():
        if n:
            print(f"   {model:20s}  → use n_jobs=-1 when n_obs ≥ {n:,}")
        else:
            print(f"   {model:20s}  → parallel showed no benefit in this range")

    if args.plot or args.save_plot:
        runner.plot_elbow_curves(save_path=args.save_plot)
