"""
Distributed AR, MA, ARMA, and ARIMA with Spark (GenericParallelProcessor)

All four TSLib high-level models can run in parallel over many series: each
Spark task fits one group with the same linear model (fixed order) and returns
forecasts. Requires Java 17+ and: pip install -r requirements-spark.txt

Run:
    python examples/spark_parallel_generic_models.py
"""

import numpy as np
import pandas as pd

from tslib.utils.checks import check_spark_availability


def main():
    if not check_spark_availability():
        print("PySpark not installed. pip install -r requirements-spark.txt")
        return

    from pyspark.sql import SparkSession
    from tslib.spark.parallel_processor import GenericParallelProcessor

    # Build long-format data: three short series
    rows = []
    rng = np.random.default_rng(0)
    for sid, phi in [("s1", 0.5), ("s2", 0.3), ("s3", -0.4)]:
        eps = rng.normal(0, 1, 40)
        y = np.zeros(40)
        for t in range(1, 40):
            y[t] = phi * y[t - 1] + eps[t]
        for t in range(40):
            rows.append((sid, float(y[t])))

    pdf = pd.DataFrame(rows, columns=["series_id", "y"])

    spark = SparkSession.builder.appName("TSLib-GenericModels-Demo").getOrCreate()
    sdf = spark.createDataFrame(pdf)

    configs = [
        ("AR", 2, "AR(2) fixed order"),
        ("MA", 2, "MA(2) fixed order"),
        ("ARMA", (1, 1), "ARMA(1,1) fixed order"),
        ("ARIMA", (1, 1, 1), "ARIMA(1,1,1) fixed order"),
    ]

    for model_type, order, label in configs:
        proc = GenericParallelProcessor(
            model_type=model_type,
            spark=spark,
            n_jobs=1,
            master="local[*]",
            app_name=f"TSLib-demo-{model_type}",
        )
        out = proc.fit_multiple(
            sdf, group_col="series_id", value_col="y", order=order, steps=3
        )
        n_ok = out.filter(out.status == "ok").count()
        print(f"{label}: rows with status ok (forecast steps) = {n_ok}")
        out.show(6, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
