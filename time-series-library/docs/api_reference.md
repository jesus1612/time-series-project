# TSLib — Referencia de API

> Guía de integración rápida con todos los métodos públicos, parámetros y ejemplos funcionales.  
> Para la teoría matemática completa ver [`docs/MATHEMATICS_BY_MODEL.md`](MATHEMATICS_BY_MODEL.md).

---

## Instalación

```bash
# Instalación básica (sin Spark)
pip install -e /ruta/a/time-series-library

# Con soporte PySpark
pip install -e /ruta/a/time-series-library[spark]

# Verificar
python -c "from tslib import ARIMAModel; print('OK')"
```

---

## Modelos de alto nivel

Todos los modelos de alto nivel siguen la misma interfaz (scikit-learn style).

### `ARModel`

**Importación:**
```python
from tslib import ARModel
```

**Constructor:**
```python
ARModel(
    order: int | None = None,       # Orden p; None = auto-selección
    trend: str = 'c',               # 'c' (constante) o 'n' (sin constante)
    auto_select: bool = True,       # Activar selección automática
    max_order: int = 5,             # Máximo p a probar en auto-selección
    selection_method: str = 'pacf', # 'pacf' (corte PACF) o 'aic' (AIC)
    validation: bool = True,        # Validar datos de entrada
    n_jobs: int = 1,                # Núcleos para MLE/ACF/PACF; -1 = todos
)
```

> [!TIP]
> Usa `n_jobs=-1` con series de más de ~1 000 observaciones para aprovechar todos los núcleos del CPU.

**Métodos:**

| Método | Descripción | Retorna |
|--------|-------------|---------|
| `fit(data)` | Ajusta el modelo | `self` |
| `predict(steps=1)` | Pronostica h pasos | `np.ndarray` (shape: `[steps]`) |
| `get_residuals()` | Residuales del ajuste | `np.ndarray` |
| `get_fitted_values()` | Valores ajustados en training | `np.ndarray` |
| `summary()` | Resumen del modelo | `dict` |
| `get_residual_diagnostics()` | Tests sobre residuales | `dict` |
| `evaluate_forecast(test, steps)` | Métricas vs valores reales | `dict` |
| `plot_diagnostics()` | Gráficas de diagnóstico | — (muestra figura) |
| `plot_forecast(steps)` | Gráfica con intervalo de confianza | — (muestra figura) |

**Ejemplo:**
```python
import numpy as np
from tslib import ARModel

# Serie AR(2) sintética
np.random.seed(42)
n = 200
phi1, phi2 = 0.6, -0.3
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.5 + phi1 * y[t-1] + phi2 * y[t-2] + np.random.normal(0, 1)

# Secuencial (default)
model = ARModel(auto_select=True, max_order=5)
model.fit(y)
print(f"Orden seleccionado: AR({model.order})")
print("Pronóstico:", model.predict(steps=6))

# Paralelo (todos los núcleos)
model_par = ARModel(order=2, auto_select=False, n_jobs=-1)
model_par.fit(y)
print("Pronóstico paralelo:", model_par.predict(steps=6))
```

---

### `MAModel`

**Importación:**
```python
from tslib import MAModel
```

**Constructor:**
```python
MAModel(
    order: int | None = None,       # Orden q
    trend: str = 'c',
    auto_select: bool = True,
    max_order: int = 5,
    selection_method: str = 'acf',  # 'acf' (corte ACF) o 'aic'
    validation: bool = True,
    n_jobs: int = 1,                # Núcleos para MLE/ACF; -1 = todos
)
```

**Mismos métodos** que `ARModel` (`fit`, `predict`, `get_residuals`, etc.)

**Ejemplo:**
```python
import numpy as np
from tslib import MAModel

# Serie MA(2) sintética
np.random.seed(42)
n = 200
eps = np.random.normal(0, 1, n + 2)
theta1, theta2 = 0.7, -0.4
y = np.array([eps[t] + theta1*eps[t-1] + theta2*eps[t-2] for t in range(2, n+2)])

model = MAModel(auto_select=True, max_order=5)
model.fit(y)

print(f"Orden seleccionado: MA({model.order})")
print("Pronóstico:", model.predict(steps=4))
```

---

### `ARMAModel`

**Constructor:**
```python
ARMAModel(
    order: tuple | None = None,         # (p, q); None = auto-selección
    trend: str = 'c',
    auto_select: bool = True,
    max_p: int = 3,                     # Máximo p en grid search
    max_q: int = 3,                     # Máximo q en grid search
    selection_criterion: str = 'aic',   # 'aic' o 'bic'
    validation: bool = True,
    n_jobs: int = 1,                    # Núcleos para MLE/ACF; -1 = todos
)
```

**Ejemplo:**
```python
import numpy as np
from tslib import ARMAModel

# Serie ARMA(1,1) sintética
np.random.seed(42)
n = 300
phi, theta = 0.7, 0.4
eps = np.random.normal(0, 1, n + 1)
y = np.zeros(n + 1)
for t in range(1, n + 1):
    y[t] = phi * y[t-1] + eps[t] + theta * eps[t-1]
y = y[1:]

model = ARMAModel(auto_select=True, max_p=3, max_q=3)
model.fit(y)

p, q = model.order
print(f"Orden seleccionado: ARMA({p}, {q})")
print("Pronóstico:", model.predict(steps=6))
```

---

### `ARIMAModel`

**Constructor:**
```python
ARIMAModel(
    order: tuple | None = None,         # (p, d, q); None = auto-selección
    auto_select: bool = True,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    selection_criterion: str = 'aic',
    validation: bool = True,
    n_jobs: int = -1,                   # Default: todos los núcleos
)
```

**Métodos adicionales:**

| Método | Descripción |
|--------|-------------|
| `get_exploratory_analysis()` | ACF/PACF y tests de estacionariedad |

**Ejemplo:**
```python
import numpy as np
from tslib import ARIMAModel

# Serie no estacionaria con tendencia
np.random.seed(42)
n = 150
trend = 0.05 * np.arange(n)
y = np.cumsum(np.random.normal(0, 1, n)) + trend  # paseo aleatorio + tendencia

model = ARIMAModel(auto_select=True, max_p=3, max_d=2, max_q=3)
model.fit(y)

p, d, q = model.order
print(f"Orden seleccionado: ARIMA({p}, {d}, {q})")
forecast = model.predict(steps=10)
print("Pronóstico:", forecast)

# Diagnósticos
diagnostics = model.get_residual_diagnostics()
print("Residuales normales:", diagnostics.get('normality', {}).get('is_normal', '—'))
```

---

## Utilidades de Preprocesado

### `DataValidator`

```python
from tslib.preprocessing.validation import DataValidator

validator = DataValidator(
    min_length: int = 10,
    check_stationarity: bool = True,
    outlier_method: str = 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
)

report = validator.validate(data)     # DataQualityReport
clean  = validator.clean_data(data)   # np.ndarray limpio
```

`DataQualityReport` contiene:
- `is_valid: bool`
- `issues: list[str]`
- `recommendations: list[str]`
- `statistics: dict`  (media, std, NaN count, outlier count…)

---

## ACF / PACF

### Sin Spark

```python
from tslib.core.acf_pacf import ACFCalculator, PACFCalculator, ACFPACFAnalyzer

acf   = ACFCalculator(max_lags=20)
lags, acf_values = acf.calculate(data)     # np.ndarray, np.ndarray

pacf  = PACFCalculator(max_lags=20)
lags, pacf_values = pacf.calculate(data)

analyzer = ACFPACFAnalyzer(max_lags=20)
result   = analyzer.analyze(data)
# result["suggested_orders"]["suggested_p"], ["suggested_q"]
```

### Con Spark

```python
from tslib.spark.acf_pacf import SparkACFCalculator, SparkACFPACFAnalyzer

acf    = SparkACFCalculator(max_lags=20, spark_session=spark)
lags, acf_values = acf.calculate(data)

result = SparkACFPACFAnalyzer(spark_session=spark).analyze(data)
```

---

## Tests de Estacionariedad

```python
from tslib.core.stationarity import ADFTest, KPSSTest

# ADF
adf = ADFTest(max_lags=None, regression='c')  # 'c', 'ct', 'n'
result = adf.test(data)
# result["statistic"], result["p_value"], result["is_stationary"]

# KPSS
kpss = KPSSTest()
result = kpss.test(data)
```

---

## MLE Optimizer (bajo nivel)

```python
from tslib.core.optimization import MLEOptimizer

opt = MLEOptimizer(
    method: str = 'L-BFGS-B',
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
)

result = opt.estimate(data, model_type='ARIMA', p=1, d=1, q=1)
# result["params"] → np.ndarray (φ_1…φ_p, θ_1…θ_q, σ²)
# result["log_likelihood"]
# result["std_errors"]
# result["aic"], result["bic"]
```

---

## Paralelización Spark

### `ParallelARIMAWorkflow` (flujo completo)

```python
from tslib.spark.parallel_arima_workflow import ParallelARIMAWorkflow
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ARIMA").getOrCreate()

workflow = ParallelARIMAWorkflow(
    spark_session=spark,
    max_p=3, max_d=2, max_q=3,
    forecast_steps=12,
    group_col='series_id',    # columna que identifica cada serie
    time_col='date',
    value_col='value',
)

# df_spark: DataFrame Spark con columnas [group_col, time_col, value_col]
results = workflow.run(df_spark)
# results: DataFrame Spark con columnas [series_id, model_order, forecast, metrics]
```

Internamente ejecuta los 11 pasos documentados en `docs/MATHEMATICS_BY_MODEL.md`.

### `ParallelARIMAProcessor` (una sola llamada)

```python
from tslib.spark.parallel_arima import ParallelARIMAProcessor

processor = ParallelARIMAProcessor(spark_session=spark)
result_df = processor.fit_predict(
    df=df_spark,
    group_col='series_id',
    value_col='value',
    p=1, d=1, q=1,
    forecast_steps=6,
)
```

---

### `GenericParallelProcessor` (todos los modelos)

```python
from tslib.spark.parallel_processor import GenericParallelProcessor
```

**Constructor:**
```python
GenericParallelProcessor(
    model_type: str = 'ARIMA',  # 'AR', 'MA', 'ARMA', 'ARIMA'
    spark: SparkSession = None, # None → getOrCreate()
    n_jobs: int = 1,            # Paralelismo dentro de cada modelo
)
```

**Métodos:**

| Método | Descripción | Retorna |
|--------|-------------|---------|
| `fit_multiple(df, group_col, value_col, order, steps)` | Ajusta y pronostica en Spark (Pandas UDF) | `DataFrame` |
| `fit_and_collect(df, group_col, value_col, order, steps)` | Como `fit_multiple` pero devuelve lista de dicts | `list[dict]` |
| `fit_multiple_sequential(series_dict, model_type, order, steps, n_jobs)` | **Sin Spark** — fallback puro Python | `dict[str, dict]` |

**Ejemplo sin Spark (útil para pruebas y lotes pequeños):**
```python
import numpy as np
from tslib.spark.parallel_processor import GenericParallelProcessor

series = {
    'ventas_norte': np.random.normal(100, 10, 500).cumsum(),
    'ventas_sur':   np.random.normal( 80, 12, 500).cumsum(),
}

results = GenericParallelProcessor.fit_multiple_sequential(
    series, model_type='ARIMA', order=(1, 1, 0), steps=12
)

for sid, res in results.items():
    print(f"{sid}: status={res['status']}, forecast={res['forecast'].round(2)}")
```

**Ejemplo con Spark:**
```python
from pyspark.sql import SparkSession
from tslib.spark.parallel_processor import GenericParallelProcessor

spark = SparkSession.builder.appName('tslib').getOrCreate()
proc  = GenericParallelProcessor(model_type='AR', spark=spark, n_jobs=4)

# df_spark: columnas [id_serie, valor, ...]
result_df = proc.fit_multiple(
    df        = df_spark,
    group_col = 'id_serie',
    value_col = 'valor',
    order     = 2,
    steps     = 6,
)
result_df.show()
```

---

## Métricas de Evaluación

```python
from tslib.metrics.evaluation import ForecastMetrics, InformationCriteria, ResidualAnalyzer

# Métricas de pronóstico
fm = ForecastMetrics()
metrics = fm.calculate(y_true, y_pred)
# metrics["mae"], ["rmse"], ["mape"], ["smape"]

# Criterios de información
ic = InformationCriteria()
criteria = ic.calculate(log_likelihood, n_params, n_obs)
# criteria["aic"], ["bic"], ["aicc"], ["hqic"]

# Análisis de residuales
ra = ResidualAnalyzer()
diag = ra.analyze(residuals)
# diag["normality"]["is_normal"]
# diag["autocorrelation"]["is_white_noise"]
# diag["heteroscedasticity"]["is_homoscedastic"]
```

---

## Resumen de imports

```python
# Modelos de alto nivel
from tslib import ARModel, MAModel, ARMAModel, ARIMAModel

# Core
from tslib.core.acf_pacf   import ACFCalculator, PACFCalculator, ACFPACFAnalyzer
from tslib.core.stationarity import ADFTest, KPSSTest
from tslib.core.optimization import MLEOptimizer
from tslib.core.arima        import ARProcess, MAProcess, ARMAProcess, ARIMAProcess

# Preprocesado
from tslib.preprocessing.validation import DataValidator

# Métricas
from tslib.metrics.evaluation import ForecastMetrics, InformationCriteria, ResidualAnalyzer

# Spark (requiere PySpark instalado)
from tslib.spark             import check_spark_availability
from tslib.spark.acf_pacf    import SparkACFCalculator, SparkPACFCalculator
from tslib.spark.parallel_arima           import ParallelARIMAProcessor
from tslib.spark.parallel_arima_workflow  import ParallelARIMAWorkflow
from tslib.spark.parallel_processor       import GenericParallelProcessor
```
