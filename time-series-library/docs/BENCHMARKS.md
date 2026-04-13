# TSLib — Guía de Benchmarking y Paralelismo

> Referencia práctica para medir rendimiento y elegir la configuración óptima de `n_jobs`.

---

## ¿Qué mide el benchmark?

La suite compara el tiempo de ajuste (fit) en modo **secuencial** (`n_jobs=1`) vs **paralelo** (`n_jobs=-1`, todos los núcleos) para los cuatro modelos de TSLib, barriendo distintos tamaños de serie (`n_obs`).

La métrica clave es el **speedup**:

```
speedup = tiempo_secuencial / tiempo_paralelo
```

| speedup | Significado                                |
|---------|--------------------------------------------|
| > 1.10  | Paralelo gana ≥ 10% → recomendado          |
| ~ 1.00  | Neutro — overhead ≈ ganancia              |
| < 1.00  | Paralelo pierde (overhead domina ← normal en series cortas) |

---

## Cómo ejecutar el benchmark

### Ejemplo rápido (5 tamaños)

```bash
python examples/benchmark_example.py \
    --n-obs 100 500 1000 5000 10000 \
    --plot \
    --save-plot docs/images/elbow_curves.png
```

### Rango completo con plots

```bash
python examples/benchmark_example.py \
    --n-obs 100 500 1000 5000 10000 50000 \
    --repeats 5 \
    --plot
```

### Vía pytest (marcador `slow`)

```bash
pytest tests/test_benchmark_parallelism.py -v -m slow -s
```

---

## Resultados de referencia (macOS, 8 núcleos M-series)

> Tiempos en segundos. Cada celda es el mejor de 3 repeticiones.

### Tiempo de ajuste — Secuencial (`n_jobs=1`)

| Modelo         | n=100  | n=500  | n=1 000 | n=5 000 | n=10 000 |
|----------------|--------|--------|---------|---------|----------|
| AR(2)          | 0.005  | 0.019  | 0.037   | 0.156   | 0.320    |
| MA(2)          | 0.005  | 0.018  | 0.033   | 0.126   | 0.243    |
| ARMA(1,1)      | 0.006  | 0.018  | 0.029   | 0.163   | 0.421    |
| ARIMA(1,1,1)   | 0.006  | 0.030  | 0.070   | 0.259   | 0.476    |

### Speedup (`n_jobs=-1` vs `n_jobs=1`)

| Modelo         | n=100 | n=500 | n=1 000 | n=5 000 | n=10 000 |
|----------------|-------|-------|---------|---------|----------|
| AR(2)          | 1.01  | 1.00  | 0.64    | 0.57    | 0.66     |
| MA(2)          | 0.99  | 1.06  | 0.60    | 0.53    | 0.50     |
| ARMA(1,1)      | 1.01  | 1.07  | 0.52    | 0.62    | 0.75     |
| **ARIMA(1,1,1)**| 1.01 | 1.07  | **1.29 ✓**| 0.67  | 0.66     |

> [!NOTE]
> El speedup cae por debajo de 1 para series grandes porque el cuello de botella en AR/MA/ARMA puro es el `MLEOptimizer` (single-threaded dentro de scipy). El beneficio real de `n_jobs=-1` aparece cuando se ajustan **muchas series** con `GenericParallelProcessor`.

---

## Umbral de uso recomendado (por modelo)

| Modelo       | Usa `n_jobs=-1` cuando…                              |
|--------------|------------------------------------------------------|
| ARIMA        | n_obs ≥ 1 000                                        |
| AR / MA / ARMA | n_obs ≥ 5 000 **ó** ajustando ≥ 50 series a la vez |
| Todas + Spark | Siempre — Spark maneja la paralelización entre series |

---

## API: clase `BenchmarkRunner`

```python
from tests.test_benchmark_parallelism import BenchmarkRunner

runner = BenchmarkRunner(
    n_obs_grid=[100, 500, 1_000, 5_000, 10_000],
    repeats=3,   # número de repeticiones (se toma el mínimo)
)
runner.run()            # ejecuta todos los experimentos
runner.print_summary()  # imprime tabla ASCII
runner.plot_elbow_curves(save_path='docs/images/elbow.png')

# Acceder a resultados programáticamente
speedups   = runner.speedups()         # dict[model_name → dict[n → float]]
thresholds = runner.elbow_threshold()  # dict[model_name → int | None]
```

---

## Cómo interpretar las gráficas

Las gráficas generadas tienen dos columnas por modelo:

- **Izquierda — Fit Time:** Tiempo real en segundos. La línea **roja** es secuencial, la **verde** es paralela. Si la verde está encima, el paralelo pierde.
- **Derecha — Speedup:** El ratio. La línea de puntos naranja en 1.10 marca el umbral del 10%. El marcador vertical naranja es el "codo" (punto donde el paralelo empieza a ganar).

![Elbow curves benchmark](images/elbow_curves.png)

---

## Benchmark de muchas series (GenericParallelProcessor)

Para medir el beneficio real del procesamiento paralelo de múltiples series:

```python
import numpy as np
import time
from tslib.spark.parallel_processor import GenericParallelProcessor

# Generar 100 series de 1 000 observaciones
np.random.seed(42)
series = {f's{i}': np.cumsum(np.random.normal(0, 1, 1_000)) for i in range(100)}

# Secuencial
t0 = time.perf_counter()
r  = GenericParallelProcessor.fit_multiple_sequential(
    series, model_type='ARIMA', order=(1, 1, 0), steps=12, n_jobs=1
)
t_seq = time.perf_counter() - t0

# Paralelo (n_jobs=-1 dentro de cada modelo)
t0 = time.perf_counter()
r  = GenericParallelProcessor.fit_multiple_sequential(
    series, model_type='ARIMA', order=(1, 1, 0), steps=12, n_jobs=-1
)
t_par = time.perf_counter() - t0

print(f"Secuencial: {t_seq:.2f}s  |  Paralelo: {t_par:.2f}s  |  Speedup: {t_seq/t_par:.2f}x")
```

---

## Tests de paralelismo disponibles

| Archivo de test | Qué cubre |
|-----------------|-----------|
| `tests/test_arima.py` | Corrección de ARProcess, MAProcess, ARMAProcess, ARIMAProcess y ARIMAModel |
| `tests/test_models.py` | ARModel, MAModel, ARMAModel: recuperación de parámetros, `n_jobs`, residuales, IC |
| `tests/test_parallel_internal.py` | MLEOptimizer, ACF/PACF con `n_jobs`; propagación de parámetros |
| `tests/test_benchmark_parallelism.py` | Tiempos, speedup, codo; marcado `@pytest.mark.slow` |

```bash
# Todos los tests (excepto slow)
pytest tests/ -v

# Solo tests de rendimiento
pytest tests/test_benchmark_parallelism.py -v -m slow -s

# Todos incluyendo slow
pytest tests/ -v -m "slow or not slow"
```
