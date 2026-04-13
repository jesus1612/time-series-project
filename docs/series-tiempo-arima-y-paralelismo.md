# Series de tiempo: de lo básico a ARIMA, diagnóstico, benchmarks y paralelismo en TSLib

Guía de referencia para el monorepo **time-series-project**: fundamentos, gráficos útiles para incertidumbre y error, comparaciones de benchmark frente a **statsmodels**, y qué partes del código usan **Spark** frente al diagrama de flujo del ARIMA paralelo.

---

## 1. De lo simple a ARIMA

### Ruido blanco

Proceso \(\{\varepsilon_t\}\) incorrelado, media cero y varianza constante \(\sigma^2\). Base para construir MA y como componente de innovación en AR/MA/ARMA.

### MA(\(q\)) — media móvil

\(y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}\). Depende de \(q\) innovaciones pasadas.

### AR(\(p\)) — autorregresivo

\(y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t\). Depende de \(p\) valores pasados de la propia serie (bajo condiciones de estabilidad).

### ARMA(\(p,q\))

Combina AR(\(p\)) y MA(\(q\)) sobre la misma serie (estacionaria).

### Integración y ARIMA(\(p,d,q\))

Si la serie no es estacionaria, se aplican \(d\) diferencias: \(\Delta^d y_t\). Sobre la serie diferenciada (o sobre la original si \(d=0\)) se ajusta un ARMA(\(p,q\)), lo que define un **ARIMA(\(p,d,q\))** en niveles.

En TSLib, el orden \(d\) puede acotarse con pruebas de estacionariedad (p. ej. ADF/KPSS) en el flujo paralelo (`ParallelARIMAWorkflow`).

---

## 2. Gráficos para estimación e incertidumbre del error

### Serie temporal

Inspección de tendencia, estacionalidad explícita, cambios de varianza y valores atípicos antes de modelar.

### ACF y PACF

- **ACF**: autocorrelación de \(y_t\) con retardos; sugiere componentes MA o necesidad de diferenciación.
- **PACF**: correlación parcial; suele indicar orden AR.

En el enfoque **Box–Jenkins** clásico, ACF/PACF guían la elección inicial de \(p\) y \(q\). En `ParallelARIMAWorkflow`, la búsqueda de \((p,q)\) se hace principalmente por **rejilla** y criterios de información sobre ventanas; un paso explícito de identificación ACF/PACF previa al grid **no** es obligatorio en el código (ver tabla diagrama vs código más abajo).

### Residuos frente al tiempo

Comprueba si el modelo deja estructura temporal (patrones, heterocedasticidad).

### ACF de los residuos

Si el modelo captura la dinámica, los residuos deberían parecer ruido blanco; picos fuera de bandas de confianza indican posible sub-especificación.

### Gráfico Q–Q (normalidad aproximada)

Evalúa si los residuos son compatibles con normalidad (útil para intervalos y tests asintóticos).

### Intervalos de pronóstico

Muchos procedimientos reportan bandas (frecuentemente asumiendo gaussianidad de los errores de predicción). En la práctica, el **margen de error** out-of-sample se valida también con **RMSE**, **MAE** y **MAPE** sobre un conjunto de prueba o por **backtesting** en ventanas.

### Métricas de error en hold-out (backtesting)

- **RMSE**: penaliza grandes errores.
- **MAE**: error absoluto medio.
- **MAPE**: error porcentual (cuidado con valores cercanos a cero).

TSLib las calcula en métricas de evaluación y en el workflow de validación por ventanas fijas.

---

## 3. Benchmarks: qué se compara y con qué librería

### scikit-learn no incluye ARIMA

**scikit-learn (sklearn)** no proporciona un estimador ARIMA estándar. Las comparaciones de referencia en este proyecto usan **statsmodels**.

### Referencia externa: statsmodels

Implementación de referencia: `statsmodels.tsa.arima.model.ARIMA`, encapsulada en:

- [`time-series-library/tslib/benchmarks/arima_evaluation.py`](../time-series-library/tslib/benchmarks/arima_evaluation.py)

### Suite de benchmark en la aplicación Shiny

- [`tslib-shiny-app/features/benchmark/arima_benchmark.py`](../tslib-shiny-app/features/benchmark/arima_benchmark.py)

### Rutas que se comparan (tiempos y, cuando aplica, error frente a datos hold-out)

| Ruta | Descripción breve |
|------|---------------------|
| **TSLib lineal** | `ARIMAModel` en un proceso, orden fijo, sin workflow Spark. |
| **`ParallelARIMAWorkflow`** | Pipeline multi-paso con búsqueda en ventanas y Spark en las etapas paralelizables. |
| **Spark + statsmodels** | Pronóstico usando ARIMA de statsmodels dentro de tareas distribuidas (referencia bajo Spark). |
| **statsmodels local** | `ARIMA(...).fit()` en el mismo proceso Python (sin partición de tareas en el cluster). |

No existe una ruta “ARIMA de sklearn” en estos benchmarks.

### Cuándo compensa lo distribuido

Spark añade **overhead** (serialización, JVM, coordinación). Para series **cortas** o pocas tareas, el camino lineal o statsmodels local puede ser más rápido. A medida que crece el número de observaciones, ventanas o combinaciones \((p,d,q)\), el reparto de trabajo suele mejorar el tiempo total. Para análisis de umbrales \(N^\*\) (tamaño mínimo donde el speedup supera 1), véase la plantilla en [`sampler/ANALISIS_CRUCE.md`](../sampler/ANALISIS_CRUCE.md).

---

## 4. Qué está distribuido en el código y fidelidad al diagrama

### Dos patrones distintos

1. **`ParallelARIMAWorkflow`** ([`parallel_arima_workflow.py`](../time-series-library/tslib/spark/parallel_arima_workflow.py)): una serie larga; se crean **ventanas** y **combinaciones** \((p,d,q)\); el coste masivo es el **ajuste por tarea** (ventana × combinación). Pasos posteriores pueden paralelizarse por **ventana fija** (backtesting, diagnóstico).

2. **`GenericParallelProcessor`** ([`parallel_processor.py`](../time-series-library/tslib/spark/parallel_processor.py)): muchas **series** independientes; paralelismo por **grupo = serie** (`applyInPandas`).

### Tabla: diagrama (cajas naranjas = distribuido) vs `ParallelARIMAWorkflow`

| Elemento del diagrama | Comportamiento en el código |
|----------------------|-----------------------------|
| Granularidad / agrupación por ventanas (naranja) | La **lista de ventanas** se construye en el **driver** (no es en sí un job Spark). |
| ADF / KPSS | **Local** (`StationarityAnalyzer`). |
| ACF/PACF de identificación antes del grid | **No** hay paso explícito tipo Box–Jenkins previo; \((p,q)\) vienen de la **rejilla** y \(d\) de los tests. La **ACF de residuos** entra en el **diagnóstico** posterior. |
| Ventanas deslizantes según cómputo (naranja) | **Generación** de ventanas: local; **ajuste MLE + AICc** por tarea: **Spark** (`mapInPandas`). |
| Grid \((p,q)\) | **Local** (lista de combinaciones). |
| MLE por ventana + AICc (naranja) | **Spark** (tareas fila a fila). |
| Puntuación / ponderación entre ventanas (naranja) | **Agregación** sobre resultados ya recolectados (típicamente **pandas local**); el volumen suele ser manejable. |
| Backtesting y métricas RMSE/MAE/MAPE (naranja) | **Spark** (`mapInPandas` por ventana fija) en `_backtest_fixed_windows`. |
| Diagnóstico de residuos (naranja) | **Spark** (`mapInPandas` por ventana) en `_diagnose_residuals`. |

### Conclusión

El diagrama describe bien el **reparto de trabajo pesado** (muchas estimaciones y validaciones por ventana). Algunas cajas “naranjas” del diagrama son **conceptualmente** paralelas (pueden ejecutarse en distintos ejecutores); otras etapas son **baratas** y siguen siendo razonables en el driver.

---

## 5. Pestaña «Series temporales paralelas» (Shiny): dónde está cada cosa

Código de la vista: [`tslib-shiny-app/features/benchmark/ui.py`](../tslib-shiny-app/features/benchmark/ui.py), servidor: [`server.py`](../tslib-shiny-app/features/benchmark/server.py), figuras y métricas: [`arima_benchmark.py`](../tslib-shiny-app/features/benchmark/arima_benchmark.py).

Tras pulsar **Generar benchmark**, aparecen en orden:

| Orden | Qué es | Cómo interpretarlo |
|-------|--------|-------------------|
| 1 | **Tiempos de ajuste** (eje Y log) | Tiempo de *fit* frente a \(n\) en serie sintética. Cuatro curvas: TSLib lineal, **ParallelARIMAWorkflow**, Spark+statsmodels, statsmodels local. Escala log en Y para que una curva muy lenta no aplaste el resto. |
| 2 | **Speedup** | Cociente \(t_{\text{TSLib lineal}} / t_{\text{otro}}\). Por encima de 1, la otra ruta es más rápida que TSLib lineal en ese \(n\). Línea horizontal de referencia en 1×. |
| 3 | **Barras RMSE / MAE / MAPE** | Un número por método en el hold-out del CSV elegido (misma partición train/test). Compara el error global del tramo de prueba; no es por paso. |
| 4 | **ACF, PACF, residuos, Q-Q** | Diagnóstico sobre la **porción de entrenamiento** con ARIMA lineal TSLib (orden fijo del benchmark). Sirve para ver estructura residual, no para comparar Spark frente a statsmodels. |
| 5 | **\|error\| por horizonte** | Para cada paso \(h=1,\ldots,H\) fuera de muestra, \(\|y_{T+h} - \hat y_{T+h}\|\). Se dibujan **líneas** (no solo dos barras): **ParallelARIMAWorkflow** en línea más gruesa, más TSLib lineal, statsmodels ref. y, si Spark responde, Spark+statsmodels. Así el gráfico centra la lectura en **tu pipeline paralelo** frente a referencias; antes solo comparaba dos rutas “locales” y por eso resultaba confuso. Si falta Spark, solo se muestran las rutas que se pudieron calcular. |
| 6 | **Secuencial vs n_jobs en ARIMAModel** | Contrasta `n_jobs=1` y `n_jobs=-1` en el **mismo** `ARIMAModel` de TSLib (paralelismo interno del ajuste). **No** es el mismo objeto que `ParallelARIMAWorkflow` (sección 1–5). |
| Texto | **Resumen** | Cruces \(N^\*\) donde el speedup respecto a TSLib lineal llega a ≥ 1, más métricas hold-out por método. El bloque usa estilos claros sobre fondo oscuro (clase `bench-summary-pre`). |

**Nota:** Si en Spark aparece un error tipo sesión inválida (`'NoneType' object has no attribute 'sc'`), la sesión se descarta y las rutas que dependen de Spark quedan como error o ausentes hasta reiniciar JVM/Spark de forma estable.

---

## 6. Referencias de código

| Tema | Ruta |
|------|------|
| Workflow ARIMA paralelo | `time-series-library/tslib/spark/parallel_arima_workflow.py` |
| Procesador genérico AR/MA/ARMA/ARIMA en Spark | `time-series-library/tslib/spark/parallel_processor.py` |
| Benchmark UI (tiempos / métricas) | `tslib-shiny-app/features/benchmark/arima_benchmark.py` |
| statsmodels helpers | `time-series-library/tslib/benchmarks/arima_evaluation.py` |
| Matemática y pasos del workflow (docs internos) | `time-series-library/docs/MATHEMATICS_BY_MODEL.md` |

---

## 7. Puesta en marcha rápida del monorepo

Desde la raíz del repositorio:

```bash
make install-all   # instala TSLib (dev) y la UI con TSLib editable
make up            # instala si hace falta y arranca Shiny (puerto 8000)
```

Ver también el [`README.md`](../README.md) del repositorio.
