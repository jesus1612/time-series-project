# TSLib - Time Series Library

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/genaromelgar/time-series-library)

Una librería completa de análisis de series de tiempo con implementación de modelos ARIMA desde cero, diseñada con principios de programación orientada a objetos y soporte opcional para procesamiento distribuido con PySpark.

## Características

- ✅ **4 Modelos Independientes**: AR, MA, ARMA y ARIMA implementados desde cero matemáticamente
- ✅ **Implementación matemática pura**: Sin dependencias de librerías externas de modelos (statsmodels, etc.)
- ✅ **Arquitectura orientada a objetos**: Diseño limpio siguiendo principios SOLID con jerarquía de clases bien definida
- ✅ **Selección automática de orden**: Cada modelo tiene su propio selector optimizado (PACF para AR, ACF para MA, etc.)
- ✅ **APIs intuitivas**: Interfaces de alto nivel siguiendo convenciones de scikit-learn para cada modelo
- ✅ **Documentación matemática completa**: Fundamentos, ecuaciones y casos de uso documentados en español
- ✅ **Validación de datos integrada**: Sistema centralizado de validación y diagnóstico de calidad
- ✅ **Soporte PySpark opcional**: Procesamiento paralelo de múltiples series con Pandas UDF
- ✅ **Tests comprehensivos**: Alta cobertura de código con unit, integration y performance tests
- ✅ **Benchmarks de rendimiento**: Comparación detallada entre implementación normal y Spark
- ✅ **Compatibilidad amplia**: Soporte para Python 3.9-3.12

## Estado actual y notas

- **Librería core**: Este repo es la parte de biblioteca del proyecto de titulación; se conecta con la interfaz gráfica (tslib-shiny-app). Toda la lógica de modelos, validación, ACF/PACF y flujo paralelo con Spark vive aquí.
- **Procesamiento paralelo**: Los ejemplos y el flujo con Spark están activos: `ParallelARIMAProcessor`, `ParallelARIMAWorkflow`, `examples/spark_parallel_arima.py` y `examples/spark_parallel_arima_workflow_demo.py` inicializan Spark y ejecutan el flujo completo. Requieren Java 17+ y `pip install -r requirements-spark.txt`.
- **Documentación matemática**: La referencia única por modelo (solo matemática: ecuaciones, condiciones, identificación y cómo lo implementamos) está en **[docs/MATHEMATICS_BY_MODEL.md](docs/MATHEMATICS_BY_MODEL.md)**. Los documentos en `docs/modelos/` amplían con narrativa y casos de uso en español.
- **Desactualizado**: Si en la app o en otro repo se usan APIs antiguas de esta librería, conviene alinear con los exports actuales de `tslib` y `tslib.spark` (ver sección Documentación más abajo).

## Requisitos y Compatibilidad

- **Python**: 3.9, 3.10, 3.11, 3.12
- **Java**: 17+ (requerido para funcionalidad PySpark)
- **Dependencias principales**: NumPy ≥1.24.0, SciPy ≥1.10.0, Pandas ≥1.5.0, Matplotlib ≥3.6.0
- **PySpark**: 4.0.1 (opcional, para procesamiento distribuido)
- **Sistema operativo**: Windows, macOS, Linux

### Instalación de Java 17+

**macOS (con Homebrew):**
```bash
brew install openjdk@17
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install openjdk-17-jdk
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
```

**Windows:**
1. Descargar desde [Adoptium](https://adoptium.net/temurin/releases/?version=17)
2. Ejecutar el instalador
3. Configurar variable de entorno `JAVA_HOME`

**Verificar instalación:**
```bash
java -version  # Debe mostrar Java 17 o superior
```

## Instalación

### Usando Makefile (Recomendado)

```bash
# Verificar requisitos
make check-version  # Verifica Python 3.9+
make check-java     # Verifica Java 17+ (para Spark)

# Instalación básica
make install

# Con soporte PySpark (requiere Java 17+)
make install-spark

# Con herramientas de desarrollo
make install-dev

# Instalar Java 17+ (si es necesario)
make install-java-macos    # macOS
make install-java-linux    # Linux
make install-java-windows  # Windows

# Ver todas las opciones
make help
```

### Instalación manual

```bash
# Instalación básica
pip install -r requirements.txt
pip install -e .

# Con soporte PySpark
pip install -r requirements.txt
pip install -r requirements-spark.txt
pip install -e .

# Con herramientas de desarrollo
pip install -e ".[dev]"
```

## Uso Rápido

### AR Model - Modelo Autoregresivo

```python
from tslib import ARModel

# Create AR model with automatic order selection
model = ARModel(auto_select=True, max_order=5)
model.fit(data)

# Make predictions with confidence intervals
forecast, conf_int = model.predict(steps=10, return_conf_int=True)

# View diagnostics
model.plot_diagnostics()
print(model.summary())
```

**Cuándo usar AR:** Series estacionarias donde el valor actual depende de valores pasados. Común en economía y finanzas.

### MA Model - Modelo de Media Móvil

```python
from tslib import MAModel

# Create MA model with automatic order selection
model = MAModel(auto_select=True, max_order=5)
model.fit(data)

# Predictions (converge to mean after q steps)
forecast = model.predict(steps=10)
model.plot_forecast(steps=10)
```

**Cuándo usar MA:** Series donde los shocks tienen efectos de corto plazo. Útil para modelar errores de pronóstico.

### ARMA Model - Combinación AR + MA

```python
from tslib import ARMAModel

# Create ARMA model with automatic order selection
model = ARMAModel(auto_select=True, max_ar=5, max_ma=5, criterion='aic')
model.fit(data)

# Make predictions
forecast, conf_int = model.predict(steps=10, return_conf_int=True)
print(model.summary())
```

**Cuándo usar ARMA:** Series estacionarias con estructura compleja que combina autocorrelación y efectos de shocks.

### ARIMA Model - Para Series No Estacionarias

```python
from tslib import ARIMAModel

# Create ARIMA model (handles non-stationary data)
model = ARIMAModel(auto_select=True, max_p=5, max_d=2, max_q=5)
model.fit(data)  # Automatically applies differencing if needed

# Make predictions
forecast = model.predict(steps=10)
model.plot_diagnostics()
```

**Cuándo usar ARIMA:** Series con tendencia o no estacionarias. Aplica diferenciación automáticamente.

### Todos los Modelos con PySpark (Procesamiento Paralelo)

TSLib proporciona `GenericParallelProcessor`, una interfaz unificada para entrenar múltiples series de tiempo de forma simultánea a través de un cluster Spark, soportando modelos **AR, MA, ARMA y ARIMA**.

```python
from pyspark.sql import SparkSession
from tslib.spark import GenericParallelProcessor

spark = SparkSession.builder.appName("TimeSeriesAnalysis").getOrCreate()

# DataFrame con múltiples series agrupadas por 'group_id'
df = spark.read.parquet("multiple_series.parquet")

# 1. Ejecutar múltiples modelos AR(2) en paralelo
ar_processor = GenericParallelProcessor(model_type='AR', spark=spark)
resultados_ar = ar_processor.fit_multiple(df, group_col="group_id", value_col="value", order=2, steps=10)

# 2. Ejecutar múltiples modelos MA(3) en paralelo
ma_processor = GenericParallelProcessor(model_type='MA', spark=spark)
resultados_ma = ma_processor.fit_multiple(df, group_col="group_id", value_col="value", order=3, steps=10)

# 3. Ejecutar múltiples modelos ARMA(1,1) en paralelo
arma_processor = GenericParallelProcessor(model_type='ARMA', spark=spark)
resultados_arma = arma_processor.fit_multiple(df, group_col="group_id", value_col="value", order=(1, 1), steps=10)

# 4. Ejecutar múltiples modelos ARIMA(1,1,1) en paralelo
arima_processor = GenericParallelProcessor(model_type='ARIMA', spark=spark)
resultados_arima = arima_processor.fit_multiple(df, group_col="group_id", value_col="value", order=(1, 1, 1), steps=10)

# Ver los pronósticos generados
resultados_ar.show()
```

## Guía de Selección de Modelos

| Modelo | Requiere Estacionariedad | Identificación | Mejor para |
|--------|-------------------------|----------------|------------|
| **AR(p)** | Sí | PACF se corta en lag p | Series con autocorrelación, persistencia |
| **MA(q)** | Sí | ACF se corta en lag q | Shocks transitorios, errores de pronóstico |
| **ARMA(p,q)** | Sí | ACF y PACF decaen | Estructuras complejas estacionarias |
| **ARIMA(p,d,q)** | No | Tests ADF/KPSS + ACF/PACF | Series con tendencia, datos no estacionarios |

### Flujo de Trabajo Recomendado

1. **Análisis exploratorio**: Visualizar la serie, verificar patrones
2. **Test de estacionariedad**: Si no es estacionaria → usar ARIMA; si es estacionaria → continuar
3. **Analizar ACF/PACF**:
   - PACF se corta → considerar AR
   - ACF se corta → considerar MA
   - Ambos decaen → considerar ARMA
4. **Ajustar modelo**: Usar auto_select=True para selección automática
5. **Diagnóstico**: Verificar residuos, ACF de residuos, Q-Q plot
6. **Predicción**: Generar forecasts con intervalos de confianza

## Sistema de Carpetas y Organización

```
time-series-library/
├── tslib/                      # Librería principal
│   ├── __init__.py            # Exports: ARModel, MAModel, ARMAModel, ARIMAModel
│   │
│   ├── core/                  # Algoritmos fundamentales (bajo nivel)
│   │   ├── base.py           # Clases base abstractas (BaseModel, BaseEstimator, etc.)
│   │   ├── arima.py          # ARProcess, MAProcess, ARMAProcess, ARIMAProcess
│   │   ├── optimization.py   # MLEOptimizer - Maximum Likelihood Estimation
│   │   ├── acf_pacf.py       # Cálculo de ACF/PACF para identificación
│   │   └── stationarity.py   # Tests ADF y KPSS
│   │
│   ├── models/                # Interfaces de alto nivel (APIs públicas)
│   │   ├── ar_model.py       # ARModel - API de usuario para AR
│   │   ├── ma_model.py       # MAModel - API de usuario para MA
│   │   ├── arma_model.py     # ARMAModel - API de usuario para ARMA
│   │   ├── arima_model.py    # ARIMAModel - API de usuario para ARIMA
│   │   └── selection.py      # Selectores automáticos de orden
│   │
│   ├── preprocessing/         # Transformaciones y validación
│   │   ├── validation.py     # DataValidator - validación centralizada
│   │   └── transformations.py # Diferenciación, log, Box-Cox
│   │
│   ├── metrics/               # Métricas de evaluación
│   │   └── evaluation.py     # AIC, BIC, RMSE, MAE, análisis de residuos
│   │
│   ├── spark/                 # Integración PySpark (opcional)
│   │   ├── parallel_arima.py           # Procesamiento distribuido (ParallelARIMAProcessor)
│   │   ├── parallel_arima_workflow.py  # Workflow 11 pasos (ParallelARIMAWorkflow)
│   │   └── core.py                     # Utilidades Spark
│   │
│   └── utils/                 # Funciones utilitarias
│       └── checks.py         # Verificaciones del sistema
│
├── docs/                      # Documentación
│   ├── justificacion_arquitectura_tslib.txt   # Arquitectura / validaciones / fallos (texto plano)
│   ├── justificacion_entrada_datos_tslib.txt # Entrada, limpieza, transformación, imputación (texto plano)
│   ├── inventario_y_plan_capacidades_tslib.txt  # Capacidades, Spark, paralelismo, plan de comparativas
│   ├── MATHEMATICS_BY_MODEL.md  # Referencia matemática única por modelo (ecuaciones, condiciones, implementación)
│   ├── modelos/              # Documentación extendida por modelo (español)
│   │   ├── AR.md            # Fundamentos del modelo AR
│   │   ├── MA.md            # Fundamentos del modelo MA
│   │   ├── ARMA.md          # Fundamentos del modelo ARMA
│   │   └── ARIMA.md         # Fundamentos del modelo ARIMA
│   ├── mathematical_foundations.md  # Fundamentos generales (ACF, PACF, MLE, métricas)
│   └── IMPLEMENTATION_SUMMARY.md
│
├── examples/                  # Ejemplos de uso
│   ├── ar_example.py         # Ejemplos comprehensivos de AR
│   ├── ma_example.py         # Ejemplos comprehensivos de MA
│   ├── arma_example.py       # Ejemplos comprehensivos de ARMA
│   ├── basic_arima.py                    # Ejemplos de ARIMA
│   ├── spark_parallel_arima.py          # Procesamiento paralelo múltiples series (PySpark)
│   └── spark_parallel_arima_workflow_demo.py  # Demo workflow 11 pasos con Spark
│
├── tests/                     # Suite de tests
│   ├── test_arima.py         # Tests de modelos
│   ├── test_performance_benchmark.py
│   └── ...
│
└── README.md                  # Este archivo
```

## Integración y Arquitectura

### Capa 1: Algoritmos Core (tslib/core/)

Los **algoritmos fundamentales** implementan la matemática pura:

- `ARProcess`, `MAProcess`, `ARMAProcess`, `ARIMAProcess`: Implementación matemática desde cero
- `MLEOptimizer`: Estimación de parámetros usando Maximum Likelihood
- `ACFCalculator`, `PACFCalculator`: Análisis de correlación para identificación
- Tests de estacionariedad: ADF y KPSS

**Estos no se usan directamente**, sino a través de las APIs de alto nivel.

### Capa 2: APIs de Alto Nivel (tslib/models/)

Las **interfaces de usuario** proporcionan funcionalidad completa:

```python
from tslib import ARModel, MAModel, ARMAModel, ARIMAModel

# Cada modelo incluye:
# - Auto-selección de orden
# - Validación de datos
# - Análisis exploratorio automático
# - Diagnósticos completos
# - Visualizaciones
```

**Flujo interno al llamar `model.fit(data)`:**

1. **Validación**: `DataValidator` verifica calidad de datos
2. **Análisis exploratorio**: Calcula ACF/PACF, tests de estacionariedad
3. **Selección de orden**: 
   - AR → `AROrderSelector` usa PACF
   - MA → `MAOrderSelector` usa ACF
   - ARMA → `ARMAOrderSelector` usa grid search
   - ARIMA → `ARIMAOrderSelector` usa tests de raíz unitaria
4. **Ajuste**: Llama al proceso core correspondiente
5. **Diagnóstico**: Calcula residuos y métricas

### Capa 3: Selectores de Orden (tslib/models/selection.py)

Sistema modular de **selección automática de orden**:

```python
# Cada selector usa la estrategia óptima para su modelo
AROrderSelector → Analiza PACF (se corta en lag p)
MAOrderSelector → Analiza ACF (se corta en lag q)
ARMAOrderSelector → Grid search con AIC/BIC
ARIMAOrderSelector → Tests ADF/KPSS + ARMA selection
```

### Integración Vertical

```
Usuario
   ↓
ARModel.fit(data) ← API de alto nivel
   ↓
AROrderSelector.select(data) ← Selección automática
   ↓
ARProcess.fit(data) ← Algoritmo core
   ↓
MLEOptimizer.estimate() ← Optimización matemática
```

### Validación Centralizada

Todos los modelos comparten el mismo sistema de validación:

```python
DataValidator verifica:
├── Longitud mínima de datos
├── Valores faltantes (con límites configurables)
├── Valores infinitos
├── Detección de outliers (IQR, z-score, modified z-score)
├── Verificación de estacionariedad
└── Detección de tendencias y estacionalidad
```

### Imputación y Valores Faltantes

La librería no ajusta modelos sobre NaN; los datos deben limpiarse antes. `DataValidator` en `tslib/preprocessing/validation.py` ofrece:

- **Límite configurable**: `max_missing_ratio` (por defecto 0.1). Si el ratio de faltantes supera el límite, la validación falla.
- **Limpieza opcional**: `DataValidator.clean_data(data, method=...)` con métodos:
  - `'interpolate'`: interpolación lineal.
  - `'forward_fill'`: rellenar hacia adelante (ffill).
  - `'backward_fill'`: rellenar hacia atrás (bfill).
  - `'drop'`: eliminar observaciones con faltantes.

En pipelines Spark (`tslib/spark/`), las series se limpian con `dropna()` antes del ajuste; si quedan demasiados NaN o la serie es inválida, se devuelven pronósticos/métricas como NaN.

### Integración con TSLib Shiny App

Este repositorio es la **librería core** del proyecto de titulación. La aplicación web **tslib-shiny-app** (repositorio independiente) consume TSLib para ofrecer un flujo guiado: carga de datos, visualización, selección de modelo AR/MA/ARMA/ARIMA, ejecución y resultados. La app instala TSLib en modo editable (`pip install -e <ruta/time-series-library>`). Toda la lógica de modelos, validación, ACF/PACF y opcionalmente el flujo paralelo ARIMA con Spark proviene de este repositorio. Ver en la app: `INTEGRATION_README.md` y `services/tslib_service.py`.

## Uso de la Librería

### Instalación y Setup

```bash
# 1. Clonar o descargar el repositorio
git clone https://github.com/genaromelgar/time-series-library.git
cd time-series-library

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar
pip install -e .

# 4. Para desarrollo (incluye pytest, black, etc.)
pip install -e ".[dev]"

# 5. Para soporte Spark (opcional)
pip install -r requirements-spark.txt
```

### Uso Básico Paso a Paso

#### 1. Importar y Cargar Datos

```python
import numpy as np
import pandas as pd
from tslib import ARModel, MAModel, ARMAModel, ARIMAModel

# Cargar datos
data = pd.read_csv("tu_serie.csv")["valor"]
# O generar datos sintéticos
data = np.random.randn(200).cumsum()  # Random walk
```

#### 2. Elegir el Modelo Apropiado

```python
# Si no sabes qué modelo usar, prueba ARIMA con auto-selección
model = ARIMAModel(auto_select=True, validation=True)
model.fit(data)

# El modelo automáticamente:
# - Detecta si necesita diferenciación (determina d)
# - Selecciona órdenes p y q óptimos
# - Valida la calidad de los datos
# - Realiza análisis exploratorio
```

#### 3. Análisis Exploratorio

```python
# Obtener resultados del análisis
analysis = model.get_exploratory_analysis()

print("Estacionariedad:", analysis['stationarity']['is_stationary'])
print("Orden sugerido:", model.order)
print("AIC:", model._fitted_params['aic'])
```

#### 4. Predicción

```python
# Predecir con intervalos de confianza
forecast, (lower, upper) = model.predict(steps=10, return_conf_int=True)

print("Pronósticos:", forecast)
print("Límite inferior 95%:", lower)
print("Límite superior 95%:", upper)
```

#### 5. Diagnóstico

```python
# Visualizar diagnósticos
model.plot_diagnostics()  # Muestra: residuos, Q-Q plot, ACF, etc.

# Obtener diagnóstico numérico
diag = model.get_residual_diagnostics()
print("Ljung-Box p-value:", diag['ljung_box_test']['p_value'])
# p > 0.05 indica residuos son ruido blanco (bueno)
```

#### 6. Visualizar Pronóstico

```python
model.plot_forecast(steps=20, include_data=True)
```

### Ejemplos Específicos por Modelo

#### AR Model - Series Económicas

```python
from tslib import ARModel

# Para series con persistencia (tasas de interés, inflación)
model = ARModel(
    auto_select=True,     # Selección automática de orden p
    max_order=5,          # Considerar hasta AR(5)
    selection_method='pacf',  # Usar PACF cutoff
    validation=True       # Validar calidad de datos
)

model.fit(data)
print(f"Orden seleccionado: AR({model.order})")

# AR forecasts convergen a la media
forecast_10 = model.predict(steps=10)
```

#### MA Model - Errores de Pronóstico

```python
from tslib import MAModel

# Para modelar efectos de shocks de corto plazo
model = MAModel(
    auto_select=True,
    max_order=5,
    selection_method='acf'  # Usar ACF cutoff
)

model.fit(errores_pronostico)

# MA forecasts convergen rápidamente a la media
# Después de q pasos, pronóstico = media
forecast = model.predict(steps=20)
```

#### ARMA Model - Series Financieras

```python
from tslib import ARMAModel

# Para retornos de activos (combinan persistencia y shocks)
model = ARMAModel(
    auto_select=True,
    max_ar=3,
    max_ma=3,
    criterion='aic'  # Usar AIC para selección
)

model.fit(retornos)
print(f"Modelo: ARMA({model.order[0]}, {model.order[1]})")

# Evaluar pronóstico
actual = retornos_test
predicted = model.predict(steps=len(actual))
metrics = model.evaluate_forecast(actual, predicted)
print(f"RMSE: {metrics['rmse']}")
```

#### ARIMA Model - Series con Tendencia

```python
from tslib import ARIMAModel

# Para precios, PIB, población (series no estacionarias)
model = ARIMAModel(
    auto_select=True,
    max_p=3,
    max_d=2,  # Permitir hasta 2 diferenciaciones
    max_q=3
)

model.fit(precios)  # No necesita pre-procesamiento
print(f"Orden: ARIMA{model.order}")  # e.g., (1,1,1)

# Pronósticos pueden tener tendencia
forecast, ci = model.predict(steps=30, return_conf_int=True)
```

### Comparación de Modelos

```python
from tslib import ARModel, MAModel, ARMAModel

# Comparar diferentes modelos para la misma serie
modelos = [
    ('AR', ARModel(order=2)),
    ('MA', MAModel(order=2)),
    ('ARMA', ARMAModel(order=(1, 1)))
]

resultados = []
for nombre, modelo in modelos:
    modelo.fit(data)
    aic = modelo._fitted_params['aic']
    bic = modelo._fitted_params['bic']
    resultados.append((nombre, aic, bic))
    print(f"{nombre}: AIC={aic:.2f}, BIC={bic:.2f}")

# Seleccionar modelo con menor AIC
mejor = min(resultados, key=lambda x: x[1])
print(f"\nMejor modelo: {mejor[0]}")
```

### Train-Test Split y Validación

```python
# Dividir datos
train_size = int(0.8 * len(data))
train, test = data[:train_size], data[train_size:]

# Entrenar
model = ARIMAModel(auto_select=True)
model.fit(train)

# Predecir período de prueba
predictions = model.predict(steps=len(test))

# Evaluar
metrics = model.evaluate_forecast(test, predictions)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"MAPE: {metrics['mape']:.2f}%")
```

## Arquitectura del Proyecto

### Estructura Modular Actualizada

```
tslib/
├── core/                   # Algoritmos fundamentales (bajo nivel)
│   ├── base.py            # Clases base: BaseModel, BaseEstimator, BaseTransformer
│   ├── arima.py           # ARProcess, MAProcess, ARMAProcess, ARIMAProcess
│   ├── optimization.py    # MLEOptimizer - Maximum Likelihood Estimation
│   ├── acf_pacf.py        # ACFCalculator, PACFCalculator, ACFPACFAnalyzer
│   └── stationarity.py    # ADFTest, KPSSTest, StationarityAnalyzer
│
├── models/                 # Interfaces de alto nivel (APIs públicas)
│   ├── ar_model.py        # ARModel - Usuario final
│   ├── ma_model.py        # MAModel - Usuario final
│   ├── arma_model.py      # ARMAModel - Usuario final
│   ├── arima_model.py     # ARIMAModel - Usuario final
│   └── selection.py       # AROrderSelector, MAOrderSelector, ARMAOrderSelector, ARIMAOrderSelector
│
├── preprocessing/          # Transformaciones y validación
│   ├── validation.py      # DataValidator, DataQualityReport
│   └── transformations.py # DifferencingTransformer, LogTransformer, BoxCoxTransformer
│
├── metrics/                # Métricas de evaluación
│   └── evaluation.py      # InformationCriteria, ForecastMetrics, ResidualAnalyzer
│
├── spark/                  # Integración PySpark (opcional)
│   ├── parallel_arima.py  # Procesamiento distribuido con Pandas UDF
│   └── core.py            # SparkDataConverter, SparkMathOperations
│
└── utils/                  # Funciones utilitarias
    └── checks.py          # check_spark_availability, verificaciones del sistema
```

### Principios de Diseño OOP

La librería sigue una arquitectura orientada a objetos basada en principios SOLID:

- **Single Responsibility**: Cada clase tiene una responsabilidad específica
- **Open/Closed**: Extensible sin modificar código existente
- **Liskov Substitution**: Jerarquía de herencia consistente
- **Interface Segregation**: Interfaces específicas para diferentes funcionalidades
- **Dependency Inversion**: Dependencia de abstracciones, no implementaciones

### Jerarquía de Clases

```
BaseModel (ABC)
├── TimeSeriesModel (ABC)
│   ├── ARProcess           # Core: Implementación matemática AR
│   ├── MAProcess           # Core: Implementación matemática MA
│   ├── ARMAProcess         # Core: Implementación matemática ARMA
│   └── ARIMAProcess        # Core: Implementación matemática ARIMA
│
└── High-level APIs (usan los procesos core)
    ├── ARModel             # API pública para AR
    ├── MAModel             # API pública para MA
    ├── ARMAModel           # API pública para ARMA
    └── ARIMAModel          # API pública para ARIMA

BaseEstimator (ABC)
└── MLEOptimizer            # Optimización MLE compartida

BaseTransformer (ABC)
└── DifferencingTransformer, LogTransformer

OrderSelector (ABC)
├── AROrderSelector         # Selección usando PACF
├── MAOrderSelector         # Selección usando ACF
├── ARMAOrderSelector       # Grid search con AIC/BIC
└── ARIMAOrderSelector      # Tests de raíz unitaria + ARMA

SparkEnabled (Mixin)
└── ParallelARIMAProcessor  # Procesamiento distribuido
```

## Componentes Matemáticos

### ARIMA(p, d, q)

- **AR (AutoRegressive)**: y_t = c + Σφ_i * y_{t-i} + ε_t
- **MA (Moving Average)**: y_t = μ + ε_t + Σθ_i * ε_{t-i}
- **I (Integration)**: Diferenciación de orden d

### Tests de Estacionariedad

- Augmented Dickey-Fuller (ADF) test
- KPSS test

### Métricas de Evaluación

- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- RMSE, MAE, MAPE

## Desarrollo

### Comandos Makefile

```bash
# Configuración del entorno de desarrollo
make dev-setup

# Ejecutar tests
make test                    # Todos los tests
make test-coverage          # Con reporte de cobertura
make test-spark             # Tests específicos de Spark
make benchmark              # Benchmarks de rendimiento

# Herramientas de desarrollo
make format                 # Formatear código con black
make lint                   # Linter con flake8
make clean                  # Limpiar archivos temporales

# Ejemplos
make examples               # Ejecutar scripts de ejemplo
```

### Comandos manuales

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest tests/ -v

# Tests con cobertura
pytest tests/ --cov=tslib --cov-report=html

# Formatear código
black tslib/ tests/ examples/

# Linter
flake8 tslib/ tests/ examples/
```

## Benchmarks de Rendimiento

La librería incluye tests comprehensivos de rendimiento que comparan la implementación normal vs Spark:

### Métricas Evaluadas

- **Tiempo de ejecución**: Para diferentes tamaños de datasets (1-100 series)
- **Escalabilidad**: Comportamiento con múltiples series temporales
- **Uso de memoria**: Comparación Python vs Spark
- **Precisión**: Verificación de resultados numéricamente idénticos
- **Overhead**: Costo de inicialización de Spark
- **Speedup**: Factor de aceleración y eficiencia

### Ejecutar Benchmarks

```bash
# Benchmarks completos (requiere Java 17+)
make benchmark

# O directamente
python tests/test_performance_benchmark.py
```

**Nota**: Los benchmarks de Spark requieren Java 17+ instalado y configurado correctamente.

### Resultados Típicos

- **Punto de equilibrio**: Spark se vuelve más eficiente con 25+ series
- **Precisión**: Resultados numéricamente idénticos entre implementaciones
- **Escalabilidad**: Spark muestra mejor escalabilidad para datasets grandes

## Documentación

### Documentación Técnica

#### Referencia matemática por modelo (recomendada)

- **[MATHEMATICS_BY_MODEL.md](docs/MATHEMATICS_BY_MODEL.md)** — Documento único solo matemático: ecuaciones, condiciones de estacionariedad/invertibilidad, identificación (ACF/PACF) y cómo lo implementa TSLib por modelo (AR, MA, ARMA, ARIMA). Incluye el rol del procesamiento paralelo (misma matemática, ejecución distribuida).

#### Documentación extendida por modelo (español)

Cada modelo tiene documentación ampliada de fundamentos y casos de uso:

- **[AR - Modelo Autoregresivo](docs/modelos/AR.md)**
  - Definición: y_t = c + φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t
  - Identificación: PACF se corta en lag p
  - Casos de uso: Economía, finanzas, series con persistencia

- **[MA - Modelo de Media Móvil](docs/modelos/MA.md)**
  - Definición: y_t = μ + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
  - Identificación: ACF se corta en lag q
  - Casos de uso: Errores de pronóstico, shocks transitorios

- **[ARMA - Modelo AR + MA](docs/modelos/ARMA.md)**
  - Combinación de componentes AR y MA
  - Identificación: ACF y PACF decaen gradualmente
  - Casos de uso: Series con estructura compleja estacionaria

- **[ARIMA - ARMA Integrado](docs/modelos/ARIMA.md)**
  - ARMA con diferenciación para series no estacionarias
  - Identificación: Tests ADF/KPSS + ACF/PACF
  - Casos de uso: Series con tendencia, precios, PIB

#### Otros Documentos

- [justificacion_arquitectura_tslib.txt](docs/justificacion_arquitectura_tslib.txt) — Arquitectura, validaciones y fallos (texto plano para memoria/ensayo)
- [justificacion_entrada_datos_tslib.txt](docs/justificacion_entrada_datos_tslib.txt) — Entrada de datos, limpieza, transformación e imputación; inicio exacto del flujo en código (texto plano)
- [inventario_y_plan_capacidades_tslib.txt](docs/inventario_y_plan_capacidades_tslib.txt) — Inventario de modelos, validación, Spark, fallos, paralelismo y plan de benchmarks (texto plano)
- [Mathematical Foundations](docs/mathematical_foundations.md) - Fundamentos matemáticos generales
- [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) - Resumen de implementación
- [JUSTIFICACION_TECNICA.txt](JUSTIFICACION_TECNICA.txt) - Justificación académica del diseño OOP

### API Reference Completa

#### AR Model

```python
from tslib import ARModel

model = ARModel(
    order=None,              # Orden AR (p), None para auto-selección
    trend='c',               # 'c' (constante) o 'nc' (sin constante)
    auto_select=True,        # Selección automática de orden
    max_order=5,             # Orden máximo a considerar
    selection_method='pacf', # 'pacf' o 'aic'
    validation=True          # Validar calidad de datos
)

model.fit(data)
forecast = model.predict(steps=10, return_conf_int=True)
model.plot_diagnostics()
print(model.summary())
```

#### MA Model

```python
from tslib import MAModel

model = MAModel(
    order=None,              # Orden MA (q), None para auto-selección
    auto_select=True,        # Selección automática de orden
    max_order=5,             # Orden máximo a considerar
    selection_method='acf',  # 'acf' o 'aic'
    validation=True
)

model.fit(data)
forecast = model.predict(steps=10, return_conf_int=True)
# Nota: MA forecasts convergen a la media después de q steps
```

#### ARMA Model

```python
from tslib import ARMAModel

model = ARMAModel(
    order=None,              # Tupla (p, q), None para auto-selección
    trend='c',               # 'c' o 'nc'
    auto_select=True,
    max_ar=5,                # Orden AR máximo
    max_ma=5,                # Orden MA máximo
    criterion='aic',         # 'aic' o 'bic'
    validation=True
)

model.fit(data)
forecast = model.predict(steps=10, return_conf_int=True)
```

#### ARIMA Model

```python
from tslib import ARIMAModel

model = ARIMAModel(
    order=None,              # Tupla (p, d, q), None para auto-selección
    trend='c',
    auto_select=True,
    max_p=5,                 # Orden AR máximo
    max_d=2,                 # Orden de diferenciación máximo
    max_q=5,                 # Orden MA máximo
    validation=True
)

model.fit(data)  # Maneja series no estacionarias automáticamente
forecast = model.predict(steps=10, return_conf_int=True)
```

#### Métodos Comunes a Todos los Modelos

```python
# Después de fit():
model.predict(steps=10, return_conf_int=True)  # Predicción
model.get_residuals()                           # Obtener residuos
model.get_fitted_values()                       # Valores ajustados
model.summary()                                 # Resumen del modelo
model.plot_diagnostics()                        # Gráficos de diagnóstico
model.plot_forecast(steps=20)                   # Gráfico de pronóstico
model.get_exploratory_analysis()                # Análisis exploratorio
model.get_residual_diagnostics()                # Diagnósticos de residuos
model.evaluate_forecast(actual, predicted)      # Métricas de evaluación
```

#### Procesamiento Paralelo con Spark

```python
from tslib.spark import ParallelARIMAProcessor

processor = ParallelARIMAProcessor()

# Procesar múltiples series en paralelo
results = processor.fit_multiple_arima(
    df=spark_df,
    group_column='series_id',
    value_column='value',
    order=(1, 1, 1)
)
```

### Ejemplos Disponibles

```bash
# Ejecutar todos los ejemplos
make examples

# Ejemplos individuales
python examples/ar_example.py         # AR: 4 ejemplos completos
python examples/ma_example.py         # MA: 5 ejemplos completos
python examples/arma_example.py       # ARMA: 5 ejemplos completos
python examples/basic_arima.py        # ARIMA: Ejemplo básico

# Con Spark
python examples/spark_parallel_arima.py
```

Cada archivo de ejemplo incluye:
- Generación de datos sintéticos
- Múltiples casos de uso
- Auto-selección de orden
- Comparación de modelos
- Evaluación train-test
- Visualizaciones completas

## Contribución

### Estructura del Proyecto para Desarrolladores

1. **Core**: Implementa algoritmos matemáticos fundamentales
2. **Models**: Proporciona interfaces de alto nivel
3. **Preprocessing**: Maneja transformaciones de datos
4. **Spark**: Integración con procesamiento distribuido
5. **Tests**: Suite completa de pruebas unitarias e integración

### Guías de Contribución

1. Seguir principios SOLID en el diseño de clases
2. Mantener cobertura de tests >80%
3. Documentar métodos públicos en inglés
4. Usar type hints para mejor mantenibilidad
5. Ejecutar `make lint` antes de commits

### Testing

```bash
# Suite completa de tests
make full-test

# Tests específicos
make test-spark
make benchmark
```

## Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

## Autores

- **Genaro Melgar** - ESCOM, Instituto Politécnico Nacional
- **Aarón Merlos** - ESCOM, Instituto Politécnico Nacional
- **Luis Miranda** - ESCOM, Instituto Politécnico Nacional
  - Implementación ARIMA desde cero
  - Arquitectura orientada a objetos
  - Integración PySpark
  - Tests de rendimiento

## Agradecimientos

- Instituto Politécnico Nacional - ESCOM
- Comunidad de Python para herramientas de desarrollo
- Apache Spark para procesamiento distribuido

## Directores
- **Miguel Brito**
- **Ituriel Flores**

