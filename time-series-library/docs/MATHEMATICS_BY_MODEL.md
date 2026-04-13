# Fundamentos Matemáticos de TSLib

> **Referencia canónica.** Este documento describe de forma completa la teoría matemática que sustenta cada componente de TSLib.  
> - Para API de uso (clases, parámetros, ejemplos de código), ver [`docs/api_reference.md`](api_reference.md).  
> - Para guía de integración, ver [`docs/INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md).  
> - La notación sigue la convención de Box & Jenkins (2015) y Hamilton (1994).

---

## Tabla de contenidos

1. [Notación y preliminares](#1-notación-y-preliminares)
2. [Función de Autocorrelación (ACF)](#2-función-de-autocorrelación-acf)
3. [Función de Autocorrelación Parcial (PACF)](#3-función-de-autocorrelación-parcial-pacf)
4. [Tests de Estacionariedad](#4-tests-de-estacionariedad)
5. [Máxima Verosimilitud (MLE)](#5-máxima-verosimilitud-mle)
6. [Modelo AR(p)](#6-modelo-arp)
7. [Modelo MA(q)](#7-modelo-maq)
8. [Modelo ARMA(p,q)](#8-modelo-armap-q)
9. [Modelo ARIMA(p,d,q)](#9-modelo-arimap-d-q)
10. [Paralelización con Spark](#10-paralelización-con-spark)
11. [Criterios de Información](#11-criterios-de-información)
12. [Métricas de Pronóstico](#12-métricas-de-pronóstico)
13. [Referencias](#13-referencias)

---

## 1. Notación y preliminares

| Símbolo | Significado |
|---------|-------------|
| `y_t` | Observación en el tiempo t |
| `ȳ` | Media muestral de la serie |
| `ε_t ~ N(0, σ²)` | Ruido blanco (innovaciones) |
| `B` | Operador de rezago: `B·y_t = y_{t-1}` |
| `∇ = 1 − B` | Operador de diferenciación: `∇y_t = y_t − y_{t-1}` |
| `p, q` | Órdenes AR y MA respectivamente |
| `d` | Orden de diferenciación |
| `n` | Número de observaciones |
| `L(θ; y)` | Función de verosimilitud |
| `ℓ(θ; y)` | Log-verosimilitud |

---

## 2. Función de Autocorrelación (ACF)

### Definición poblacional

La **covarianza de rezago k** de un proceso estacionario es:

```
γ_k = Cov(y_t, y_{t-k}) = E[(y_t − μ)(y_{t-k} − μ)]
```

La **ACF poblacional** normaliza por la varianza:

```
ρ_k = γ_k / γ_0 = γ_k / Var(y_t)
```

Propiedades: ρ_0 = 1, ρ_k = ρ_{-k} (simetría).

### Estimación muestral

```
ρ̂_k = Σ_{t=k+1}^{n} (y_t − ȳ)(y_{t-k} − ȳ)
        ──────────────────────────────────────────
              Σ_{t=1}^{n} (y_t − ȳ)²
```

**Implementación TSLib** (`ACFCalculator`): cálculo vectorizado con NumPy; paralelización por hilos para n > 1000.

### Bandas de confianza

Bajo la hipótesis nula de no autocorrelación (ruido blanco), por el TLC la distribución asintótica de ρ̂_k es:

```
√n · ρ̂_k  →  N(0, 1)
```

La banda de confianza al 95 % (Bartlett):

```
± 1.96 / √n
```

---

## 3. Función de Autocorrelación Parcial (PACF)

### Definición

La PACF en el lag k mide la correlación entre `y_t` e `y_{t-k}` **eliminando la influencia de los rezagos intermedios** `y_{t-1}, …, y_{t-k+1}`.

### Algoritmo de Durbin-Levinson

La PACF se calcula recursivamente a partir de la ACF:

**Inicialización:**
```
φ_{1,1} = ρ_1
```

**Para k ≥ 2:**
```
           ρ_k − Σ_{j=1}^{k-1} φ_{k-1,j} · ρ_{k-j}
φ_{k,k} = ─────────────────────────────────────────────
           1 − Σ_{j=1}^{k-1} φ_{k-1,j} · ρ_j
```

**Actualización de coeficientes:**
```
φ_{k,j} = φ_{k-1,j} − φ_{k,k} · φ_{k-1,k-j}    j = 1, …, k-1
```

La PACF en lag k es `φ_{k,k}`.

**Implementación TSLib** (`PACFCalculator`): implementación directa de Durbin-Levinson sobre la ACF muestral.

---

## 4. Tests de Estacionariedad

Un proceso `{y_t}` es **estacionario en covarianza** si:
- E[y_t] = μ (constante)  
- Cov(y_t, y_{t-k}) = γ_k solo depende de k

### 4.1 Test ADF (Augmented Dickey-Fuller)

**H₀**: existe raíz unitaria (serie no estacionaria).  
**H₁**: no hay raíz unitaria (serie estacionaria).

La regresión ADF con tendencia (caso general) es:

```
∆y_t = α + β·t + δ·y_{t-1} + Σ_{i=1}^{k} γ_i·∆y_{t-i} + ε_t
```

El estadístico de prueba es el t-ratio de δ̂:

```
τ = δ̂ / SE(δ̂)
```

Se rechaza H₀ (la serie es estacionaria) cuando τ < valor crítico tabulado (Dickey-Fuller, 1979).

El número óptimo de rezagos k se elige por AIC o BIC.

**Implementación TSLib** (`ADFTest`): regresión OLS desde cero; valores críticos aproximados de Mackinnon (1994).

### 4.2 Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

**H₀**: la serie es estacionaria (alrededor de nivel o tendencia).  
**H₁**: existe raíz unitaria.

El estadístico KPSS es:

```
η = (1 / n²) · Σ_{t=1}^{n} S_t² / λ²
```

donde `S_t = Σ_{i=1}^{t} ê_i` (suma parcial de residuales) y `λ²` es la varianza de largo plazo estimada por el estimador de Newey-West.

**Uso en TSLib**: ADF y KPSS se combinan para determinar d en ARIMA. Si ADF rechaza H₀ y KPSS no rechaza H₀ → d = 0 (ya estacionaria).

---

## 5. Máxima Verosimilitud (MLE)

### Función de log-verosimilitud Gaussiana

Dado un vector de innovaciones (residuales) `ε = (ε₁, …, ε_n)`:

```
ℓ(θ, σ²; y) = −(n/2)·ln(2π) − (n/2)·ln(σ²) − (1/2σ²)·Σ_{t=1}^{n} ε_t²(θ)
```

Donde `ε_t(θ)` son los residuales del modelo evaluados en el vector de parámetros θ.

La estimación de σ² concentrada:

```
σ̂² = (1/n) · Σ_{t=1}^{n} ε_t²(θ̂)
```

Sustituyendo, la log-verosimilitud concentrada equivale a minimizar la suma de cuadrados de residuales:

```
SSR(θ) = Σ_{t=1}^{n} ε_t²(θ)
```

### Algoritmo de optimización

TSLib usa el optimizador **L-BFGS-B** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno con restricciones de caja) de `scipy.optimize.minimize`:

1. Se inicializan parámetros con estimadores preliminares (mínimos cuadrados ordinarios).
2. Se minimizan `−ℓ(θ)` con respecto a θ sujeto a las restricciones de estacionariedad e invertibilidad.
3. Para AR: |raíces de Φ(z)| > 1. Para MA: |raíces de Θ(z)| > 1.

**Errores estándar** de los estimadores: diagonal de `−H⁻¹` donde H es la matriz Hessiana de `ℓ` evaluada en θ̂.

**Implementación TSLib** (`MLEOptimizer`): parámetros iniciales → L-BFGS-B → actualización de residuales → convergencia → cálculo de SE.

---

## 6. Modelo AR(p)

### Definición

```
y_t = c + φ₁·y_{t-1} + φ₂·y_{t-2} + … + φ_p·y_{t-p} + ε_t
```

**Con operador de rezago** (B·y_t = y_{t-1}):

```
Φ(B)·y_t = c + ε_t
Φ(B) = 1 − φ₁·B − φ₂·B² − … − φ_p·Bᵖ
```

### Condición de estacionariedad

AR(p) es estacionario ⟺ todas las raíces del polinomio characteristic `Φ(z) = 0` satisfacen `|z| > 1`.

Casos particulares:
- AR(1): `|φ₁| < 1`
- AR(2): `φ₁ + φ₂ < 1`,  `φ₂ − φ₁ < 1`,  `|φ₂| < 1`

### Identificación

| Función | Comportamiento |
|---------|---------------|
| **PACF** | **Corte abrupto** después del lag p → indica el orden |
| **ACF** | Decaimiento exponencial o sinusoidal amortiguado |

### Pronóstico

El pronóstico h pasos adelante (h-step ahead forecast):

```
ŷ_{n+h|n} = c + φ₁·ŷ_{n+h-1|n} + … + φ_p·ŷ_{n+h-p|n}
```

donde `ŷ_{n+j|n} = y_{n+j}` para j ≤ 0 (valores observados).

**Implementación TSLib**: `ARProcess.fit()` con `MLEOptimizer` → `ARProcess.predict()`. API de alto nivel: `ARModel`.

---

## 7. Modelo MA(q)

### Definición

```
y_t = μ + ε_t + θ₁·ε_{t-1} + θ₂·ε_{t-2} + … + θ_q·ε_{t-q}
```

**Con operador de rezago:**

```
y_t = μ + Θ(B)·ε_t
Θ(B) = 1 + θ₁·B + θ₂·B² + … + θ_q·Bᵠ
```

### Condición de invertibilidad

MA(q) es invertible ⟺ todas las raíces de `Θ(z) = 0` satisfacen `|z| > 1`.

Caso MA(1): `|θ₁| < 1`.

La invertibilidad garantiza que un MA(q) se puede representar como un AR(∞) convergente, haciendo posible la estimación y el pronóstico.

### Identificación

| Función | Comportamiento |
|---------|---------------|
| **ACF** | **Corte abrupto** después del lag q → indica el orden |
| **PACF** | Decaimiento exponencial o sinusoidal |

### Pronóstico

Las innovaciones pasadas `ε_{t-1}, …, ε_{t-q}` se "heredan" del ajuste:

```
ŷ_{n+h|n} = μ    para h > q
ŷ_{n+h|n} = μ + Σ_{j=h}^{q} θ_j · ε_{n+h-j}    para h ≤ q
```

**Implementación TSLib**: `MAProcess.fit()` con `MLEOptimizer` → `MAProcess.predict()`. API: `MAModel`.

---

## 8. Modelo ARMA(p,q)

### Definición

Combina AR y MA en una sola ecuación:

```
y_t = c + φ₁·y_{t-1} + … + φ_p·y_{t-p}
        + ε_t + θ₁·ε_{t-1} + … + θ_q·ε_{t-q}
```

**Con operadores:**

```
Φ(B)·y_t = c + Θ(B)·ε_t
```

### Condiciones

| Condición | Requisito |
|-----------|-----------|
| **Estacionariedad** | Raíces de Φ(z) = 0 fuera del círculo unitario |
| **Invertibilidad** | Raíces de Θ(B) = 0 fuera del círculo unitario |

### Identificación

Cuando ambas ACF y PACF decaen sin corte abrupto → sugerir ARMA. El orden (p, q) se determina por **búsqueda en rejilla (grid search)** con AIC/BIC sobre el espacio `{0,…,p_max} × {0,…,q_max}`.

**Implementación TSLib**: `ARMAOrderSelector` busca en rejilla; `ARMAProcess` estima con `MLEOptimizer`. API: `ARMAModel`.

---

## 9. Modelo ARIMA(p,d,q)

### Motivación: series no estacionarias

Si `{y_t}` tiene raíz unitaria (tendencia estocástica), diferenciar produce una serie estacionaria:

```
∇y_t = y_t − y_{t-1}       (d=1)
∇²y_t = ∇(∇y_t)            (d=2)
```

### Definición

```
Φ(B) · (1−B)^d · y_t = c + Θ(B) · ε_t
```

Definiendo la serie diferenciada `y*_t = ∇^d y_t`:

```
y*_t = c + φ₁·y*_{t-1} + … + φ_p·y*_{t-p}
           + ε_t + θ₁·ε_{t-1} + … + θ_q·ε_{t-q}
```

Es decir, **ARIMA(p,d,q)** es un **ARMA(p,q)** sobre la serie d-veces diferenciada.

### Proceso de identificación completo

```
Serie original y_t
        │
        ▼
¿Es estacionaria?  ◄── Test ADF + KPSS
        │  No
        ▼
Diferenciar: ∇y_t, ∇²y_t, …  → determina d
        │
        ▼
Serie diferenciada y*_t
        │
        ▼
Analizar ACF y PACF
  ┌─────────────────────────────────────────┐
  │ PACF se corta en lag p → AR(p)          │
  │ ACF se corta en lag q → MA(q)           │
  │ Ambas decaen → búsqueda en rejilla ARMA │
  └─────────────────────────────────────────┘
        │
        ▼
Grid search AIC/BIC sobre (p, q)
        │
        ▼
ARIMA(p, d, q) → MLE → θ̂ = (φ_1…φ_p, θ_1…θ_q, σ²)
```

### Pronóstico y reintegración

Para d = 1, el pronóstico de `y_t` se recupera integrando:

```
ŷ_{n+h|n} = y_n + Σ_{j=1}^{h} ∆ŷ_{n+j|n}
```

Para d = 2:

```
ŷ_{n+h|n} = y_n + h·∆y_n + Σ_{j=1}^{h} (h − j + 1) · ∆²ŷ_{n+j|n}
```

**Implementación TSLib**: `ARIMAOrderSelector` (ADF/KPSS → d, luego selección ARMA) → `ARIMAProcess.fit()` (diferencia, ajusta ARMA, revierte) → `ARIMAProcess.predict()`. API: `ARIMAModel`.

---

## 10. Paralelización con Spark

### Principio matemático

La **matemática de los modelos no cambia** al distribuir con Spark. Lo que se paraleliza es la *ejecución* de múltiples ajustes independientes.

Hay dos niveles de paralelismo:

#### Nivel 1: paralelismo interno (threading, sin Spark)

Se usa para **una sola serie larga** (n > 1 000):

- **ACF**: cada lag k es independiente de los demás → `ThreadPoolExecutor`.
- **MLE**: paralelismo de L-BFGS-B en NumPy/SciPy (multi-thread).

```
Serie y_t  ──►  [lag 0] [lag 1] [lag 2] … [lag K]  ──►  ACF vector
               └────────────── parallel ──────────────┘
```

#### Nivel 2: paralelismo externo (Spark)

Se usa para **muchas series independientes** (por ejemplo, múltiples activos, múltiples nodos de una red, o ventanas temporales deslizantes):

```
[serie_1]  ──►  ARIMA(p,d,q)  ──►  [parámetros_1]
[serie_2]  ──►  ARIMA(p,d,q)  ──►  [parámetros_2]     (en paralelo)
   ...                               ...
[serie_N]  ──►  ARIMA(p,d,q)  ──►  [parámetros_N]
```

El mecanismo es una **Pandas UDF** que recibe un grupo de datos de Spark, llama a `ARIMAProcess.fit()` + `predict()` puro NumPy, y devuelve el resultado al DataFrame de Spark.

```python
# En cada ejecutor de Spark:
def _fit_arima_group(pdf: pd.DataFrame) -> pd.DataFrame:
    series = pdf["value"].values          # numpy array
    process = ARIMAProcess(p, d, q)       # misma lógica que el caso serial
    process.fit(series)
    forecast = process.predict(steps)
    return pd.DataFrame({"forecast": forecast})
```

### Flujo del ParallelARIMAWorkflow (11 pasos)

```
Paso 1  Crear SparkSession + configuración
Paso 2  Cargar y validar datos (DataFrame Spark)
Paso 3  Analizar ACF/PACF distribuido por grupo (SparkACFCalculator)
Paso 4  Detectar estacionariedad por grupo (ADF/KPSS distribuido)
Paso 5  Determinar d por grupo (diferenciación paralela)
Paso 6  Grid search distribuido de (p, q) por AIC
Paso 7  Ajustar ARIMA(p, d, q) en paralelo por grupo (Pandas UDF)
Paso 8  Generar pronósticos en paralelo
Paso 9  Reintegrar la serie diferenciada (↑ reconstruir y_t)
Paso 10 Calcular métricas de evaluación por grupo
Paso 11 Consolidar resultados y generar reporte
```

**Implementación TSLib**: `ParallelARIMAWorkflow` en `tslib/spark/parallel_arima_workflow.py`.

---

## 11. Criterios de Información

Se usan para comparar modelos con diferente número de parámetros k sobre una serie de longitud n.

```
AIC  = 2k − 2·ℓ(θ̂)
BIC  = k·ln(n) − 2·ℓ(θ̂)
AICc = AIC + (2k² + 2k) / (n − k − 1)   (AIC corregido para muestras pequeñas)
HQIC = 2k·ln(ln(n)) − 2·ℓ(θ̂)
```

El modelo preferido minimiza el criterio elegido. BIC penaliza más fuertemente el número de parámetros y favorece modelos más parsimoniosos.

**Implementación TSLib** (`InformationCriteria`): calcula los cuatro criterios a partir de `ℓ(θ̂)`, k, y n.

---

## 12. Métricas de Pronóstico

Dado el vector de valores reales `y` y pronósticos `ŷ`:

```
MAE   = (1/n) · Σ |y_t − ŷ_t|
RMSE  = √[(1/n) · Σ (y_t − ŷ_t)²]
MAPE  = (100/n) · Σ |y_t − ŷ_t| / |y_t|
SMAPE = (200/n) · Σ |y_t − ŷ_t| / (|y_t| + |ŷ_t|)
```

**Varianza del pronóstico h-step (AR(p) ilustrativo):**

```
Var(y_{n+h} − ŷ_{n+h|n})  = σ² · Σ_{j=0}^{h-1} ψ_j²
```

donde `{ψ_j}` son los coeficientes de la representación MA(∞) del proceso.

**Implementación TSLib** (`ForecastMetrics`, `ResidualAnalyzer`): cálculo de todas las métricas + análisis de residuales (normalidad, homocedasticidad, no autocorrelación).

---

## 13. Referencias

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
- Mackinnon, J.G. (1994). Approximate asymptotic distribution functions for unit-root and cointegration tests. *Journal of Business & Economic Statistics*, 12(2), 167–176.
- Durbin, J. (1960). The fitting of time series models. *Revue de l'Institut International de Statistique*, 28(3), 233–244.
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.
