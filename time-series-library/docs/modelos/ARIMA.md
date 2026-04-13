# Modelo Autoregresivo Integrado de Media Móvil (ARIMA)

## Definición Matemática

Un proceso Autoregresivo Integrado de Media Móvil de orden (p,d,q), denotado como **ARIMA(p,d,q)**, es una extensión del modelo ARMA que incorpora diferenciación para manejar series **no estacionarias**.

### Ecuación General

$$\Phi(B)(1-B)^d y_t = c + \Theta(B) \varepsilon_t$$

Donde:
- $y_t$ es el valor de la serie en el tiempo $t$
- $(1-B)^d$ es el operador de diferenciación de orden $d$
- $\Phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ (polinomio AR)
- $\Theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ (polinomio MA)
- $c$ es una constante
- $\varepsilon_t \sim N(0, \sigma^2)$ es ruido blanco
- $p$ es el orden AR
- $d$ es el orden de diferenciación (integración)
- $q$ es el orden MA

### Forma Expandida

Aplicando la diferenciación:

$$\nabla^d y_t = y_t^*$$

Donde $y_t^*$ es la serie diferenciada $d$ veces. Entonces:

$$y_t^* = c + \phi_1 y_{t-1}^* + \cdots + \phi_p y_{t-p}^* + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}$$

### Operador de Diferenciación

- **Primera diferencia** ($d=1$): $\nabla y_t = (1-B)y_t = y_t - y_{t-1}$
- **Segunda diferencia** ($d=2$): $\nabla^2 y_t = (1-B)^2 y_t = y_t - 2y_{t-1} + y_{t-2}$
- **d-ésima diferencia**: $\nabla^d y_t = (1-B)^d y_t$

## Propiedades Estadísticas

### Estacionariedad e Integración

**Concepto de Integración:**
- Una serie es integrada de orden $d$, denotada $I(d)$, si se requieren $d$ diferenciaciones para hacerla estacionaria
- $y_t \sim I(d)$ significa que $\nabla^d y_t$ es estacionaria
- $\nabla^d y_t \sim I(0)$ (estacionaria)

**Propiedades:**
1. **Serie Original**: Puede ser no estacionaria
2. **Serie Diferenciada**: Debe ser estacionaria para aplicar ARMA
3. **Componente ARMA**: Aplicado a la serie diferenciada

### Raíces Unitarias

Una serie no estacionaria tiene **raíces unitarias** en su polinomio AR:

$$\Phi^*(B) = \Phi(B)(1-B)^d$$

Las raíces unitarias son eliminadas por la diferenciación.

### Media y Varianza

Para la serie diferenciada $y_t^* = \nabla^d y_t$:

**Media:**
$$E[y_t^*] = \mu^* = \frac{c}{1 - \phi_1 - \cdots - \phi_p}$$

**Varianza:**
La varianza de $y_t^*$ es finita y constante (estacionaria).

**Nota**: La serie original $y_t$ no tiene varianza finita si $d > 0$.

### Implicaciones de la Diferenciación

- **d = 0**: Serie ya es estacionaria → usar ARMA(p,q)
- **d = 1**: Serie tiene tendencia estocástica → aplicar una diferencia
- **d = 2**: Serie tiene curvatura/aceleración → aplicar dos diferencias
- **d ≥ 3**: Muy raro en la práctica, posible sobre-diferenciación

## Requisitos de los Datos

### Características Necesarias

1. **Puede ser No Estacionaria**: A diferencia de ARMA, ARIMA maneja no estacionariedad
   - Tendencia estocástica permitida
   - Media no constante permitida
   - **No** debe tener tendencia determinística fuerte (usar regresión primero)

2. **Longitud Mínima**:
   - **Mínimo absoluto**: $n > p + d + q$
   - **Recomendado**: $n \geq 4 \times \max(p, q) + d \times 10$
   - **Óptimo**: $n \geq 100$ para estimaciones robustas

3. **Diferenciación Apropiada**:
   - $d$ debe ser el mínimo necesario para estacionariedad
   - Sobre-diferenciar introduce autocorrelación MA negativa
   - Sub-diferenciar deja tendencia residual

4. **Sin Cambios Estructurales**:
   - No debe haber quiebres en la serie
   - Relaciones estables en el tiempo
   - Parámetros constantes

### Naturaleza de las Series Apropiadas

Los modelos ARIMA son apropiados para series que:
- Exhiben tendencia estocástica (random walk con drift)
- Requieren diferenciación para estacionariedad
- Tienen autocorrelación después de diferenciar
- No tienen patrones estacionales (usar SARIMA)
- Muestran integración de primer o segundo orden

## Patrones ACF/PACF Característicos

### Identificación de d (Orden de Diferenciación)

**Antes de Diferenciar (serie no estacionaria):**
- ACF decae muy lentamente
- PACF muestra un spike grande en lag 1

**Después de Diferenciar Correctamente:**
- ACF decae rápidamente
- Patrones característicos de ARMA emergen

### Tests Estadísticos para d

1. **Augmented Dickey-Fuller (ADF)**:
   - $H_0$: La serie tiene raíz unitaria ($d \geq 1$)
   - $H_1$: La serie es estacionaria ($d = 0$)
   - Si $p < 0.05$: Rechazar $H_0$ → serie es estacionaria

2. **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**:
   - $H_0$: La serie es estacionaria ($d = 0$)
   - $H_1$: La serie tiene raíz unitaria ($d \geq 1$)
   - Si $p > 0.05$: No rechazar $H_0$ → serie es estacionaria

3. **Phillips-Perron**:
   - Similar a ADF pero robusto a heterocedasticidad

### Estrategia Combinada

| ADF | KPSS | Interpretación | Acción |
|-----|------|----------------|--------|
| Rechaza | No rechaza | Estacionaria | d = 0 |
| No rechaza | Rechaza | Raíz unitaria | d = 1 |
| No rechaza | No rechaza | Cerca de estacionariedad | d = 0 o d = 1, usar AIC |
| Rechaza | Rechaza | Ambiguo | Aplicar d = 1 y reverificar |

### Identificación de (p,q) después de Diferenciar

Una vez determinado $d$, aplicar diferenciación y analizar ACF/PACF de $\nabla^d y_t$:

| Patrón | ACF | PACF | Modelo |
|--------|-----|------|--------|
| Solo AR | Decae gradual | Corte en lag p | ARIMA(p,d,0) |
| Solo MA | Corte en lag q | Decae gradual | ARIMA(0,d,q) |
| Mixto | Ambas decaen | Ambas decaen | ARIMA(p,d,q) |

## Casos de Uso y Aplicaciones

### Aplicaciones Comunes

1. **Economía y Finanzas**
   - **Precios de activos**: Típicamente I(1) - random walk
   - **Tasas de cambio**: Modelado de monedas
   - **Índices bursátiles**: Predicción de tendencias
   - **Variables macroeconómicas**: PIB, inflación (con tendencia)

2. **Ventas y Demanda**
   - **Ventas al por menor**: Con tendencia de crecimiento
   - **Demanda de productos**: Evolución temporal
   - **Tráfico web**: Crecimiento de usuarios

3. **Demografía**
   - **Población**: Crecimiento con tendencia
   - **Migraciones**: Flujos temporales
   - **Indicadores sociales**

4. **Ciencias Ambientales**
   - **Niveles de CO₂**: Tendencia de largo plazo
   - **Temperatura global**: Calentamiento con variabilidad
   - **Nivel del mar**: Aumento sostenido

5. **Ingeniería**
   - **Degradación de sistemas**: Deterioro acumulativo
   - **Consumo energético**: Con tendencia de crecimiento

### Cuándo Usar ARIMA

✅ **Usar ARIMA cuando:**
- La serie muestra tendencia estocástica
- Tests de raíz unitaria indican no estacionariedad
- Primera/segunda diferencia la hace estacionaria
- Hay autocorrelación después de diferenciar
- No hay estacionalidad fuerte

❌ **NO usar ARIMA cuando:**
- La serie ya es estacionaria (usar ARMA)
- Hay tendencia determinística (usar regresión con ARMA en errores)
- Hay componentes estacionales (usar SARIMA)
- La serie tiene cambios estructurales (usar modelos con quiebres)
- Varianza no constante (aplicar transformación logarítmica primero)

## Ventajas y Limitaciones

### Ventajas

1. **Maneja No Estacionariedad**: No requiere serie estacionaria inicial
2. **Flexibilidad**: Captura tendencias, autocorrelación y shocks
3. **Metodología Establecida**: Box-Jenkins bien documentada
4. **Ampliamente Usado**: Estándar en forecasting univariado
5. **Base Teórica Sólida**: Fundamentado en teoría de series temporales
6. **Interpretable**: Parámetros tienen significado claro

### Limitaciones

1. **Univariado**: No incorpora variables explicativas (usar ARIMAX o SARIMAX)
2. **Lineal**: No captura relaciones no lineales
3. **Identificación Compleja**: Determinar (p,d,q) requiere experiencia
4. **Horizonte Limitado**: Predicciones de largo plazo poco confiables
5. **Supuestos Fuertes**: Linealidad, homocedasticidad, normalidad
6. **Requiere Datos Suficientes**: Mínimo 50-100 observaciones

## Metodología Box-Jenkins

### Ciclo Iterativo

1. **Identificación**:
   - Análisis exploratorio de datos
   - Transformaciones (log, Box-Cox) si es necesario
   - Tests de estacionariedad (ADF, KPSS)
   - Determinar $d$ diferenciando hasta estacionariedad
   - Analizar ACF/PACF de serie diferenciada
   - Proponer órdenes (p, q) candidatos

2. **Estimación**:
   - Estimar parámetros usando MLE
   - Verificar convergencia
   - Verificar condiciones de estacionariedad e invertibilidad

3. **Diagnóstico**:
   - Análisis de residuos (deben ser ruido blanco)
   - Ljung-Box test de autocorrelación residual
   - Verificar normalidad (Q-Q plot, Jarque-Bera)
   - Calcular AIC, BIC

4. **Selección**:
   - Si diagnóstico no es satisfactorio, volver a paso 1
   - Comparar modelos candidatos usando AIC/BIC
   - Seleccionar modelo más parsimonioso

5. **Predicción**:
   - Generar forecasts con intervalos de confianza
   - Evaluar performance out-of-sample si es posible

## Estimación de Parámetros

### Maximum Likelihood Estimation

1. **Transformar**: Aplicar diferenciación $y_t^* = \nabla^d y_t$
2. **Estimar ARMA**: Ajustar ARMA(p,q) a $y_t^*$ usando MLE
3. **Algoritmo**:
   - Calcular innovaciones recursivamente
   - Evaluar log-verosimilitud
   - Optimizar usando L-BFGS-B o similar
4. **Verificar**: Condiciones de estacionariedad e invertibilidad

### Función de Verosimilitud

Para serie diferenciada $y^*$:

$$L(\phi, \theta, \sigma^2 | y^*) = -\frac{n^*}{2}\ln(2\pi) - \frac{n^*}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^{n^*} \varepsilon_t^2$$

Donde $n^* = n - d$ es el número de observaciones después de diferenciar.

## Selección de Orden (p,d,q)

### Estrategia Completa

```python
# Paso 1: Determinar d
d = 0
while not is_stationary(data):
    data = diff(data)
    d += 1
    if d > 2:  # Precaución
        break

# Paso 2: Grid search para (p,q)
best_aic = inf
for p in range(max_p + 1):
    for q in range(max_q + 1):
        if p == 0 and q == 0:
            continue
        model = ARIMA(p, d, q)
        model.fit(original_data)
        if model.aic < best_aic:
            best_aic = model.aic
            best_order = (p, d, q)
```

### Criterios de Información

**AIC** (favorece ajuste):
$$AIC = -2\ln(L) + 2(p + q + k)$$

**BIC** (penaliza complejidad):
$$BIC = -2\ln(L) + (p + q + k)\ln(n)$$

Donde $k = 1$ si hay constante, $k = 0$ si no.

### Heurísticas Prácticas

- **d**: Raramente > 2; casi siempre 0, 1 o 2
- **p**: Típicamente ≤ 3
- **q**: Típicamente ≤ 3
- **p + q**: Raramente > 5

## Diagnóstico del Modelo

### Checks Esenciales

1. **Residuos como Ruido Blanco**:
   ```python
   - ACF de residuos: todos los lags no significativos
   - Ljung-Box Q-statistic: p-value > 0.05
   - Plot de residuos vs tiempo: sin patrones
   ```

2. **Normalidad**:
   ```python
   - Q-Q plot: aproximadamente lineal
   - Histograma: forma de campana
   - Jarque-Bera test: p-value > 0.05
   ```

3. **Homocedasticidad**:
   ```python
   - Residuos² vs tiempo: sin tendencia
   - No debe haber clusters de volatilidad
   ```

4. **Estabilidad de Parámetros**:
   ```python
   - Raíces AR fuera del círculo unitario
   - Raíces MA fuera del círculo unitario
   ```

### Problemas Comunes y Soluciones

| Problema | Diagnóstico | Solución |
|----------|-------------|----------|
| ACF residuos significativa | Modelo insuficiente | Aumentar p o q |
| Residuos no normales | Q-Q plot no lineal | Transformar datos (log) |
| Heterocedasticidad | Varianza cambiante | Usar GARCH complementario |
| Sobre-diferenciación | θ₁ ≈ -1 en MA(1) | Reducir d |
| Sub-diferenciación | ACF decae lentamente | Aumentar d |

## Predicción

### Proceso de Predicción

1. **Predecir en Escala Diferenciada**:
   $$\hat{y}_{t+h|t}^* = E[\nabla^d y_{t+h} | I_t]$$
   
   Usando ecuación ARMA(p,q).

2. **Integrar (Revertir Diferenciación)**:
   - Para $d=1$: $\hat{y}_{t+h|t} = \hat{y}_{t+h-1|t} + \hat{y}_{t+h|t}^*$
   - Para $d=2$: Aplicar integración dos veces

3. **Calcular Intervalos de Confianza**:
   Varianza de error de predicción aumenta con $h$.

### Fórmula de Predicción Recursiva

Para ARIMA(p,d,q):

$$\hat{y}_{t+h|t} = \hat{y}_{t+h-1|t} + c + \sum_{i=1}^p \phi_i \nabla^d \hat{y}_{t+h-i|t} + \sum_{j=h}^q \theta_j \varepsilon_{t+h-j}$$

### Intervalos de Confianza (95%)

$$IC_{95\%} = \hat{y}_{t+h|t} \pm 1.96 \times SE(e_{t+h})$$

Donde la varianza del error aumenta con $h$:

$$\text{Var}(e_{t+h}) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$$

**Nota**: Para series integradas, la varianza crece sin límite cuando $h \to \infty$.

## Referencias y Lecturas Recomendadas

1. **Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015)**. *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
   - La referencia clásica y fundamental

2. **Brockwell, P. J., & Davis, R. A. (2016)**. *Introduction to Time Series and Forecasting* (3rd ed.). Springer.
   - Tratamiento matemático riguroso

3. **Hamilton, J. D. (1994)**. *Time Series Analysis*. Princeton University Press.
   - Enfoque en series económicas

4. **Hyndman, R. J., & Athanasopoulos, G. (2021)**. *Forecasting: Principles and Practice* (3rd ed.). OTexts.
   - Excelente recurso práctico, disponible online

5. **Tsay, R. S. (2010)**. *Analysis of Financial Time Series* (3rd ed.). Wiley.
   - Aplicaciones en finanzas

## Ejemplo Práctico

### Serie con Tendencia: Random Walk con Drift

```python
from tslib.models import ARIMAModel
import numpy as np
import matplotlib.pyplot as plt

# Generate random walk with drift: y_t = 0.5 + y_{t-1} + ε_t
np.random.seed(42)
n = 200
drift = 0.5
epsilon = np.random.normal(0, 1, n)

y = np.zeros(n)
y[0] = epsilon[0]
for t in range(1, n):
    y[t] = drift + y[t-1] + epsilon[t]

# This series is I(1) - integrated of order 1
print(f"Series has trend: {y[-1] - y[0]:.2f}")

# Fit ARIMA model with automatic order selection
model = ARIMAModel(auto_select=True, max_p=3, max_d=2, max_q=3)
model.fit(y)

print(model.summary())
# Expected: ARIMA(0,1,0) with drift ≈ 0.5 (random walk with drift)
# Or possibly ARIMA(0,1,1) with θ₁ ≈ 0

# Forecast
forecast, conf_int = model.predict(steps=50, return_conf_int=True)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y, label='Historical Data')
plt.plot(range(n, n+50), forecast, label='Forecast', color='red')
plt.fill_between(range(n, n+50), conf_int[0], conf_int[1], 
                 alpha=0.3, color='red', label='95% CI')
plt.legend()
plt.title('ARIMA Forecast with Confidence Intervals')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Notice: CI gets wider as h increases (uncertainty grows)
```

## Notas Adicionales

- ARIMA es el modelo univariado más utilizado en forecasting
- La metodología Box-Jenkins es sistemática pero requiere práctica
- En la práctica, ARIMA(0,1,1) o ARIMA(1,1,0) son muy comunes para datos financieros
- Para series con estacionalidad, usar SARIMA
- Para incluir variables explicativas, usar ARIMAX o modelos de regresión dinámicacon errores ARIMA
- Los modelos ARIMA asumen relaciones lineales; para no linealidad usar GARCH, TAR, o modelos de machine learning
- Siempre verificar supuestos y realizar análisis de residuos
- La calidad de las predicciones disminuye rápidamente con el horizonte

