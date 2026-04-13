# Modelo de Media Móvil (MA)

## Definición Matemática

Un proceso de Media Móvil de orden q, denotado como **MA(q)**, es un modelo de series temporales donde el valor actual de la serie es una combinación lineal de errores aleatorios actuales y pasados.

### Ecuación General

$$y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}$$

Donde:
- $y_t$ es el valor de la serie en el tiempo $t$
- $\mu$ es la media del proceso
- $\varepsilon_t \sim N(0, \sigma^2)$ es ruido blanco (innovaciones)
- $\theta_1, \theta_2, \ldots, \theta_q$ son los parámetros de media móvil
- $q$ es el orden del modelo MA

### Forma Compacta con Operador de Rezago

Usando el operador de rezago $B$ donde $B \varepsilon_t = \varepsilon_{t-1}$:

$$y_t = \mu + \Theta(B) \varepsilon_t$$

Donde $\Theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ es el polinomio de media móvil.

## Propiedades Estadísticas

### Invertibilidad

Un proceso MA(q) es **invertible** si puede expresarse como un proceso AR(∞). Esto ocurre cuando todas las raíces del polinomio característico:

$$\Theta(z) = 1 + \theta_1 z + \theta_2 z^2 + \cdots + \theta_q z^q = 0$$

están **fuera del círculo unitario** (módulo mayor a 1).

**Condiciones de invertibilidad para casos comunes:**

- **MA(1)**: $|\theta_1| < 1$
- **MA(2)**: $\theta_1 + \theta_2 < 1$, $\theta_2 - \theta_1 < 1$, $|\theta_2| < 1$

**Nota importante**: La invertibilidad es necesaria para garantizar una representación única y para permitir la estimación de parámetros.

### Media y Varianza

Para un proceso MA(q):

**Media:**
$$E[y_t] = \mu$$

**Varianza:**
$$\text{Var}(y_t) = \gamma_0 = \sigma^2 (1 + \theta_1^2 + \theta_2^2 + \cdots + \theta_q^2)$$

**Autocovarianza:**
$$\gamma_k = \begin{cases}
\sigma^2 (\theta_k + \theta_1\theta_{k+1} + \cdots + \theta_{q-k}\theta_q) & \text{si } k \leq q \\
0 & \text{si } k > q
\end{cases}$$

### Función de Autocorrelación (ACF)

La ACF de un MA(q):
- **Se corta abruptamente después del lag q**
- $\rho_k \neq 0$ para $k \leq q$
- $\rho_k = 0$ para $k > q$

Esta es la **característica distintiva** para identificar el orden de un modelo MA.

**Fórmula de autocorrelación:**
$$\rho_k = \frac{\gamma_k}{\gamma_0} = \frac{\theta_k + \theta_1\theta_{k+1} + \cdots + \theta_{q-k}\theta_q}{1 + \theta_1^2 + \theta_2^2 + \cdots + \theta_q^2}$$

### Función de Autocorrelación Parcial (PACF)

La PACF de un MA(q):
- **Decae exponencialmente** o muestra un patrón sinusoidal amortiguado
- No se corta abruptamente
- Similar al comportamiento de ACF en modelos AR

## Requisitos de los Datos

### Características Necesarias

1. **Estacionariedad Débil**: El proceso MA es siempre débilmente estacionario
   - Media constante $\mu$
   - Varianza constante
   - Autocovarianza finita que solo depende del rezago

2. **Longitud Mínima**:
   - **Mínimo absoluto**: $n > q$ (más observaciones que parámetros)
   - **Recomendado**: $n \geq 4q$ para estimaciones confiables
   - **Óptimo**: $n \geq 50$ para órdenes bajos ($q \leq 3$)

3. **Innovaciones**: El modelo asume que los errores $\varepsilon_t$ son:
   - Independientes e idénticamente distribuidos
   - Media cero
   - Varianza constante $\sigma^2$

4. **Ausencia de Estructura Adicional**: 
   - No debe haber autocorrelación residual más allá del lag q
   - Los shocks deben tener efectos transitorios

### Naturaleza de las Series Apropiadas

Los modelos MA son apropiados para series que:
- Reaccionan a shocks externos pero retornan a la media
- No tienen "memoria" de largo plazo
- Muestran correlación solo en los primeros lags
- Presentan volatilidad agrupada en ventanas cortas

## Patrones ACF/PACF Característicos

### Identificación Visual

| Orden | ACF | PACF |
|-------|-----|------|
| MA(1) | Spike en lag 1, luego 0 | Decaimiento exponencial |
| MA(2) | Spikes en lags 1 y 2, luego 0 | Decaimiento sinusoidal o exponencial |
| MA(q) | Spikes hasta lag q, luego 0 | Decaimiento gradual |

### Ejemplo MA(1) con θ₁ = 0.6

```
ACF:  ▓▓▓▓▓▓ (lag 1)
      ·      (lag 2)
      ·      (lag 3)
      ·      (lag 4)
      
PACF: ▓▓▓▓▓  (lag 1)
      ▓▓▓    (lag 2)
      ▓▓     (lag 3)
      ▓      (lag 4)
```

## Casos de Uso y Aplicaciones

### Aplicaciones Comunes

1. **Economía y Finanzas**
   - Retornos de activos financieros (shocks de mercado)
   - Errores de pronóstico
   - Efectos de anuncios económicos

2. **Control de Calidad**
   - Variaciones de producción por ajustes de maquinaria
   - Efectos de cambios de proceso

3. **Meteorología**
   - Anomalías climáticas de corto plazo
   - Efectos de fenómenos transitorios

4. **Procesamiento de Señales**
   - Filtrado de ruido
   - Suavizamiento de señales
   - Detección de eventos

5. **Marketing**
   - Efectos de campañas publicitarias
   - Respuesta a promociones de corto plazo

### Cuándo Usar MA

✅ **Usar MA cuando:**
- Los shocks tienen efectos de corto plazo que se disipan rápidamente
- ACF muestra corte claro después de q lags
- La serie reacciona a eventos pero retorna a la media
- No hay autocorrelación de largo plazo
- Los errores de predicción están correlacionados

❌ **NO usar MA cuando:**
- Hay dependencia persistente de valores pasados (usar AR)
- La serie tiene tendencia o no es estacionaria (aplicar diferenciación)
- Hay componentes estacionales fuertes (usar SMA)
- Se necesita capturar memoria de largo plazo

## Ventajas y Limitaciones

### Ventajas

1. **Siempre Estacionario**: Todo proceso MA es débilmente estacionario
2. **Interpretación de Shocks**: Los parámetros representan el efecto de innovaciones pasadas
3. **Parsimonia**: Puede capturar efectos de corto plazo con pocos parámetros
4. **Flexibilidad**: Útil para modelar efectos transitorios
5. **Predicción Simple**: Más allá de q pasos, la predicción es la media

### Limitaciones

1. **Estimación Compleja**: Requiere métodos no lineales (MLE)
2. **Identificación Difícil**: Menos intuitivo que AR en la práctica
3. **No Observable**: Los errores $\varepsilon_t$ no son directamente observables
4. **Alcance Limitado**: Solo captura efectos de corto plazo
5. **Predicción de Largo Plazo**: Converge rápidamente a la media

## Estimación de Parámetros

### Métodos Principales

1. **Maximum Likelihood Estimation (MLE)**
   - Método estándar y óptimo
   - Usado en esta implementación
   - Requiere optimización numérica iterativa
   - Usa algoritmo de innovaciones o filtro de Kalman

2. **Método de Momentos**
   - Basado en ecuaciones de autocorrelación
   - Puede producir estimaciones no invertibles
   - Útil para valores iniciales

3. **Mínimos Cuadrados Condicionales**
   - Condicional en primeros q errores
   - Más simple computacionalmente
   - Menos eficiente que MLE

### Algoritmo de Innovaciones

Para estimar un MA(q), se usa el algoritmo de innovaciones que:
1. Predice $\hat{y}_t$ basado en errores pasados
2. Calcula innovación: $\varepsilon_t = y_t - \hat{y}_t$
3. Actualiza parámetros iterativamente
4. Maximiza la función de verosimilitud

## Diagnóstico del Modelo

### Verificaciones Post-Ajuste

1. **Residuos**:
   - Deben ser ruido blanco
   - No debe haber autocorrelación
   - Media ≈ 0, varianza constante

2. **ACF de Residuos**:
   - Todos los lags deben ser no significativos
   - Ljung-Box test: $p > 0.05$

3. **Invertibilidad**:
   - Verificar que $|\theta_i| < 1$ para MA(1)
   - Raíces del polinomio fuera del círculo unitario

4. **Criterios de Información**:
   - AIC, BIC para selección de orden
   - Menor valor indica mejor ajuste

## Predicción

### Predicción Multi-paso

Para un MA(q), la predicción es:

$$\hat{y}_{t+h|t} = \begin{cases}
\mu + \theta_h \varepsilon_t + \theta_{h+1} \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t+q-h} & \text{si } h \leq q \\
\mu & \text{si } h > q
\end{cases}$$

**Característica importante**: Para $h > q$, la predicción es simplemente la media $\mu$.

### Intervalos de Confianza

La varianza del error de predicción:

$$\text{Var}(\varepsilon_{t+h}) = \sigma^2 (1 + \theta_1^2 + \cdots + \theta_{\min(h-1,q)}^2)$$

Se estabiliza después de q pasos:

$$\text{Var}(\varepsilon_{t+h}) = \sigma^2 (1 + \theta_1^2 + \cdots + \theta_q^2) \quad \text{para } h > q$$

## Relación con Otros Modelos

### MA vs AR

- **MA**: Efecto de shocks se disipa en q períodos
- **AR**: Efecto de shocks persiste indefinidamente (decae exponencialmente)

### Dualidad MA-AR

Por el teorema de Wold:
- Todo proceso AR estacionario tiene representación MA(∞)
- Todo proceso MA invertible tiene representación AR(∞)

### MA en ARMA

Un MA(q) es un caso especial de ARMA(0,q).

## Referencias y Lecturas Recomendadas

1. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
2. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer.
3. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
4. Chatfield, C. (2003). *The Analysis of Time Series: An Introduction*. Chapman and Hall/CRC.

## Ejemplo Práctico

### Serie Sintética MA(2)

```python
from tslib.models import MAModel
import numpy as np

# Generate MA(2) data: y_t = μ + ε_t + 0.7*ε_{t-1} - 0.4*ε_{t-2}
np.random.seed(42)
n = 200
mu = 10
theta = [0.7, -0.4]
epsilon = np.random.normal(0, 1, n)

y = np.zeros(n)
for t in range(2, n):
    y[t] = mu + epsilon[t] + theta[0]*epsilon[t-1] + theta[1]*epsilon[t-2]

# Fit MA model with automatic order selection
model = MAModel(auto_select=True, max_order=5)
model.fit(y)

print(model.summary())
# Expected: MA(2) with θ₁ ≈ 0.7, θ₂ ≈ -0.4

# Forecast
forecast, conf_int = model.predict(steps=10, return_conf_int=True)
print(f"Forecast beyond lag 2 converges to mean: {mu:.2f}")
```

## Notas Adicionales

- Los modelos MA son fundamentales en la metodología Box-Jenkins
- Son especialmente útiles para modelar errores de pronóstico
- La estimación de MA es más compleja que AR porque los errores no son observables
- En la práctica, órdenes altos (q > 2) son raros
- Los modelos MA puros son menos comunes que ARMA en aplicaciones reales

