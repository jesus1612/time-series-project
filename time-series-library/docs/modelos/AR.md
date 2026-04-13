# Modelo Autoregresivo (AR)

## Definición Matemática

Un proceso Autoregresivo de orden p, denotado como **AR(p)**, es un modelo de series temporales donde el valor actual de la serie depende linealmente de sus valores pasados más un término de error aleatorio.

### Ecuación General

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t$$

Donde:
- $y_t$ es el valor de la serie en el tiempo $t$
- $c$ es una constante (intercepto)
- $\phi_1, \phi_2, \ldots, \phi_p$ son los parámetros autorregresivos
- $\varepsilon_t \sim N(0, \sigma^2)$ es ruido blanco (error aleatorio)
- $p$ es el orden del modelo AR

### Forma Compacta con Operador de Rezago

Usando el operador de rezago $B$ donde $B y_t = y_{t-1}$:

$$\Phi(B) y_t = c + \varepsilon_t$$

Donde $\Phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ es el polinomio autorregresivo.

## Propiedades Estadísticas

### Estacionariedad

Un proceso AR(p) es **estacionario en covarianza** si todas las raíces del polinomio característico:

$$\Phi(z) = 1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$$

están **fuera del círculo unitario** (módulo mayor a 1).

**Condiciones de estacionariedad para casos comunes:**

- **AR(1)**: $|\phi_1| < 1$
- **AR(2)**: $\phi_1 + \phi_2 < 1$, $\phi_2 - \phi_1 < 1$, $|\phi_2| < 1$

### Media y Varianza

Para un proceso AR(p) estacionario:

**Media:**
$$E[y_t] = \mu = \frac{c}{1 - \phi_1 - \phi_2 - \cdots - \phi_p}$$

**Varianza:**
$$\text{Var}(y_t) = \gamma_0 = \frac{\sigma^2}{1 - \phi_1\rho_1 - \phi_2\rho_2 - \cdots - \phi_p\rho_p}$$

Donde $\rho_k$ es la autocorrelación en el lag $k$.

### Función de Autocorrelación (ACF)

Para un AR(p), la ACF:
- **Decae exponencialmente** o muestra un patrón sinusoidal amortiguado
- No se corta abruptamente
- La forma depende de las raíces del polinomio característico

### Función de Autocorrelación Parcial (PACF)

La PACF de un AR(p):
- **Se corta abruptamente después del lag p**
- $\text{PACF}(k) \neq 0$ para $k \leq p$
- $\text{PACF}(k) = 0$ para $k > p$

Esta es la **característica distintiva** para identificar el orden de un modelo AR.

## Requisitos de los Datos

### Características Necesarias

1. **Estacionariedad**: La serie debe ser estacionaria o transformarse para serlo
   - Media constante en el tiempo
   - Varianza constante en el tiempo
   - Autocovarianza que solo depende del rezago

2. **Longitud Mínima**:
   - **Mínimo absoluto**: $n > p$ (más observaciones que parámetros)
   - **Recomendado**: $n \geq 4p$ para estimaciones confiables
   - **Óptimo**: $n \geq 50$ para órdenes bajos ($p \leq 3$)

3. **Frecuencia de Muestreo**: Debe ser apropiada para capturar la dinámica temporal
   - Muy frecuente: puede capturar ruido
   - Muy espaciada: puede perder dependencias

4. **Ausencia de Valores Atípicos Extremos**: Los outliers pueden distorsionar las estimaciones de parámetros

### Transformaciones Comunes

Si la serie no es estacionaria:
- **Diferenciación**: $\nabla y_t = y_t - y_{t-1}$
- **Transformación logarítmica**: Para estabilizar varianza
- **Transformación Box-Cox**: Para normalizar la distribución

## Patrones ACF/PACF Característicos

### Identificación Visual

| Orden | ACF | PACF |
|-------|-----|------|
| AR(1) | Decaimiento exponencial | Spike en lag 1, luego 0 |
| AR(2) | Decaimiento sinusoidal o exponencial | Spikes en lags 1 y 2, luego 0 |
| AR(p) | Decaimiento gradual | Spikes hasta lag p, luego 0 |

### Ejemplo AR(1) con φ₁ = 0.7

```
ACF:  ▓▓▓▓▓▓▓ (lag 1)
      ▓▓▓▓▓   (lag 2)
      ▓▓▓     (lag 3)
      ▓▓      (lag 4)
      
PACF: ▓▓▓▓▓▓▓ (lag 1)
      ·       (lag 2)
      ·       (lag 3)
      ·       (lag 4)
```

## Casos de Uso y Aplicaciones

### Aplicaciones Comunes

1. **Economía y Finanzas**
   - Modelado de tasas de interés
   - Precios de acciones (versión débil)
   - Índices económicos con inercia

2. **Climatología**
   - Temperatura diaria/mensual
   - Precipitaciones con memoria
   - Niveles de agua en reservorios

3. **Demografía**
   - Crecimiento poblacional
   - Tasas de natalidad/mortalidad

4. **Ingeniería**
   - Señales con autocorrelación
   - Procesos industriales con inercia
   - Sistemas de control

### Cuándo Usar AR

✅ **Usar AR cuando:**
- La serie muestra dependencia de sus valores pasados
- PACF muestra corte claro
- Los efectos de shocks se disipan gradualmente
- Hay "memoria" o inercia en el sistema

❌ **NO usar AR cuando:**
- La serie tiene componentes estacionales fuertes (usar SAR)
- Los shocks tienen efectos inmediatos que desaparecen rápido (usar MA)
- Hay tendencia no estacionaria (aplicar diferenciación primero)

## Ventajas y Limitaciones

### Ventajas

1. **Interpretabilidad**: Los parámetros tienen significado directo
2. **Simplicidad**: Fácil de entender y estimar
3. **Parsimonia**: Pocos parámetros pueden capturar estructuras complejas
4. **Predicción Eficiente**: Buena performance para horizontes cortos
5. **Base Teórica Sólida**: Bien fundamentado matemáticamente

### Limitaciones

1. **Requiere Estacionariedad**: No funciona con series con tendencia
2. **Orden Desconocido**: Requiere identificación del orden p
3. **No Captura Shocks**: No modela efectos inmediatos de innovaciones
4. **Horizonte de Predicción**: Las predicciones convergen a la media para horizontes largos
5. **Sensibilidad a Outliers**: Valores atípicos pueden afectar las estimaciones

## Estimación de Parámetros

### Métodos Principales

1. **Maximum Likelihood Estimation (MLE)**
   - Método óptimo asintóticamente
   - Usado en esta implementación
   - Requiere optimización numérica

2. **Yule-Walker (Método de Momentos)**
   - Basado en ecuaciones de autocorrelación
   - Más rápido pero menos eficiente
   - Útil para valores iniciales

3. **Mínimos Cuadrados**
   - Regresión de $y_t$ en $y_{t-1}, \ldots, y_{t-p}$
   - Simple pero ignora estructura de errores

## Diagnóstico del Modelo

### Verificaciones Post-Ajuste

1. **Residuos**:
   - Deben parecer ruido blanco
   - Media cercana a 0
   - Varianza constante

2. **ACF de Residuos**:
   - No debe mostrar correlación significativa
   - Ljung-Box test: $p > 0.05$

3. **Normalidad**:
   - Q-Q plot debe ser aproximadamente lineal
   - Jarque-Bera test

4. **Información Criteria**:
   - AIC, BIC para comparar modelos
   - Menor valor indica mejor ajuste

## Predicción

### Predicción Multi-paso

Para predecir $h$ pasos adelante:

$$\hat{y}_{t+h|t} = c + \phi_1 \hat{y}_{t+h-1|t} + \cdots + \phi_p \hat{y}_{t+h-p|t}$$

Donde $\hat{y}_{t+k|t} = y_{t+k}$ si $k \leq 0$ (valores observados).

### Intervalos de Confianza

La varianza del error de predicción aumenta con el horizonte:

$$\text{Var}(\varepsilon_{t+h}) = \sigma^2 \left(1 + \sum_{i=1}^{h-1} \psi_i^2 \right)$$

Donde $\psi_i$ son los coeficientes de la representación MA(∞) del proceso AR.

## Referencias y Lecturas Recomendadas

1. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
2. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer.
3. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
4. Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley.

## Ejemplo Práctico

### Serie Sintética AR(2)

```python
from tslib.models import ARModel
import numpy as np

# Generate AR(2) data: y_t = 0.5*y_{t-1} - 0.3*y_{t-2} + ε_t
np.random.seed(42)
n = 200
y = np.zeros(n)
epsilon = np.random.normal(0, 1, n)

for t in range(2, n):
    y[t] = 0.5*y[t-1] - 0.3*y[t-2] + epsilon[t]

# Fit AR model with automatic order selection
model = ARModel(auto_select=True, max_order=5)
model.fit(y)

print(model.summary())
# Expected: AR(2) with φ₁ ≈ 0.5, φ₂ ≈ -0.3

# Forecast
forecast = model.predict(steps=10)
```

## Notas Adicionales

- El modelo AR es un caso especial de ARMA(p,0)
- Todo proceso AR estacionario tiene una representación MA(∞)
- La función de impulso-respuesta decae exponencialmente
- Los modelos AR son ampliamente usados como benchmark en forecasting

