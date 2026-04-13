# Modelo Autoregresivo de Media Móvil (ARMA)

## Definición Matemática

Un proceso Autoregresivo de Media Móvil de orden (p,q), denotado como **ARMA(p,q)**, combina componentes autorregresivos (AR) y de media móvil (MA) en un solo modelo unificado.

### Ecuación General

$$y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}$$

Donde:
- $y_t$ es el valor de la serie en el tiempo $t$
- $c$ es una constante (intercepto)
- $\phi_1, \ldots, \phi_p$ son los parámetros autorregresivos (componente AR)
- $\theta_1, \ldots, \theta_q$ son los parámetros de media móvil (componente MA)
- $\varepsilon_t \sim N(0, \sigma^2)$ es ruido blanco
- $p$ es el orden AR
- $q$ es el orden MA

### Forma Compacta con Operador de Rezago

Usando el operador de rezago $B$:

$$\Phi(B) y_t = c + \Theta(B) \varepsilon_t$$

Donde:
- $\Phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ (polinomio AR)
- $\Theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ (polinomio MA)

### Forma Equivalente

$$y_t = c + \frac{\Theta(B)}{\Phi(B)} \varepsilon_t$$

Esta expresión muestra que ARMA es un "filtro" racional aplicado al ruido blanco.

## Propiedades Estadísticas

### Estacionariedad e Invertibilidad

Un proceso ARMA(p,q) requiere **dos condiciones**:

1. **Estacionariedad** (del componente AR):
   - Todas las raíces de $\Phi(z) = 0$ deben estar fuera del círculo unitario
   - Garantiza que la serie no explote y tenga momentos finitos

2. **Invertibilidad** (del componente MA):
   - Todas las raíces de $\Theta(z) = 0$ deben estar fuera del círculo unitario
   - Garantiza representación única y posibilidad de estimación

### Media y Varianza

Para un proceso ARMA(p,q) estacionario:

**Media:**
$$E[y_t] = \mu = \frac{c}{1 - \phi_1 - \phi_2 - \cdots - \phi_p}$$

**Varianza:**
La varianza se obtiene resolviendo las ecuaciones de Yule-Walker modificadas:

$$\gamma_0 - \phi_1\gamma_1 - \cdots - \phi_p\gamma_p = \sigma^2[1 + \theta_1\psi_1 + \cdots + \theta_q\psi_q]$$

Donde $\psi_i$ son coeficientes de la representación MA(∞) del proceso.

### Función de Autocorrelación (ACF)

La ACF de un ARMA(p,q):
- **No se corta abruptamente** (a diferencia de MA puro)
- **Decae exponencialmente o con patrón sinusoidal** después de los primeros q - p lags
- Comportamiento mezcla características de AR y MA

**Ecuaciones de Yule-Walker para ARMA:**

Para $k > q$:
$$\rho_k = \phi_1\rho_{k-1} + \phi_2\rho_{k-2} + \cdots + \phi_p\rho_{k-p}$$

### Función de Autocorrelación Parcial (PACF)

La PACF de un ARMA(p,q):
- **No se corta abruptamente** (a diferencia de AR puro)
- **Decae exponencialmente o con patrón sinusoidal** después de los primeros p - q lags
- Comportamiento mixto entre AR y MA

## Requisitos de los Datos

### Características Necesarias

1. **Estacionariedad**: La serie debe ser estacionaria
   - Media constante en el tiempo
   - Varianza constante
   - Autocovarianza que solo depende del rezago
   - Si no es estacionaria, aplicar transformaciones primero

2. **Longitud Mínima**:
   - **Mínimo absoluto**: $n > p + q$ (más observaciones que parámetros)
   - **Recomendado**: $n \geq 4 \times \max(p, q)$ para estimaciones confiables
   - **Óptimo**: $n \geq 80$ para órdenes moderados ($p, q \leq 3$)

3. **Complejidad Moderada**:
   - Evitar órdenes muy altos (p, q > 5)
   - En la práctica, raramente se necesita p + q > 5
   - Órdenes altos indican posible sobreajuste

4. **Ausencia de Patrones Adicionales**:
   - No debe haber estacionalidad (usar SARMA en ese caso)
   - No debe haber cambios estructurales
   - Varianza debe ser homogénea

### Naturaleza de las Series Apropiadas

Los modelos ARMA son apropiados para series que:
- Tienen autocorrelación de corto y mediano plazo
- Muestran tanto inercia (AR) como respuesta a shocks (MA)
- Son estacionarias o estacionarizadas
- Requieren flexibilidad para capturar estructuras complejas

## Patrones ACF/PACF Característicos

### Identificación Visual

| Modelo | ACF | PACF |
|--------|-----|------|
| ARMA(1,1) | Decae exponencialmente desde lag 1 | Decae exponencialmente desde lag 1 |
| ARMA(2,1) | Decae exponencialmente/sinusoidal | Decae después de lag 2 |
| ARMA(1,2) | Decae después de lag 2 | Decae exponencialmente/sinusoidal |
| ARMA(p,q) | Decaimiento gradual después de lag q-p | Decaimiento gradual después de lag p-q |

### Característica Distintiva

**Ambas funciones (ACF y PACF) decaen gradualmente** sin corte abrupto. Esto distingue ARMA de AR puro o MA puro.

### Ejemplo ARMA(1,1)

```
ACF:  ▓▓▓▓▓▓ (lag 1)
      ▓▓▓▓   (lag 2)
      ▓▓     (lag 3)
      ▓      (lag 4)
      
PACF: ▓▓▓▓▓▓ (lag 1)
      ▓▓▓    (lag 2)
      ▓▓     (lag 3)
      ▓      (lag 4)
```

## Casos de Uso y Aplicaciones

### Aplicaciones Comunes

1. **Economía y Finanzas**
   - Retornos de activos financieros
   - Tasas de interés con shocks y persistencia
   - Variables macroeconómicas complejas
   - Índices de volatilidad

2. **Ciencias Ambientales**
   - Niveles de contaminación
   - Caudales de ríos
   - Variables climáticas con múltiples factores

3. **Telecomunicaciones**
   - Tráfico de red
   - Demanda de ancho de banda
   - Patrones de llamadas

4. **Producción Industrial**
   - Demanda de productos
   - Inventarios
   - Utilización de capacidad

5. **Epidemiología**
   - Incidencia de enfermedades
   - Tasas de contagio
   - Efectos de intervenciones

### Cuándo Usar ARMA

✅ **Usar ARMA cuando:**
- Ni ACF ni PACF muestran corte claro
- Ambas funciones decaen gradualmente
- Se necesita capturar tanto inercia como shocks
- Modelos puros AR o MA no ajustan bien
- Hay estructura de correlación compleja pero estacionaria

❌ **NO usar ARMA cuando:**
- AR o MA puro es suficiente (parsimonia)
- La serie tiene tendencia (usar ARIMA)
- Hay componentes estacionales fuertes (usar SARMA o SARIMA)
- Los datos no son estacionarios

## Ventajas y Limitaciones

### Ventajas

1. **Flexibilidad**: Puede capturar estructuras complejas con pocos parámetros
2. **Parsimonia**: Combina AR y MA eficientemente (p + q < max(p AR puro, q MA puro))
3. **Generalidad**: AR y MA son casos especiales
4. **Representación Doble**: Todo ARMA tiene representaciones AR(∞) y MA(∞)
5. **Base Teórica**: Fundamentado en teorema de Wold

### Limitaciones

1. **Identificación Compleja**: Determinar (p,q) es más difícil que para AR o MA
2. **Estimación Compleja**: Más parámetros que estimar
3. **Riesgo de Sobreajuste**: Fácil incluir demasiados parámetros
4. **Requiere Estacionariedad**: No funciona con series no estacionarias
5. **Identificabilidad**: Puede haber parámetros redundantes (cancelación de raíces)

## Estimación de Parámetros

### Métodos Principales

1. **Maximum Likelihood Estimation (MLE)**
   - Método óptimo y estándar
   - Usado en esta implementación
   - Combina algoritmo de innovaciones con optimización numérica
   - Proporciona estimaciones eficientes asintóticamente

2. **Mínimos Cuadrados Condicionales**
   - Condicional en primeras observaciones
   - Más rápido pero menos eficiente

3. **Mínimos Cuadrados No Lineales**
   - Minimiza suma de cuadrados de residuos
   - Similar a MLE para errores normales

### Algoritmo de Estimación

1. Obtener valores iniciales (Hannan-Rissanen o método de momentos)
2. Usar algoritmo de innovaciones para calcular log-verosimilitud
3. Optimizar numéricamente (BFGS, Levenberg-Marquardt)
4. Verificar convergencia y condiciones de estacionariedad/invertibilidad

## Selección de Orden (p,q)

### Métodos de Identificación

1. **Inspección ACF/PACF**:
   - Buscar patrones de decaimiento
   - Identificar lags significativos

2. **Criterios de Información**:
   - **AIC**: $AIC = -2\ln(L) + 2(p+q+1)$
   - **BIC**: $BIC = -2\ln(L) + (p+q+1)\ln(n)$
   - Buscar el mínimo sobre una grilla de valores (p,q)

3. **Validación Cruzada**:
   - Dividir datos en entrenamiento y prueba
   - Evaluar performance predictiva

4. **Parsimonia**:
   - Preferir modelos simples si performance es similar
   - Regla: $p + q \leq 5$ en la mayoría de aplicaciones

### Grid Search

Estrategia típica:
```python
best_aic = inf
for p in range(max_p + 1):
    for q in range(max_q + 1):
        if p == 0 and q == 0:
            continue
        model = ARMA(p, q)
        model.fit(data)
        if model.aic < best_aic:
            best_aic = model.aic
            best_order = (p, q)
```

## Diagnóstico del Modelo

### Verificaciones Post-Ajuste

1. **Residuos**:
   - ACF de residuos debe ser no significativa
   - Ljung-Box test: $H_0$: residuos son ruido blanco
   - Test de normalidad (Jarque-Bera, Shapiro-Wilk)

2. **Estabilidad de Parámetros**:
   - Verificar estacionariedad: raíces de $\Phi(z)$ fuera del círculo unitario
   - Verificar invertibilidad: raíces de $\Theta(z)$ fuera del círculo unitario

3. **Significancia de Parámetros**:
   - t-tests para cada parámetro
   - Eliminar parámetros no significativos

4. **Análisis de Residuos**:
   - Q-Q plot para normalidad
   - Plot de residuos vs tiempo (homocedasticidad)
   - No debe haber patrones

## Predicción

### Predicción Multi-paso

La predicción óptima para $h$ pasos adelante es:

$$\hat{y}_{t+h|t} = E[y_{t+h} | y_t, y_{t-1}, \ldots]$$

Se calcula recursivamente:

$$\hat{y}_{t+h|t} = c + \sum_{i=1}^p \phi_i \hat{y}_{t+h-i|t} + \sum_{j=h}^q \theta_j \hat{\varepsilon}_{t+h-j|t}$$

Donde:
- $\hat{y}_{t+k|t} = y_{t+k}$ si $k \leq 0$ (valores observados)
- $\hat{\varepsilon}_{t+k|t} = 0$ si $k > 0$ (errores futuros desconocidos)

### Varianza de Predicción

$$\text{Var}(e_{t+h}) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$$

Donde $\psi_j$ son los coeficientes de la representación MA(∞):

$$\Phi(B)^{-1}\Theta(B) = \sum_{j=0}^\infty \psi_j B^j$$

### Horizonte de Predicción

- **Corto plazo** (h < q): Influenciado por componente MA
- **Mediano plazo** (q ≤ h < horizonte AR): Transición
- **Largo plazo** (h → ∞): Converge a la media $\mu$

## Relación con Otros Modelos

### Casos Especiales

- **ARMA(p,0) = AR(p)**: Sin componente MA
- **ARMA(0,q) = MA(q)**: Sin componente AR
- **ARMA(0,0)**: Ruido blanco

### Representaciones Equivalentes

Por el teorema de Wold y la invertibilidad:
- Todo ARMA estacionario tiene representación MA(∞)
- Todo ARMA invertible tiene representación AR(∞)

### Extensiones

- **ARIMA(p,d,q)**: ARMA con diferenciación (d veces)
- **SARMA(p,q)(P,Q)s**: ARMA con componente estacional
- **VARMA**: Versión multivariada

## Referencias y Lecturas Recomendadas

1. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
2. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer.
3. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
4. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer.

## Ejemplo Práctico

### Serie Sintética ARMA(2,1)

```python
from tslib.models import ARMAModel
import numpy as np

# Generate ARMA(2,1) data
# y_t = 0.6*y_{t-1} - 0.3*y_{t-2} + ε_t + 0.5*ε_{t-1}
np.random.seed(42)
n = 200
phi = [0.6, -0.3]
theta = [0.5]
epsilon = np.random.normal(0, 1, n)

y = np.zeros(n)
for t in range(2, n):
    y[t] = (phi[0]*y[t-1] + phi[1]*y[t-2] + 
            epsilon[t] + theta[0]*epsilon[t-1])

# Fit ARMA model with automatic order selection
model = ARMAModel(auto_select=True, max_ar=5, max_ma=5)
model.fit(y)

print(model.summary())
# Expected: ARMA(2,1) with φ₁ ≈ 0.6, φ₂ ≈ -0.3, θ₁ ≈ 0.5

# Forecast
forecast, conf_int = model.predict(steps=20, return_conf_int=True)

# Check convergence to mean
print(f"Long-term forecast converges to mean: {np.mean(y):.2f}")
```

## Notas Adicionales

- ARMA es el modelo fundamental en la metodología Box-Jenkins
- En la práctica, ARMA(1,1) y ARMA(2,1) son muy comunes
- La identificación de (p,q) es más arte que ciencia
- Siempre verificar supuestos sobre residuos
- Preferir parsimonia: el modelo más simple que ajuste bien
- ARMA es la base para entender ARIMA y modelos más complejos

