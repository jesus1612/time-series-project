# TSLib Shiny App - Guía de Integración

## Resumen de la Integración

Se ha integrado exitosamente la librería TSLib en la aplicación Shiny para análisis de series temporales. La aplicación ahora soporta análisis completo con los siguientes modelos:

- **AR** (Autoregresivo)
- **MA** (Media Móvil)
- **ARMA** (Combinado AR + MA)
- **ARIMA** (Integrado con diferenciación)

## Instalación

### 1. Instalar TSLib

```bash
cd /path/to/tslib-shiny-app
source venv/bin/activate
pip install -e /path/to/time-series-library
```

### 2. Instalar dependencias adicionales

```bash
pip install openpyxl pyspark
```

**Nota:** PySpark se instala como dependencia de TSLib aunque no se usa directamente en esta aplicación.

## Ejecución

```bash
cd /path/to/tslib-shiny-app
source venv/bin/activate
python app.py
```

La aplicación estará disponible en: `http://localhost:8000`

## Flujo de Trabajo

### Paso 1: Carga de Datos 📁

1. **Subir archivo**: CSV o Excel (.xlsx, .xls)
2. **Seleccionar columna de valores**: La serie temporal a analizar
3. **Seleccionar columna de fecha** (opcional): Para el eje X en gráficos
4. **Validar datos**: TSLib validará:
   - Longitud mínima de datos
   - Valores faltantes
   - Outliers
   - Calidad general de los datos

**Validación exitosa requerida** para avanzar al siguiente paso.

### Paso 2: Visualización 📊

- **Gráfico de serie temporal**: Visualización interactiva de tus datos
- **Estadísticas básicas**: Media, desviación estándar, mínimo, máximo
- **ACF y PACF**: Análisis de autocorrelación para identificar patrones

### Paso 3: Modelo y ejecución ⚙️

1. **Elegir tipo de modelo**:
   - **AR**: Para series con persistencia (valores pasados influyen)
   - **MA**: Para modelar shocks transitorios
   - **ARMA**: Para estructuras complejas estacionarias
   - **ARIMA**: Para series con tendencia (no estacionarias)

2. **Configuración**:
   - **Auto-selección**: TSLib selecciona automáticamente los parámetros óptimos
   - **Manual**: Especifica los órdenes p, d, q según el modelo

3. **Opciones adicionales**:
   - Pasos a pronosticar (default: 10)
   - Incluir intervalos de confianza

La **ejecución** (ajustar modelo y pronóstico) se realiza en el **Paso 3** (Modelo y ejecución), con el botón "Ajustar y pronosticar". El proceso incluye: ajuste del modelo TSLib, generación de pronósticos y, para ARIMA, opcionalmente el modelo paralelo (Spark o fallback lineal).

### Paso 4: Resultados 📈

**Información del Modelo:**
- Tipo de modelo ajustado
- Orden seleccionado/calculado

**Métricas de Evaluación:**
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)

**Pronóstico:**
- Gráfico histórico + predicciones
- Intervalos de confianza al 95%
- Tabla con valores numéricos

**Diagnósticos:**
- Gráfico de residuos
- ACF de residuos (validar ruido blanco)

**Exportar:** Botón para exportar resultados a CSV

Los **reportes** (generar reporte / PDF) están en desarrollo; la UI está en `features/reports/ui.py` pero no forma parte del stepper actual.

## Arquitectura de la Integración

### Servicio TSLib (`services/tslib_service.py`)

Capa de abstracción que encapsula toda la interacción con TSLib:

```python
from services.tslib_service import TSLibService

service = TSLibService()

# Validar datos
validation = service.validate_data(df, column="valores")

# Detectar columna de fecha
date_col = service.detect_datetime_column(df)

# Ajustar modelo
model = service.fit_model(
    data=data,
    model_type="ARIMA",
    order=(1, 1, 1),
    auto_select=True
)

# Generar pronóstico
forecast = service.get_forecast(model, steps=10, return_conf_int=True)

# Obtener métricas
metrics = service.get_model_metrics(model)
```

### Estado de la Aplicación

```python
app_state = {
    "current_step": int,          # 0-3 (4 pasos)
    "data_loaded": bool,
    "data_validated": bool,
    "uploaded_data": dict,
    "value_column": str,
    "date_column": str,
    "model_type": str,            # AR, MA, ARMA, ARIMA
    "model_config": dict,
    "fitted_model": Model,
    "forecast_results": dict,
    "parallel_workflow": Any,     # Solo para ARIMA (real o fallback)
    "parallel_forecast_results": dict,
    "analysis_complete": bool,
    "execution_log": list,
    "exploratory_analysis": dict  # ACF/PACF
}
```

### Validación de Pasos

- **Paso 0 → 1**: Datos cargados, columna de valores seleccionada, validación exitosa
- **Paso 1 → 2**: Datos validados
- **Paso 2 → 3**: Análisis completado (modelo ajustado en el mismo paso 2)
- **Paso 3**: Resultados (solo lectura)

## Modelos TSLib Soportados

### AR Model - Autoregresivo

```python
from tslib import ARModel

model = ARModel(auto_select=True, max_order=5)
model.fit(data)
forecast = model.predict(steps=10, return_conf_int=True)
```

**Cuándo usar:** Series con autocorrelación, persistencia (economía, finanzas).

### MA Model - Media Móvil

```python
from tslib import MAModel

model = MAModel(auto_select=True, max_order=5)
model.fit(data)
forecast = model.predict(steps=10)
```

**Cuándo usar:** Shocks transitorios, errores de pronóstico.

### ARMA Model - Combinado

```python
from tslib import ARMAModel

model = ARMAModel(auto_select=True, max_ar=5, max_ma=5)
model.fit(data)
forecast = model.predict(steps=10, return_conf_int=True)
```

**Cuándo usar:** Series estacionarias con estructura compleja.

### ARIMA Model - Integrado

```python
from tslib import ARIMAModel

model = ARIMAModel(auto_select=True, max_p=5, max_d=2, max_q=5)
model.fit(data)
forecast = model.predict(steps=10, return_conf_int=True)
```

**Cuándo usar:** Series no estacionarias con tendencia.

## Archivos Modificados/Creados

### Nuevos Archivos

1. **`services/tslib_service.py`**: Servicio de integración con TSLib
2. **`services/__init__.py`**: Módulo de servicios

### Archivos Actualizados

1. **`requirements.txt`**: Agregado openpyxl y nota sobre TSLib
2. **`app.py`**: Lógica completa de integración (~1600 líneas)
3. **`features/upload/ui.py`**: Selectores de columnas y validación
4. **`features/visualization/ui.py`**: Gráficos reales con matplotlib
5. **`features/model_selection/ui.py`**: Selector de modelos y parámetros dinámicos
6. **`features/execution/ui.py`**: UI de ejecución actualizada
7. **`features/results/ui.py`**: Visualización completa de resultados

## Características Implementadas

✅ **Carga de Datos**
- Soporte CSV y Excel
- Detección automática de columnas
- Selección manual de columnas

✅ **Validación TSLib**
- DataValidator integrado
- Mensajes de warnings y errores
- Análisis exploratorio (ACF/PACF)

✅ **Visualización**
- Gráficos de serie temporal
- Estadísticas básicas
- ACF y PACF plots
- Tema oscuro consistente

✅ **Selección de Modelo**
- 4 tipos de modelos (AR, MA, ARMA, ARIMA)
- Auto-selección de orden
- Configuración manual de parámetros
- Parámetros dinámicos según modelo

✅ **Ejecución**
- Integración con TSLib real
- Log de progreso
- Manejo de errores

✅ **Resultados**
- Métricas AIC/BIC
- Gráfico de pronóstico con IC
- Tabla de valores
- Diagnósticos de residuos
- ACF de residuos

✅ **Navegación**
- Validación por paso
- Estado reactivo
- Notificaciones

## Notas Técnicas

- Matplotlib se usa con backend `Agg` para Shiny.
- Gráficos con tema oscuro de la app.
- Estados reactivos; la validación de pasos impide avanzar sin cumplir requisitos.
- **ARIMA paralelo**: si PySpark y Java están disponibles, se usa `ParallelARIMAWorkflow` de TSLib; si no, un flujo lineal de respaldo (sin Spark) permite seguir usando ARIMA.

## Soporte

Para problemas o mejoras:
1. Verificar que TSLib esté instalado correctamente
2. Revisar logs de la aplicación
3. Verificar que los datos cumplan requisitos mínimos (>30 observaciones)
4. Consultar documentación de TSLib para modelos específicos

