# Guía de Integración de TSLib

Esta guía te muestra cómo integrar la librería TSLib en tu proyecto de interfaz.

## 1. Instalación en tu Proyecto

### Opción A: Instalación en Modo Editable (Recomendada)

Desde el directorio de tu proyecto de interfaz:

```bash
# Activate your virtual environment
cd /path/to/your-interface-project
source venv/bin/activate

# Instala tslib en modo editable
pip install -e /path/to/time-series-library

# Verifica la instalación
python -c "from tslib import ARModel, ARIMAModel; print('✓ TSLib instalado correctamente')"
```

**Ventajas:**
- Los cambios en TSLib se reflejan automáticamente
- No necesitas reinstalar después de modificaciones
- Ideal para desarrollo activo

### Opción B: Agregar a requirements.txt

Edita o crea el archivo `requirements.txt` de tu proyecto:

```txt
# requirements.txt de tu proyecto de interfaz
-e /path/to/time-series-library

# Dependencias de tu interfaz
flask>=2.0.0  # o fastapi, django, etc.
pandas>=2.0.0
numpy>=1.24.0
```

Luego instala:

```bash
pip install -r requirements.txt
```

## 2. Estructura Básica de Uso

### Importaciones

```python
# Importar modelos
from tslib import ARModel, MAModel, ARMAModel, ARIMAModel

# Librerías auxiliares
import pandas as pd
import numpy as np
```

### Patrón Básico de Uso

```python
# 1. Preparar datos
data = np.array([...])  # o pd.Series([...])

# 2. Crear modelo
model = ARIMAModel(
    auto_select=True,  # Selección automática de orden
    max_p=3,           # Máximo orden AR
    max_d=2,           # Máximo orden de diferenciación
    max_q=3,           # Máximo orden MA
    validation=True    # Validar datos de entrada
)

# 3. Ajustar modelo
model.fit(data)

# 4. Generar pronóstico
forecast = model.predict(steps=6)

# 5. Obtener diagnósticos
summary = model.summary()
diagnostics = model.get_residual_diagnostics()
```

## 3. Ejemplo con Datos de Cash Flow

### Preparación de Datos

```python
import pandas as pd

def prepare_cash_flow_data(csv_path):
    """Preparar datos de flujo de efectivo"""
    # Cargar CSV
    df = pd.read_csv(csv_path)
    
    # Limpiar valores monetarios
    df['Cash In'] = df['Cash In'].str.replace('$', '').str.replace(',', '').astype(float)
    df['Cash Out'] = df['Cash Out'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Calcular flujo neto
    df['Net Cash Flow'] = df['Cash In'] - df['Cash Out']
    
    return df
```

### Análisis y Pronóstico

```python
from tslib import ARIMAModel

def analyze_cash_flow(data_series, forecast_months=6):
    """
    Analizar serie temporal y generar pronóstico
    
    Parameters:
    -----------
    data_series : np.ndarray or pd.Series
        Serie temporal de datos
    forecast_months : int
        Número de meses a pronosticar
    
    Returns:
    --------
    dict : Resultados del análisis
    """
    # Ajustar modelo con selección automática
    model = ARIMAModel(auto_select=True, validation=True)
    model.fit(data_series)
    
    # Generar pronóstico
    forecast = model.predict(steps=forecast_months)
    
    # Obtener orden del modelo
    p, d, q = model.order
    
    # Diagnósticos
    residuals = model.get_residuals()
    diagnostics = model.get_residual_diagnostics()
    
    return {
        'model_order': (p, d, q),
        'forecast': forecast.tolist(),
        'residuals_mean': float(residuals.mean()),
        'residuals_std': float(residuals.std()),
        'diagnostics': diagnostics,
        'summary': model.summary()
    }
```

### Ejemplo Completo de API Flask

```python
from flask import Flask, request, jsonify
from tslib import ARIMAModel
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/api/forecast', methods=['POST'])
def forecast_endpoint():
    """
    Endpoint para generar pronósticos
    
    Expects JSON:
    {
        "data": [1.5, 2.3, 1.8, ...],
        "steps": 6,
        "auto_select": true
    }
    """
    try:
        # Obtener datos del request
        request_data = request.get_json()
        data = np.array(request_data['data'])
        steps = request_data.get('steps', 6)
        auto_select = request_data.get('auto_select', True)
        
        # Validar datos
        if len(data) < 10:
            return jsonify({
                'error': 'Se requieren al menos 10 observaciones'
            }), 400
        
        # Crear y ajustar modelo
        model = ARIMAModel(auto_select=auto_select, validation=True)
        model.fit(data)
        
        # Generar pronóstico
        forecast = model.predict(steps=steps)
        
        # Obtener diagnósticos
        p, d, q = model.order
        residuals = model.get_residuals()
        
        # Preparar respuesta
        response = {
            'success': True,
            'model': {
                'type': 'ARIMA',
                'order': {'p': p, 'd': d, 'q': q}
            },
            'forecast': forecast.tolist(),
            'diagnostics': {
                'residuals_mean': float(residuals.mean()),
                'residuals_std': float(residuals.std()),
                'n_observations': len(data)
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    """
    Endpoint para comparar múltiples modelos
    
    Expects JSON:
    {
        "data": [1.5, 2.3, 1.8, ...],
        "test_size": 0.2
    }
    """
    try:
        from tslib import ARModel, MAModel, ARMAModel, ARIMAModel
        
        request_data = request.get_json()
        data = np.array(request_data['data'])
        test_size = request_data.get('test_size', 0.2)
        
        # Split train/test
        split_point = int(len(data) * (1 - test_size))
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        # Modelos a probar
        models = {
            'AR': ARModel(auto_select=True, validation=False),
            'MA': MAModel(auto_select=True, validation=False),
            'ARMA': ARMAModel(auto_select=True, validation=False),
            'ARIMA': ARIMAModel(auto_select=True, validation=False)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Ajustar
                model.fit(train_data)
                
                # Pronosticar
                forecast = model.predict(steps=len(test_data))
                
                # Calcular error
                rmse = float(np.sqrt(np.mean((test_data - forecast) ** 2)))
                mae = float(np.mean(np.abs(test_data - forecast)))
                
                # Obtener orden
                if isinstance(model.order, tuple):
                    order = model.order
                else:
                    order = (model.order,)
                
                results[name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'order': order
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        # Encontrar mejor modelo
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        if valid_models:
            best_model = min(valid_models.keys(), key=lambda k: valid_models[k]['rmse'])
        else:
            best_model = None
        
        return jsonify({
            'success': True,
            'results': results,
            'best_model': best_model
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## 4. Modelos Disponibles

### AR (AutoRegressive)

```python
from tslib import ARModel

model = ARModel(
    order=3,           # AR(3) o None para auto-selección
    auto_select=True,  # Selección automática basada en PACF
    max_order=5,       # Máximo orden a probar
    validation=True    # Validar datos
)
```

**Uso recomendado:** Series con autocorrelación fuerte y persistente

### MA (Moving Average)

```python
from tslib import MAModel

model = MAModel(
    order=2,           # MA(2) o None para auto-selección
    auto_select=True,  # Selección automática basada en ACF
    max_order=5,       # Máximo orden a probar
    validation=True
)
```

**Uso recomendado:** Series con efectos de shocks transitorios

### ARMA (AutoRegressive Moving Average)

```python
from tslib import ARMAModel

model = ARMAModel(
    order=(2, 3),      # ARMA(2,3) o None para auto-selección
    auto_select=True,  # Selección automática con grid search
    max_p=3,           # Máximo orden AR
    max_q=3,           # Máximo orden MA
    validation=True
)
```

**Uso recomendado:** Series estacionarias con componentes AR y MA

### ARIMA (AutoRegressive Integrated Moving Average)

```python
from tslib import ARIMAModel

model = ARIMAModel(
    order=(1, 1, 1),   # ARIMA(1,1,1) o None para auto-selección
    auto_select=True,  # Selección automática completa
    max_p=3,           # Máximo orden AR
    max_d=2,           # Máximo orden de diferenciación
    max_q=3,           # Máximo orden MA
    validation=True
)
```

**Uso recomendado:** Series no estacionarias (la mayoría de datos reales)

## 5. Métodos Disponibles

Todos los modelos comparten estos métodos:

```python
# Ajustar modelo
model.fit(data)

# Generar pronóstico
forecast = model.predict(steps=6)

# Obtener residuales
residuals = model.get_residuals()

# Obtener valores ajustados
fitted_values = model.get_fitted_values()

# Resumen del modelo
print(model.summary())

# Diagnósticos de residuales
diagnostics = model.get_residual_diagnostics()

# Análisis exploratorio
analysis = model.get_exploratory_analysis()

# Visualizaciones (requiere matplotlib)
model.plot_diagnostics()
model.plot_forecast(steps=12)

# Evaluación de pronóstico
evaluation = model.evaluate_forecast(test_data, steps=len(test_data))
```

## 6. Validación de Datos

TSLib incluye validación automática de datos:

```python
model = ARIMAModel(validation=True)  # Activa validación

# La validación verifica:
# - No hay valores NaN o infinitos
# - Varianza suficiente
# - Longitud mínima de datos
# - Tipos de datos correctos
```

Para datos problemáticos:

```python
from tslib.preprocessing.validation import DataValidator

validator = DataValidator()
results = validator.validate(data)

if not results['is_valid']:
    print("Problemas encontrados:", results['issues'])
    print("Recomendaciones:", results['recommendations'])
```

## 7. Tips de Rendimiento

### Desactivar Validación en Producción

```python
# Para mayor velocidad en producción
model = ARIMAModel(auto_select=True, validation=False)
```

### Limitar Búsqueda de Orden

```python
# Reducir espacio de búsqueda
model = ARIMAModel(
    auto_select=True,
    max_p=2,  # En lugar de 5
    max_d=1,  # En lugar de 2
    max_q=2   # En lugar de 5
)
```

### Usar Orden Fijo

```python
# Si conoces el orden óptimo
model = ARIMAModel(order=(1, 1, 1), auto_select=False)
```

## 8. Manejo de Errores

```python
try:
    model = ARIMAModel(auto_select=True)
    model.fit(data)
    forecast = model.predict(steps=6)
    
except ValueError as e:
    # Error en validación de datos
    print(f"Error de validación: {e}")
    
except RuntimeError as e:
    # Error en optimización
    print(f"Error de optimización: {e}")
    
except Exception as e:
    # Otros errores
    print(f"Error inesperado: {e}")
```

## 9. Ejemplos Completos

Ver ejemplos en:
- `examples/ar_example.py` - Modelo AR
- `examples/ma_example.py` - Modelo MA
- `examples/arma_example.py` - Modelo ARMA
- `examples/basic_arima.py` - Modelo ARIMA
- `examples/test_real_cash_flow.py` - Análisis con datos reales

## 10. Soporte y Documentación

- **Documentación matemática:** `docs/modelos/`
- **README completo:** `README.md`
- **Tests:** `tests/test_arima.py`

## 11. Checklist de Integración

- [ ] Instalar TSLib en modo editable
- [ ] Verificar importaciones funcionan
- [ ] Probar con datos de ejemplo
- [ ] Integrar en endpoint/función principal
- [ ] Agregar manejo de errores
- [ ] Implementar validación de entrada
- [ ] Configurar logging apropiado
- [ ] Optimizar parámetros de rendimiento
- [ ] Crear tests de integración
- [ ] Documentar uso en tu proyecto

