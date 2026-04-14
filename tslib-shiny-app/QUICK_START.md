# 🚀 Inicio Rápido - TSLib Shiny App

## Instalación en 3 Pasos

```bash
# 1. Activar entorno virtual
cd /path/to/tslib-shiny-app
source venv/bin/activate

# 2. Instalar TSLib (ajusta la ruta a tu clon de time-series-library)
pip install -e /path/to/time-series-library

# 3. Instalar dependencias faltantes
pip install openpyxl pyspark
```

## Ejecutar Aplicación

```bash
python app.py
```

Abrir navegador en: **http://localhost:8000**

## Probar Integración

```bash
python test_tslib_integration.py
```

## Flujo (4 pasos)

1. **📁 Carga** – Subir CSV/Excel, elegir columna de valores (y opcional fecha), validar datos → Siguiente
2. **📊 Exploración** – Ver serie, estadísticas, ACF/PACF → Siguiente
3. **⚙️ Modelo y ejecución** – Elegir AR/MA/ARMA/ARIMA, configurar (auto o manual), "Ajustar y pronosticar" → Siguiente
4. **📈 Resultados** – Métricas, pronóstico, diagnósticos, exportar

## Datos de Ejemplo

En `data/examples/`:
- `sales.csv` – ventas
- `temperature.csv` – temperatura
- `dummy_with_missing.csv` – serie con valores faltantes (para probar imputación)
- `generate_dummy_data.py` – script para generar más datos de prueba

## Modelos Disponibles

| Modelo | Mejor Para | Auto-Selección |
|--------|-----------|----------------|
| **AR** | Series con persistencia | ✅ Sí |
| **MA** | Shocks transitorios | ✅ Sí |
| **ARMA** | Estructuras complejas | ✅ Sí |
| **ARIMA** | Series con tendencia | ✅ Sí |

## Solución de Problemas

### Error: ModuleNotFoundError: No module named 'tslib'
```bash
pip install -e /path/to/time-series-library
```

### Error: ModuleNotFoundError: No module named 'openpyxl'
```bash
pip install openpyxl
```

### Error: ModuleNotFoundError: No module named 'pyspark'
```bash
pip install pyspark
```

### La validación falla
- TSLib exige al menos **3** observaciones y, fuera del corte estricto de la app, como mucho **10 %** de faltantes (`DataValidator`).
- En el asistente, **cualquier NaN** en la columna de valores invalida el flujo hasta completar la serie.

### Los gráficos no aparecen
- Espera unos segundos después de cambiar de paso
- Verifica que seleccionaste una columna de valores
- Asegúrate de haber validado los datos primero

## Documentación

- **`README.md`** – Visión general e integración con TSLib
- **`INTEGRATION_README.md`** – Guía detallada de uso y estado
- **TSLib** – Ver `README.md` en el repositorio time-series-library

## Características Principales

✅ Carga CSV y Excel  
✅ Validación automática con TSLib  
✅ 4 modelos de series temporales  
✅ Auto-selección de parámetros  
✅ Gráficos interactivos  
✅ Intervalos de confianza  
✅ Diagnósticos completos  
✅ Exportación de resultados  

## Contacto y Soporte

Para problemas o preguntas:
1. Revisar `INTEGRATION_README.md`
2. Ejecutar `test_tslib_integration.py`
3. Verificar logs en terminal
4. Consultar documentación de TSLib

---

**¡Listo para analizar series temporales! 🎉**

