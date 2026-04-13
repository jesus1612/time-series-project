# Documentación: Resultados (Paso 4)

Corresponde al cuarto paso del asistente en la app Shiny: **📈 Resultados** (`current_step == 3`).

Importar en [diagrams.net](https://app.diagrams.net/): **Insertar → Avanzado → Mermaid**.

---

## Diagrama — Origen de datos vs presentación (Paso 4)

Los artefactos de modelado viven en **TSLib** (objetos y vectores); la **App Shiny** los referencia en `app_state`, llama a **TSLibService** para métricas/comparaciones y pinta la interfaz del paso Resultados.

```mermaid
flowchart TB
  subgraph TSLIB["TSLib"]
    direction TB
    FM[Modelo ajustado lineal\n(tslib.models.* / ARIMAProcess)\npredict, residuos, AIC/BIC…]
    FR[Pronóstico lineal e intervalos]
    PS[Paralelo Spark\nworkflow + pronóstico\n(tslib.spark)]
  end

  subgraph SHINY["App Shiny"]
    direction TB
    ST["app_state\n(fitted_model, forecast_results,\nparallel_*, execution_metadata,\nruntime_warnings)"]
    SV[TSLibService\nget_model_metrics, compare_forecasts]
    subgraph UI["Paso 4 — interfaz Resultados"]
      MI[Información del modelo]
      WN[Avisos del motor]
      MC[Métricas lineales]
      FP[Pronóstico y tabla lineal]
      DG[Diagnósticos residuales]
      PP[Sección paralelo Spark]
    end
  end

  FM --> ST
  FR --> ST
  PS --> ST
  ST --> SV
  ST --> MI
  ST --> WN
  ST --> FP
  ST --> PP
  ST --> DG
  SV --> MC
  SV --> DG

  style TSLIB fill:#1a1a2e,color:#eee
  style SHINY fill:#16213e,color:#eee
```

**Lectura rápida:** caja **TSLib** = qué produce la librería; caja **App Shiny** = persistencia en `app_state`, servicio que interroga TSLib, y pantalla del asistente. Los avisos `runtime_warnings` se **ensamblan** en el servidor Shiny (captura de `warnings` + columnas Spark), aunque el texto suele originarse en código TSLib o PySpark.

---

## Contenido mostrado

- **Información del modelo**: tipo y orden efectivo del ajuste lineal.
- **Avisos del motor (última ejecución)**: mensajes capturados de `warnings` en el driver (ajuste lineal, pronóstico) y avisos devueltos desde los ejecutores Spark durante el ajuste paralelo (p. ej. no estacionariedad en AR/MA). No sustituyen la validación de datos del paso 1.
- **Métricas y pronóstico lineal**: AIC/BIC donde aplique, gráfico y tabla de pronóstico, ACF de residuos.
- **Diagnósticos ampliados** (`services/evaluation_plots.py`): histograma + **Q-Q** de residuos (normalidad aproximada) y serie de **residuos estandarizados** con bandas ±2σ para outliers aparentes.
- **Modelo paralelo (Spark)**: misma estructura comparativa cuando la ruta Spark completó; si falló, mensaje derivado del `execution_log`.

### Guía breve de gráficas

| Gráfica | Uso |
|--------|-----|
| Residuos vs tiempo | Patrones no aleatorios → estructura no capturada. |
| ACF de residuos | Autocorrelación remanente → posible subajuste. |
| Histograma + Q-Q | Desvíos de normalidad → IC aproximados menos fiables. |
| Residuos estandarizados | Valores fuera de ±2 → candidatos a outlier o colas pesadas. |
| Pronóstico + IC | Comparación visual trayectoria vs incertidumbre declarada. |

Funciones adicionales en `evaluation_plots.py` (error por horizonte, fan chart, barras comparativas, heatmap AICc) están listas para integrarse cuando haya datos de holdout o matrices de selección de orden.

---

## Relación con pasos anteriores

| Paso | Título              | Aporte a Resultados                          |
|------|---------------------|-----------------------------------------------|
| 1    | Carga de datos      | Columnas, validación y `runtime_warnings` de validación/exploración |
| 2    | Exploración         | Gráficos ACF/PACF (paso 1 alimenta exploración al validar) |
| 3    | Modelo y ejecución  | Genera `fitted_model`, pronósticos, paralelo, `runtime_warnings`, `execution_log` |
| 4    | Resultados          | Lectura consolidada                           |

---

## Áreas de mejora

- Exportar tabla de pronósticos y avisos a informe descargable.
- Separar visualmente avisos “informativos” vs “riesgo de interpretación del modelo”.
- Backtesting holdout para métricas frente a datos no usados en el ajuste.
