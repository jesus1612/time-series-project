# Análisis de cruce: ARIMA lineal vs paralelo

Este documento se rellena tras ejecutar el benchmark en la app (pestaña Benchmark → ARIMA extendido) o scripts locales.

## Objetivo

Estimar el menor \(N^\*\) donde el speedup medio (tiempo lineal / tiempo paralelo) supera 1.0 de forma estable para el entorno concreto (CPU, cores, Spark local vs cluster).

## Tabla de resultados (plantilla)

| N | t_lineal (s) | t_workflow (s) | t_spark_sm (s) | speedup L/W | speedup L/SM |
|---|--------------|----------------|----------------|-------------|--------------|
| 100 | | | | | |
| 500 | | | | | |
| … | | | | | |

## Veredicto (rellenar)

- A partir de **N ≥ \_\_\_** el workflow paralelo (TSLib) suele ser más rápido que el ajuste lineal con `n_jobs=1` en este hardware.
- Notas sobre overhead de Spark para series cortas: \_\_\_

## Regresión / tendencia (opcional)

Ajustar speedup vs \(\log N\) o vs \(N\) con mínimos cuadrados y documentar \(R^2\) y pendiente.
