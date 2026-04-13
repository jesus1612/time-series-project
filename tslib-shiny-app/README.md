# TSLib Shiny App – Interfaz de Análisis de Series de Tiempo

Aplicación web interactiva (Shiny for Python) para el análisis de series de tiempo. En el **monorepo** [`time-series-project`](https://github.com/jesus1612/time-series-project), esta carpeta es la **interfaz (UI)**; el núcleo algorítmico es **time-series-library** (TSLib), carpeta hermana en el mismo repositorio. La integración es unidireccional: esta app consume TSLib; TSLib no depende de esta app.

---

## Rol dentro del monorepo

| Componente | Contenido |
|--------------|-----------|
| **time-series-library** | Modelos AR/MA/ARMA/ARIMA, validación, ACF/PACF, opcionalmente flujo ARIMA paralelo con PySpark |
| **tslib-shiny-app** (esta carpeta) | Asistente por pasos (Wizard), carga de datos, visualización, selección de modelo, ejecución, Benchmark y resultados |

La app llama a TSLib para validar los datos, ajustar los modelos, y obtener tanto pronósticos como métricas. Toda la interacción algorítmica con TSLib se realiza de forma limpia mediante `services/tslib_service.py`.

---

## Características Principales

1. **Flujo Cautivador en 4 Pasos (Asistente)**: Carga de datos → Exploración → Modelo y ejecución → Resultados.
2. **Modelos Secuenciales**: AR, MA, ARMA y ARIMA con auto-selección de orden directamente por TSLib.
3. **Sección de Benchmark**: Evalúa y contrasta de manera sencilla y en una interfaz dedicada los modelos **secuenciales** versus **paralelos**.
4. **Validación**: Integración al TSLib `DataValidator` para reportar NaN y observaciones anómalas (outliers).
5. **Visualización y Exploración**: Serie temporal, ACF/PACF, estadísticas básicas, pronósticos y bandas de confianza.
6. **Diseño y UI**: Tema oscuro, textos íntegramente en español, y un proyecto altamente modular.

---

## Estructura del Proyecto

La estructura del código fomenta el aislamiento y la escalabilidad (Todo el código se mantiene en inglés siguiendo las mejores prácticas, mientras que la UI está en español):

```
tslib-shiny-app/
├── app.py                    # Entry point de la aplicación de Shiny; orquestador de pasos y navbar
├── components/               # UI components genéricos reutilizables
│   ├── layout.py             # Cards, tablas, formularios, métricas
│   └── stepper.py            # Componente de pasos del Asistente
├── features/                 # Módulos del negocio
│   ├── benchmark/            # [NUEVO] Sección y UI de Benchmark
│   │   ├── server.py         # Lógica servidora del benchmark (BenchmarkRunner)
│   │   └── ui.py             # UI de benchmark
│   ├── upload/ui.py          # Carga de archivos CSV/Excel, mapeo de columnas
│   ├── visualization/ui.py   # Gráfico de las series y calculos (ACF/PACF)
│   ├── model_selection/ui.py # Tipo de modelo, flags de parámetros y ejecución
│   └── results/ui.py         # Métricas, pronóstico, diagnósticos integrales
├── services/
│   └── tslib_service.py      # Capa de integración principal con TSLib
├── data/examples/            # Archivos dummy para pruebas locales
├── .gitignore                # Reglas de git y desactivador de archivos `.pyc`
├── requirements.txt
├── Makefile
└── README.md                 # Este archivo
```

---

## TSLib — Guía de Benchmarking y Paralelismo

> **Referencia práctica para medir rendimiento y elegir la configuración óptima de `n_jobs`.**

Se ha integrado a la interfaz una **sección nativa e independiente** para probar el Benchmark. Accede a ella desde la navegación superior (**"🚀 Benchmark"**).

### ¿Qué mide el benchmark?
La suite compara el tiempo de ajuste (fit) en modo **secuencial** (`n_jobs=1`) vs **paralelo** (`n_jobs=-1`, todos los núcleos) para los cuatro modelos de TSLib, barriendo distintos tamaños de serie (`n_obs`).

La métrica clave generada en la visualización es el **speedup**:
`speedup = tiempo_secuencial / tiempo_paralelo`

| Speedup | Significado |
|---------|-------------|
| **> 1.10** | Paralelo gana ≥ 10% → **recomendado** |
| **~ 1.00** | Neutro — overhead ≈ ganancia |
| **< 1.00** | Paralelo pierde (overhead domina ← sumamente normal en series cortas) |

### Resultados de Referencia en Interfaz (macOS, 8 núcleos M-series)

> Tiempos presentados en segundos. Configurado en el entorno al tomar el mejor de 3 repeticiones.

#### Speedup (`n_jobs=-1` vs `n_jobs=1`)

| Modelo         | n=100 | n=500 | n=1 000 | n=5 000 | n=10 000 |
|----------------|-------|-------|---------|---------|----------|
| **AR(2)**          | 1.01  | 1.00  | 0.64    | 0.57    | 0.66     |
| **MA(2)**          | 0.99  | 1.06  | 0.60    | 0.53    | 0.50     |
| **ARMA(1,1)**      | 1.01  | 1.07  | 0.52    | 0.62    | 0.75     |
| **ARIMA(1,1,1)**| 1.01 | 1.07  | **1.29 ✓**| 0.67  | 0.66     |

> **Nota**: El speedup cae por debajo de 1 para series grandes porque el cuello de botella en AR/MA/ARMA puro es el `MLEOptimizer` (single-threaded dentro de scipy). El beneficio real de `n_jobs=-1` aparece cuando se ajustan **muchas series** con `GenericParallelProcessor`.

### Umbrales de uso recomendados
| Modelo | Usa modelo paralelo (`n_jobs=-1`) cuando… |
|---------|-------------|
| **ARIMA** | Muestreos grandes `n_obs >= 1000` |
| **AR / MA / ARMA** | Muestreos inmensos `n_obs >= 5000` **ó** ajustando secuencias de >= 50 series |
| **Todas + Spark** | Siempre - Spark logra la mejor paralelización distribuyendo las particiones de la serie |

**Visualización**: Las gráficas generadas tendrán líneas correspondientes al tiempo secular versus paralelo. Además, mostrará el **"punto de codo"**, alertando visualmente dónde inicia la ganancia práctica.

---

## Requisitos y Configuración Inicial

- **Python 3.9+** (alineado con TSLib).
- **Java 17+** para PySpark 4.x (ruta paralela ARIMA y benchmarks); configurar `JAVA_HOME` / `PATH` como en el `Makefile` del proyecto.
- TSLib instalada en modo editable (el `make install` de esta app lo resuelve vía `requirements.txt`).

1. Clonar el monorepo y entrar a la UI:
```bash
git clone git@github.com:jesus1612/time-series-project.git
cd time-series-project/tslib-shiny-app
make install
```

2. TSLib queda enlazada con `-e ../time-series-library` desde `requirements.txt`. Si instalas a mano:
```bash
pip install -e ../time-series-library
```

> **Aviso de Generación .pyc**: Esta app desactiva por defecto la generación de bytecodes compilados (`.pyc`) para mantener el árbol de desarrollo y las búsquedas más limpias. Está configurado mediante reglas en `.gitignore` o seteando globalmente `export PYTHONDONTWRITEBYTECODE=1` previo a correr scripts de desarrollo.

### Para ejecutar el Servidor
```bash
make run
```
Luego visita en el navegador `http://localhost:8000`

---

## Tratamiento Computacional Avanzado

1. **Paralelismo**: La UI detecta silenciosamente el uso de Spark. De no haber PySpark nativo para Spark, la aplicación retrocede de manera transparente a una secuencia lineal (dummy-safe) para evitar bloqueos del sistema. Las implementaciones subyacentes son administradas por el sistema backend (`services/tslib_service.py`).
2. **Imputación Estricta**: TSLib requiere series matemáticamente impecables (sin `NaN`). Tras las validaciones iniciales de los datos del Wizard, la información faltante se somete a interpolación direccional (Forward-fill, ó reemplazos Cero) justo antes del entrenamiento y cálculo ACF para posibilitar el graficado. Todo el preprocesamiento fuerte lo gestiona el motor TSLib.
