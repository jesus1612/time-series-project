# Proyecto de series de tiempo (monorepo)

Repositorio unificado con la librería **TSLib**, la interfaz **Shiny for Python**, material de **diagramas** del flujo del proyecto, y el **sampler** de datasets CSV para benchmarks.

## Estructura

| Directorio | Descripción |
|------------|-------------|
| [`time-series-library/`](time-series-library/) | TSLib: modelos AR / MA / ARMA / ARIMA, validación, métricas, rutas opcionales con **PySpark**. |
| [`tslib-shiny-app/`](tslib-shiny-app/) | Aplicación web (asistente por pasos, benchmark, resultados). Depende de TSLib instalada en editable. |
| [`diagrams/`](diagrams/) | Documentación del proceso (ingesta, exploración, modelo/ejecución, resultados). Ver `DOCUMENTACION.md` y `PROCESO.txt` en cada carpeta. |
| [`sampler/`](sampler/) | Scripts y CSV de ejemplo para pruebas de rendimiento y calidad ARIMA. |

La UI importa TSLib vía `pip install -e ../time-series-library` (definido en [`tslib-shiny-app/requirements.txt`](tslib-shiny-app/requirements.txt)).

**Documentación de teoría, ARIMA, benchmarks (statsmodels) y paralelismo:** [`docs/series-tiempo-arima-y-paralelismo.md`](docs/series-tiempo-arima-y-paralelismo.md).

## Requisitos

- **Python 3.9+** (comprobado por el Makefile de TSLib).
- **Java 17+** en `PATH` con `JAVA_HOME` coherente para **PySpark 4.x** (flujo lineal vs Spark en ARIMA paralelo y tests con Spark).
- macOS/Linux con `make` y `python3`.

Instalación de JDK (referencia):

- macOS (Homebrew): `make -C time-series-library install-java-macos` o `make -C tslib-shiny-app install-java`.
- Linux: paquete `openjdk-17-jdk` o [Eclipse Temurin 17](https://adoptium.net/temurin/releases/?version=17).

## Inicio rápido

Desde la raíz del clon:

```bash
# Todo en uno: venv de TSLib + venv de la UI (editable) y arrancar Shiny
make up
```

Equivalente a `make install-all` seguido de `make run-ui`. `install-all` ejecuta `install-lib` y `install-ui` (hay **dos entornos virtuales**: `time-series-library/venv` y `tslib-shiny-app/venv`).

Flujo manual:

```bash
make install-lib   # TSLib dev en time-series-library/venv
make test-lib      # opcional; requiere JDK para tests Spark
make install-ui    # UI + TSLib editable en tslib-shiny-app/venv
make run-ui
```

Abre **http://localhost:8000** (o la IP que muestre la consola).

Otros objetivos útiles: `make help`, `make sampler` (regenera CSVs en `sampler/datasets/`), `make check-java`.

## Detalle por componente

### TSLib (`time-series-library/`)

```bash
cd time-series-library
make help          # install, install-spark, install-dev, test, benchmark, ...
```

Incluye instalación con PySpark, suite de tests, benchmarks y utilidades Java documentadas en el Makefile del subproyecto.

### Shiny (`tslib-shiny-app/`)

```bash
cd tslib-shiny-app
make install
make run
```

Documentación ampliada (benchmark, paralelismo, UI): [`tslib-shiny-app/README.md`](tslib-shiny-app/README.md).

### Sampler

```bash
make sampler
# o: python3 sampler/generate_datasets.py
```

Si existe `time-series-library/venv` (p. ej. tras `make install-lib`), el objetivo `make sampler` usa ese intérprete; si no, usa `python3` del sistema (debe tener `numpy` y `pandas`). `statsmodels` es opcional para algunos datasets clásicos. Detalles: [`sampler/README.md`](sampler/README.md).

### Diagramas

Carpetas numeradas `01-ingesta` … `04-resultados`: flujos y notas para el trabajo de titulación; no son ejecutables.

## Conservar historial Git de repos antiguos

Si antes existían dos repositorios (UI y librería) y quieres **un solo historial** con commits previos, puedes usar herramientas como `git subtree` o `git-filter-repo` para importar ramas con prefijos de subcarpeta. Un único commit inicial en este monorepo también es válido si no necesitas ese historial.

## Licencia y créditos

Proyecto académico; ver archivos y dependencias de cada subcarpeta para detalles.
