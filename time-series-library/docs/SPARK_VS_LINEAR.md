# Por qué Spark puede ser más lento que un ajuste lineal (un solo nodo)

Este documento resume motivos habituales cuando, incluso con clusters, el tiempo de **PySpark** para un pipeline como `ParallelARIMAWorkflow` supera el de un **MLE en un solo proceso** (statsmodels / TSLib en el driver).

## 1. Coste fijo (overhead) de Spark

Cada job paga:

- Arranque del **driver JVM**, comunicación driver ↔ ejecutores, **serialización** de tareas y de datos.
- Un `count()`, `repartition()` o `mapInPandas` dispara **scheduling**, envío de bytecode a los workers y a veces **shuffle**.

Para series **cortas o medianas**, el trabajo útil por tarea (ajustar un ARIMA en un subconjunto) puede ser **del mismo orden de magnitud** que ese coste fijo. El “paralelo” solo compensa cuando hay **muchas tareas independientes** o datos muy grandes.

## 2. Granularidad del paralelismo en este workflow

En el paso distribuido se construye un **DataFrame de tareas** (ventana × combinación `(p,d,q)`), se **reparticiona** y se ejecuta `mapInPandas`. Si el número de filas es moderado, cada partición hace poco trabajo; el **ratio overhead / cómputo** empeora.

## 3. Particiones: `repartition` vs “Koalas” (pandas API on Spark)

- **`repartition(n)`** (PySpark SQL): redistribuye filas del DataFrame/RDD en **n particiones** lógicas de Spark. Afecta **paralelismo** y **shuffle**; no es un concepto de pandas.
- **Koalas / pandas API on Spark** (histórico): era una capa que imitaba pandas sobre Spark con **otro** modelo de particiones y optimizaciones distintas al DataFrame SQL clásico. Hoy suele aludirse al **pandas UDF / Arrow** o a **PySpark pandas** según versión.

No son “niveles” comparables uno a uno: **repartition** opera sobre el plan de Spark; una API estilo pandas sobre Spark añade otra capa de traducción y a veces más **round-trips** al driver.

## 4. Por qué el tiempo “de distribuir datos” suma al total

Medimos el tramo **crear DataFrame de tareas + `cache` + `count()`** tras `repartition`: fuerza materialización y distribución de particiones. Ese tiempo **no** es el del `mapInPandas` posterior; ayuda a ver si el cuello es **I/O y shuffle** frente al **fit** en workers.

## 5. Cuándo suele ganar el lineal

- Una sola serie, **un** modelo final en el driver (statsmodels / TSLib).
- Sin shuffle, sin UDF masiva, sin serializar miles de filas de tareas.

## 6. Cuándo compensa Spark

- Muchas series o muchas tareas independientes (muchos `(ventana, p, d, q)`), de modo que los workers estén **saturados** y el coste fijo se **amortice**.

En resumen: **Spark no acelera por definición**; acelera cuando el **paralelismo útil** supera el **overhead** del sistema distribuido. Por eso, en benchmarks con *N* pequeño, el lineal a menudo gana.
