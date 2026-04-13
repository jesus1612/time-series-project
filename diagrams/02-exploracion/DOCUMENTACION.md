# Documentación: Exploración de datos

Este paso valida la serie elegida, calcula estadísticas básicas y genera señales exploratorias (ACF/PACF, tendencia, estacionalidad) antes del modelado.

Importar en [diagrams.net](https://app.diagrams.net/): **Insertar → Avanzado → Mermaid**.

---

## Diagrama 1 — Flujo del Paso 2

```mermaid
flowchart TB
  subgraph UI["Paso 2 — Exploración"]
    A[Usuario pulsa Validar Datos]
    B[Mostrar avisos, periodo estacional y estado de validación]
    C[Mostrar serie temporal]
    D[Mostrar estadísticas básicas]
    E[Mostrar ACF y PACF]
    F[Mostrar notas de exploración]
  end

  subgraph APP["App Shiny"]
    G[handle_validate_data]
    H[TSLibService.validate_data]
    I[TSLibService.get_exploratory_analysis]
    J[Actualizar app_state: validation_report y exploratory_analysis]
  end

  subgraph TSLIB["TSLib"]
    K[DataValidator: longitud, NaN, inf, outliers, tendencia, estacionalidad]
    L[ACFCalculator / PACFCalculator]
  end

  A --> G
  G --> H
  H --> K
  G --> I
  I --> L
  H --> J
  I --> J
  J --> B
  J --> C
  J --> D
  J --> E
  J --> F

  style TSLIB fill:#1a1a2e,color:#eee
  style APP fill:#16213e,color:#eee
```

---

## Diagrama 2 — Decisiones en validación y exploración

```mermaid
flowchart TD
  INICIO([Serie numérica seleccionada]) --> V1{¿Pasa validación base?}

  V1 -->|No| ERR["Mensajes de fallo en validación"]
  V1 -->|Sí| EXP["Continuar exploración"]

  EXP --> ACFPACF["ACF/PACF sobre serie completa (sin NaN)"]
  ACFPACF --> OUT["Guardar resultados + notas de exploración"]
  ERR --> OUT
```

---

## Qué se muestra en pantalla

- **Estado de validación**: mensajes y advertencias en español, incluyendo periodos estacionales candidatos cuando se detectan.
- **Serie temporal**: gráfica principal para inspección visual.
- **Estadísticas**: media, desviación, mínimo y máximo.
- **ACF/PACF**: evidencia de dependencia temporal por rezagos.
- **Notas de exploración**: señales detectadas (tendencia/estacionalidad); no hay imputación en la app.

---

## Áreas de mejora identificadas

- Añadir selector avanzado para **max_lag** en ACF/PACF.
- Separar en UI los avisos de tipo: dato vs modelo para mejorar trazabilidad.
- Complementar la tabla de calidad con umbrales de interpretación por métrica.
