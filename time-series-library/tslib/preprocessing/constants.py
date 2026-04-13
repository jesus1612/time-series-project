"""
Constantes base para validación e imputación en series temporales.

Justificación breve:
- Umbral de faltantes: se mantiene en 10% como regla operativa (tamiz práctico)
  y no como verdad universal. La aceptabilidad real depende del mecanismo de
  faltantes (MCAR/MAR/MNAR) y del modelo de análisis (Little & Rubin, 2019).
- Imputación por defecto: interpolación lineal sobre el índice para series
  univariadas equiespaciadas, alineada con prácticas difundidas en imputeTS
  (Moritz & Bartz-Beielstein, 2017) y la función `na.interp` de forecast para
  escenarios no estacionales.
"""

DEFAULT_MIN_SERIES_LENGTH = 3
DEFAULT_MAX_MISSING_RATIO = 0.10
