"""Product limits for the Shiny app (ingestion)."""

# Maximum upload size before pandas read; aligns with diagrams/01-ingesta spec.
MAX_UPLOAD_FILE_BYTES = 500 * 1024 * 1024
