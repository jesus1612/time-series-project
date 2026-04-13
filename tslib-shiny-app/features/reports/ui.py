# Reports feature UI components
from shiny import ui
from components.layout import create_card

def render_reports_ui() -> ui.Tag:
    """Render reports step UI components"""
    
    return create_card(
        title="📄 Generación de Reportes",
        subtitle="Crea y descarga reportes del análisis",
        content=ui.div(
            ui.div(
                ui.div("📄", class_="file-upload-icon"),
                ui.div("Los reportes se generarán automáticamente", class_="file-upload-text"),
                class_="file-upload-area"
            ),
            ui.div(
                ui.div(
                    ui.tags.h4("Configuración del reporte:"),
                    ui.div(
                        ui.input_checkbox("include_summary", "Resumen ejecutivo", value=True),
                        ui.input_checkbox("include_visualizations", "Visualizaciones", value=True),
                        ui.input_checkbox("include_metrics", "Métricas de evaluación", value=True),
                        class_="mt-2"
                    ),
                    class_="mb-3"
                ),
                ui.div(
                    ui.input_action_button("generate_report", "📄 Generar Reporte", class_="btn btn-primary btn-lg"),
                    ui.input_action_button("download_pdf", "📥 Descargar PDF", class_="btn btn-secondary"),
                    class_="d-flex gap-2 mb-3"
                ),
                ui.div(
                    ui.div("Estado: Listo para generar", id="report_status", class_="status-indicator status-info"),
                    class_="mt-2"
                )
            )
        )
    )