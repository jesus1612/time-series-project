# Visualization feature UI components
from shiny import ui
from components.layout import create_card, create_metric_card

def render_visualization_ui() -> ui.Tag:
    """Render visualization step UI components"""
    
    # Single focused visualization section
    return create_card(
        title="📊 Visualización de Serie Temporal",
        subtitle="Gráfico interactivo y estadísticas básicas",
        content=ui.div(
            # Time series plot
            ui.div(
                ui.output_plot("time_series_plot", height="400px"),
                class_="mb-4"
            ),
            # Statistics cards
            ui.div(
                ui.tags.h4("Estadísticas Básicas:"),
                ui.output_ui("statistics_cards"),
                class_="mt-3"
            ),
            # ACF/PACF plots (optional)
            ui.div(
                ui.tags.h4("Análisis de Correlación:"),
                ui.div(
                    ui.div(
                        ui.output_plot("acf_plot", height="300px"),
                        class_="col-md-6"
                    ),
                    ui.div(
                        ui.output_plot("pacf_plot", height="300px"),
                        class_="col-md-6"
                    ),
                    class_="row"
                ),
                ui.output_ui("acf_pacf_debug"),
                class_="mt-4"
            ),
            ui.div(
                ui.tags.h4("Notas de exploración:"),
                ui.output_ui("exploration_notes_ui"),
                class_="mt-4",
            ),
        )
    )