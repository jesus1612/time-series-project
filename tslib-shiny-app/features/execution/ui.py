# Execution feature UI components
from shiny import ui
from components.layout import create_card

def render_execution_ui() -> ui.Tag:
    """Render execution step UI components"""
    
    return create_card(
        title="🚀 Ejecución del Análisis",
        subtitle="Inicia el procesamiento de modelos",
        content=ui.div(
            # Summary of configuration
            ui.div(
                ui.tags.h5("Configuración:"),
                ui.output_ui("execution_summary"),
                class_="mb-4"
            ),
            # Start button
            ui.div(
                ui.input_action_button("start_execution", "▶️ Iniciar Análisis", class_="btn btn-primary btn-lg"),
                class_="mb-4 text-center"
            ),
            # Status and progress
            ui.output_ui("execution_status_ui"),
        )
    )