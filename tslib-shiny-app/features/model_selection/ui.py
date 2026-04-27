# Model selection feature UI components
from typing import Optional

from shiny import ui

from components.layout import create_card, create_form_group

_MODEL_CHOICES = {
    "__none__": "— Selecciona un modelo —",
    "AR": "AR - Autoregresivo",
    "MA": "MA - Media Móvil",
    "ARMA": "ARMA - Combinado",
    "ARIMA": "ARIMA - Integrado",
}


def render_model_selection_ui(
    auto_select_value: bool = True,
    selected_model_type: Optional[str] = None,
) -> ui.Tag:
    """Render model selection step UI components
    
    Args:
        auto_select_value: Initial value for the auto_select switch (default: True)
        selected_model_type: Hydrated model id from app state when (re)entering this step
    """
    sel = (
        selected_model_type
        if selected_model_type in ("AR", "MA", "ARMA", "ARIMA")
        else "__none__"
    )
    
    return create_card(
        title="⚙️ Modelo y ejecución",
        subtitle="Configura el modelo y ejecuta el análisis",
        content=ui.div(
            # Model type selector (static id: stable reactive deps for p/d/q outputs)
            ui.div(
                ui.tags.h5("Tipo de modelo:"),
                ui.input_select(
                    "model_type",
                    "",
                    choices=_MODEL_CHOICES,
                    selected=sel,
                ),
                ui.output_ui("model_description"),
                class_="mb-4"
            ),
            # Auto-selection switch
            ui.div(
                ui.input_switch("auto_select", "Selección automática de orden", value=auto_select_value),
                ui.tags.p("Activar para que el modelo seleccione automáticamente los parámetros óptimos", class_="text-muted"),
                class_="mb-4"
            ),
            # Manual parameters (shown when auto_select is False)
            ui.output_ui("manual_parameters_ui"),
            # Additional options
            ui.div(
                ui.tags.h5("Opciones adicionales:"),
                create_form_group(
                    label="Pasos a Pronosticar",
                    control=ui.input_numeric("forecast_steps", "", value=10, min=1, max=100),
                    help_text="Número de pasos futuros a predecir"
                ),
                ui.input_switch("include_confidence", "Incluir intervalos de confianza", value=True),
                class_="mt-3"
            ),
            # Execution controls integrated here
            ui.div(
                ui.tags.h5("Ejecución:"),
                ui.output_ui("execution_summary"),
                ui.output_ui("spark_parallel_status_ui"),
                ui.div(
                    ui.tags.h6("Registro de la corrida", class_="mt-3 mb-2"),
                    ui.output_ui("execution_log"),
                    class_="mb-2",
                ),
                ui.div(
                    ui.input_action_button("start_execution", "▶️ Ajustar y pronosticar", class_="btn btn-primary btn-lg"),
                    class_="my-3 text-center"
                )
            )
        )
    )