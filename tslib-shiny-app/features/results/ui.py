# Results feature UI components
from shiny import ui
from components.layout import create_card, create_metric_card

def render_results_ui() -> ui.Tag:
    """Render results step UI components"""
    
    return create_card(
        title="📈 Resultados del análisis",
        subtitle="Métricas y predicciones del modelo",
        content=ui.div(
            # Model info
            ui.div(
                ui.tags.h5("Información del modelo:"),
                ui.output_ui("model_info_ui"),
                class_="mb-4"
            ),
            ui.div(
                ui.tags.h5("Avisos del motor (última ejecución):"),
                ui.output_ui("runtime_warnings_ui"),
                class_="mb-4"
            ),
            # Metrics
            ui.div(
                ui.tags.h5("Métricas de evaluación:"),
                ui.output_ui("metrics_cards"),
                class_="mb-4"
            ),
            # Linear model results section (only show title for ARIMA)
            ui.div(
                ui.output_ui("linear_model_title"),
                # Forecast plot
                ui.div(
                    ui.tags.h5("Pronóstico:"),
                    ui.output_plot("forecast_plot", height="400px"),
                    class_="mb-4"
                ),
                # Forecast table
                ui.div(
                    ui.tags.h5("Valores del pronóstico:"),
                    ui.output_ui("forecast_table_ui"),
                    class_="mb-4"
                ),
                # Diagnostics
                ui.div(
                    ui.tags.h5("Diagnósticos del modelo:"),
                    ui.div(
                        ui.div(
                            ui.output_plot("residuals_plot", height="300px"),
                            class_="col-md-6"
                        ),
                        ui.div(
                            ui.output_plot("residuals_acf_plot", height="300px"),
                            class_="col-md-6"
                        ),
                        class_="row"
                    ),
                    ui.div(
                        ui.tags.h6("Histograma y Q-Q de residuos", class_="mt-3"),
                        ui.output_plot("eval_residual_hist_qq_plot", height="360px"),
                        class_="mb-3",
                    ),
                    ui.div(
                        ui.tags.h6("Residuos estandarizados", class_="mt-2"),
                        ui.output_plot("eval_standardized_residuals_plot", height="280px"),
                        class_="mb-3",
                    ),
                    class_="mb-4"
                ),
                class_="mb-5"
            ),
            ui.div(
                ui.tags.h4("Modelo paralelo (Spark)", class_="mb-3"),
                ui.output_ui("parallel_model_section"),
                class_="mb-4"
            )
        )
    )