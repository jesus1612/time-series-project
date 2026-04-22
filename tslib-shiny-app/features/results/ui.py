# Results feature UI components
from shiny import ui
from shinywidgets import output_widget
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
            ui.div(
                ui.tags.h5("Tiempos de ejecución:"),
                ui.tags.p(
                    "Barras interactivas (zoom). ARIMA, AR y MA: lineal, paralelo total, warm-up Spark y distribución del DataFrame de tareas; otros modelos: lineal vs paralelo.",
                    class_="text-muted small mb-1",
                ),
                output_widget("execution_timing_plot", height="380px"),
                class_="mb-4"
            ),
            # Linear model results section (only show title for ARIMA)
            ui.div(
                ui.output_ui("linear_model_title"),
                # Forecast plot
                ui.div(
                    ui.tags.h5("Pronóstico:"),
                    ui.tags.p(
                        "Gráfico interactivo: zoom y desplazamiento con la barra de herramientas o gestos.",
                        class_="text-muted small mb-1",
                    ),
                    output_widget("forecast_plot", height="420px"),
                    class_="mb-4"
                ),
                # Forecast table
                ui.div(
                    ui.tags.h5("Valores del pronóstico:"),
                    ui.output_ui("forecast_table_ui"),
                    class_="mb-4"
                ),
                ui.div(
                    ui.tags.h5("|Diferencia lineal − paralelo| por horizonte:"),
                    output_widget("forecast_diff_horizon_plot", height="320px"),
                    class_="mb-4",
                ),
                # Diagnostics
                ui.div(
                    ui.tags.h5("Diagnósticos del modelo (interactivos: zoom/pan):"),
                    ui.div(
                        ui.div(
                            output_widget("residuals_plot", height="320px"),
                            class_="col-md-6"
                        ),
                        ui.div(
                            output_widget("residuals_acf_plot", height="320px"),
                            class_="col-md-6"
                        ),
                        class_="row"
                    ),
                    ui.div(
                        ui.tags.h6("Histograma y Q-Q de residuos", class_="mt-3"),
                        output_widget("eval_residual_hist_qq_plot", height="380px"),
                        class_="mb-3",
                    ),
                    ui.div(
                        ui.tags.h6("Residuos estandarizados", class_="mt-2"),
                        output_widget("eval_standardized_residuals_plot", height="300px"),
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