# Benchmark feature UI components
from shiny import ui
from components.layout import create_card, create_form_group

def render_benchmark_ui() -> ui.Tag:
    """Render benchmark UI components"""
    
    return ui.div(
        ui.tags.h3("🚀 Benchmark de Rendimiento", class_="mb-4", style="color: var(--accent-primary);"),
        ui.tags.p(
            "Compara el tiempo de ajuste secuencial vs paralelo para órdenes ARIMA distintos y tamaños de serie.",
            class_="text-muted mb-4"
        ),
        
        ui.div(
            # Config panel
            ui.div(
                create_card(
                    title="Configuración",
                    subtitle="Parámetros del benchmark",
                    content=ui.div(
                        create_form_group(
                            label="Tamaños de serie (n_obs)",
                            control=ui.input_text(
                                "bench_n_obs",
                                "",
                                value="100, 500, 1000, 2000, 5000"
                            ),
                            help_text="Separados por comas (ej. 100, 500, 1000, 2000, 5000)"
                        ),
                        create_form_group(
                            label="Repeticiones por tamaño",
                            control=ui.input_numeric("bench_repeats", "", value=3, min=1, max=10),
                            help_text="Número de veces a ejecutar cada prueba para evitar ruido del CPU (se guarda el mejor tiempo)"
                        ),
                        ui.div(
                            ui.output_ui("bench_run_button_ui"),
                            class_="text-center"
                        ),
                        ui.output_ui("bench_execution_status")
                    )
                ),
                class_="col-md-4 mb-4"
            ),
            
            # Results panel
            ui.div(
                create_card(
                    title="Resultados",
                    subtitle="Comparativa de tiempos de ejecución",
                    content=ui.div(
                        ui.output_ui("bench_results_ui")
                    )
                ),
                class_="col-md-8"
            ),
            class_="row"
        ),
        ui.tags.hr(class_="my-4"),
        ui.tags.h4("Benchmark ARIMA + referencia statsmodels (local)", class_="mb-2", style="color: var(--accent-primary);"),
        ui.tags.p(
            "Compara TSLib lineal, ParallelARIMAWorkflow y Spark+statsmodels con statsmodels en proceso "
            "(solo evaluación frente a referencia aceptada; el modelo de valor operativo sigue siendo ARIMA TSLib). "
            "Rendimiento: serie sintética. Precisión: CSV en TT/sampler/datasets/.",
            class_="text-muted mb-3",
        ),
        ui.div(
            create_card(
                title="Benchmark ARIMA",
                subtitle="Rendimiento y error holdout vs referencia local",
                content=ui.div(
                    create_form_group(
                        label="Grid n_obs (rendimiento)",
                        control=ui.input_text(
                            "bench_arima_n_obs",
                            "",
                            value="100, 500, 1000, 2000, 5000",
                        ),
                        help_text="Tamaños de serie sintética para medir tiempos de ajuste",
                    ),
                    create_form_group(
                        label="CSV del sampler (precisión)",
                        control=ui.input_text(
                            "bench_arima_csv",
                            "",
                            value="synthetic_arima_211.csv",
                        ),
                        help_text="Archivo en TT/sampler/datasets/",
                    ),
                    create_form_group(
                        label="Repeticiones (rendimiento)",
                        control=ui.input_numeric(
                            "bench_arima_repeats", "", value=2, min=1, max=5
                        ),
                    ),
                    ui.div(
                        ui.input_action_button(
                            "run_arima_triple_benchmark",
                            "▶️ Ejecutar benchmark ARIMA (TSLib vs referencias)",
                            class_="btn btn-secondary btn-lg w-100 mt-2",
                        ),
                        class_="text-center",
                    ),
                    ui.output_ui("arima_bench_status_ui"),
                    ui.output_ui("arima_bench_summary_ui"),
                    ui.tags.h6("Rendimiento", class_="mt-3"),
                    ui.output_plot("arima_bench_perf_plot", height="420px"),
                    ui.tags.h6("Precisión (holdout)", class_="mt-3"),
                    ui.output_plot("arima_bench_acc_plot", height="380px"),
                ),
            ),
            class_="mb-4",
        ),
    )
