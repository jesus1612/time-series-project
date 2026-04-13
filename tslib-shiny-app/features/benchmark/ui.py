# Benchmark feature UI — single "generate" flow: ARIMA TSLib vs statsmodels, times, diagnostics, n_jobs suite
from shiny import ui
from components.layout import create_card, create_form_group


def render_benchmark_ui() -> ui.Tag:
    """Single benchmark tab: one primary action builds all figures."""

    return ui.div(
        ui.tags.h3("Benchmark integral", class_="mb-2", style="color: var(--accent-primary);"),
        ui.tags.p(
            "Genera un informe completo: (1) tiempos de ajuste y speedups para "
            "TSLib lineal, ParallelARIMAWorkflow (línea verde — implementación Spark en TSLib), "
            "Spark+statsmodels y statsmodels local; (2) errores holdout RMSE/MAE/MAPE; "
            "(3) ACF, PACF, residuos y Q-Q sobre el CSV del sampler; (4) error por horizonte; "
            "(5) benchmark secuencial vs paralelo n_jobs en los cuatro modelos.",
            class_="text-muted mb-3",
        ),
        ui.div(
            create_card(
                title="Parámetros",
                subtitle="Opcional — hay valores por defecto recomendados",
                content=ui.div(
                    create_form_group(
                        label="Grid n_obs (curva de rendimiento ARIMA)",
                        control=ui.input_text(
                            "fb_n_obs",
                            "",
                            value="100, 500, 1000, 2000, 5000",
                        ),
                        help_text="Tamaños de serie sintética. Evita 10000+ si ParallelARIMAWorkflow es lento en tu máquina.",
                    ),
                    create_form_group(
                        label="CSV del sampler (precisión y gráficos)",
                        control=ui.input_text(
                            "fb_csv",
                            "",
                            value="arima_eval_benchmark.csv",
                        ),
                        help_text="Ruta bajo TT/sampler/datasets/. Recomendado: arima_eval_benchmark.csv (~960 puntos).",
                    ),
                    create_form_group(
                        label="Repeticiones (media de tiempos)",
                        control=ui.input_numeric("fb_repeats", "", value=2, min=1, max=5),
                        help_text="Se guarda el mínimo de repeticiones por punto (como antes).",
                    ),
                    create_form_group(
                        label="Grid n_obs (benchmark n_jobs AR/MA/ARMA/ARIMA)",
                        control=ui.input_text(
                            "fb_njobs_grid",
                            "",
                            value="100, 500, 1000, 2000",
                        ),
                        help_text="Segunda parte del informe: tiempo secuencial vs n_jobs=-1.",
                    ),
                    create_form_group(
                        label="Repeticiones (n_jobs)",
                        control=ui.input_numeric("fb_njobs_repeats", "", value=3, min=1, max=10),
                    ),
                    ui.div(
                        ui.input_action_button(
                            "run_full_benchmark",
                            "Generar benchmark completo",
                            class_="btn btn-primary btn-lg w-100 mt-3",
                        ),
                        class_="text-center",
                    ),
                    ui.output_ui("fb_status_ui"),
                ),
            ),
            class_="mb-4",
        ),
        ui.tags.h4("Resultados", class_="mt-2 mb-2", style="color: var(--accent-primary);"),
        ui.tags.h6("1. Tiempos de ajuste ARIMA (eje Y log)", class_="mt-2"),
        ui.output_plot("fb_perf_time_plot", height="480px"),
        ui.tags.h6("2. Speedup vs TSLib lineal", class_="mt-2"),
        ui.output_plot("fb_perf_speedup_plot", height="420px"),
        ui.tags.h6("3. Precisión holdout (barras)", class_="mt-2"),
        ui.output_plot("fb_acc_plot", height="400px"),
        ui.tags.h6("4. Diagnóstico: ACF / PACF / residuos / Q-Q (train)", class_="mt-2"),
        ui.output_plot("fb_diag_plot", height="520px"),
        ui.tags.h6("5. Error absoluto por horizonte (TSLib vs statsmodels)", class_="mt-2"),
        ui.output_plot("fb_horizon_plot", height="380px"),
        ui.tags.h6("6. Benchmark n_jobs (secuencial vs paralelo)", class_="mt-2"),
        ui.tags.p(
            "Aquí se compara ARIMAModel de TSLib con n_jobs=1 frente a n_jobs=-1 (paralelismo interno del ajuste). "
            "No es lo mismo que ParallelARIMAWorkflow de las secciones 1–3 (pipeline Spark en tslib.spark).",
            class_="text-muted small mb-2",
        ),
        ui.output_plot("fb_elbow_plot", height="900px"),
        ui.output_ui("fb_summary_ui"),
    )
