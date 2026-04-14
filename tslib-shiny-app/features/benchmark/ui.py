# Benchmark tab — dashboard layout, compact controls, results only after run
from shiny import ui
from components.layout import create_card, create_form_group


def render_benchmark_ui() -> ui.Tag:
    return ui.div(
        {"class": "benchmark-dashboard"},
        ui.div(
            {"class": "bench-hero"},
            ui.tags.h2("Benchmark ARIMA", class_="bench-hero-title"),
            ui.tags.p(
                "Comparación TSLib frente a statsmodels: tiempos, errores de validación y diagnósticos "
                "sobre un CSV del sampler.",
                class_="bench-hero-lead",
            ),
            ui.div(
                {"class": "bench-palette-key"},
                ui.tags.span("TSLib", class_="bench-palette-swatch bench-palette-swatch--1"),
                ui.tags.span("ParallelARIMAWorkflow", class_="bench-palette-swatch bench-palette-swatch--2"),
                ui.tags.span("Spark · statsmodels", class_="bench-palette-swatch bench-palette-swatch--3"),
                ui.tags.span("statsmodels ref.", class_="bench-palette-swatch bench-palette-swatch--4"),
                ui.tags.span("Neutro / referencia", class_="bench-palette-swatch bench-palette-swatch--5"),
            ),
        ),
        ui.div(
            {"class": "row g-3"},
            ui.div(
                {"class": "col-lg-6"},
                create_card(
                    title="Rendimiento",
                    subtitle="Serie sintética por tamaño",
                    content=ui.div(
                        create_form_group(
                            "Grid n_obs",
                            ui.input_text(
                                "fb_n_obs",
                                "",
                                value="100, 500, 1000, 2000, 5000",
                            ),
                            "Separados por coma. Valores altos pueden tardar mucho con Spark.",
                        ),
                        create_form_group(
                            "Repeticiones / punto",
                            ui.input_numeric("fb_repeats", "", value=2, min=1, max=5),
                            "Se toma el mínimo tiempo por punto.",
                        ),
                    ),
                ),
            ),
            ui.div(
                {"class": "col-lg-6"},
                create_card(
                    title="Datos y n_jobs",
                    subtitle="CSV y prueba secuencial vs paralelo interno",
                    content=ui.div(
                        create_form_group(
                            "CSV",
                            ui.input_text(
                                "fb_csv",
                                "",
                                value="arima_eval_benchmark.csv",
                            ),
                            "Archivo en TT/sampler/datasets/.",
                        ),
                        create_form_group(
                            "Grid n_jobs",
                            ui.input_text(
                                "fb_njobs_grid",
                                "",
                                value="100, 500, 1000, 2000",
                            ),
                            "Tamaños para comparar n_jobs=1 vs -1 en ARIMAModel.",
                        ),
                        create_form_group(
                            "Repeticiones n_jobs",
                            ui.input_numeric("fb_njobs_repeats", "", value=3, min=1, max=10),
                            None,
                        ),
                        create_form_group(
                            "Rejilla ParallelARIMAWorkflow",
                            ui.input_select(
                                "fb_grid_mode",
                                "",
                                {
                                    "auto_n": "Automático (n)",
                                    "acf_pacf": "ACF / PACF",
                                    "manual": "Manual (max p, q)",
                                },
                                selected="auto_n",
                            ),
                            "Cómo se eligen los techos de p y q antes del grid (documentación en docs/ARIMA_METODOLOGIA_ROADMAP.md).",
                        ),
                        create_form_group(
                            "max_p y max_q (solo modo manual)",
                            ui.div(
                                ui.input_numeric("fb_manual_max_p", "max_p", value=3, min=0, max=15),
                                ui.input_numeric("fb_manual_max_q", "max_q", value=3, min=0, max=15),
                                class_="d-flex gap-2 flex-wrap",
                            ),
                            "Se usan solo si la rejilla está en manual.",
                        ),
                    ),
                ),
            ),
        ),
        ui.div(
            {"class": "d-flex justify-content-center my-3"},
            ui.input_action_button(
                "run_full_benchmark",
                "Generar benchmark",
                class_="btn btn-primary bench-run-btn px-4",
            ),
        ),
        ui.output_ui("fb_status_ui"),
        ui.output_ui("fb_results_panel"),
        ui.output_ui("fb_summary_ui"),
    )
