# Benchmark tab — ARIMA paralelo (Spark) vs ARIMA lineal (statsmodels)
from shiny import ui
from components.layout import create_card, create_form_group


def render_benchmark_ui() -> ui.Tag:
    return ui.div(
        {"class": "benchmark-dashboard"},
        ui.div(
            {"class": "bench-hero"},
            ui.tags.h2("Benchmark ARIMA", class_="bench-hero-title"),
            ui.tags.p(
                "Comparación de tiempos de ajuste, error en holdout y |error| por horizonte: "
                "ARIMA paralelo (Spark) frente a ARIMA lineal (statsmodels).",
                class_="bench-hero-lead",
            ),
            ui.div(
                {"class": "bench-palette-key bench-palette-key--two"},
                ui.tags.span("ARIMA paralelo", class_="bench-palette-swatch bench-palette-swatch--par"),
                ui.tags.span("ARIMA lineal", class_="bench-palette-swatch bench-palette-swatch--lin"),
            ),
        ),
        ui.div(
            {"class": "row g-3"},
            ui.div(
                {"class": "col-lg-6"},
                create_card(
                    title="Rendimiento",
                    subtitle="Serie sintética AR(1) por tamaño",
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
                    title="Datos y rejilla",
                    subtitle="CSV del sampler y modo de búsqueda (p,q)",
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
                            "Rejilla pipeline paralelo",
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
                            "Techos de p y q en el workflow Spark (docs/ARIMA_METODOLOGIA_ROADMAP.md).",
                        ),
                        create_form_group(
                            "max_p y max_q (solo modo manual)",
                            ui.div(
                                ui.input_numeric("fb_manual_max_p", "max_p", value=3, min=0, max=15),
                                ui.input_numeric("fb_manual_max_q", "max_q", value=3, min=0, max=15),
                                class_="d-flex gap-2 flex-wrap",
                            ),
                            "Solo si la rejilla está en manual.",
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
    )
