# Benchmark feature server — ARIMA paralelo vs ARIMA lineal (statsmodels)
from __future__ import annotations

import contextlib
import io as _io
import traceback

import matplotlib.pyplot as plt
from shiny import reactive, render, req, ui

try:
    from .runner import BenchmarkRunner

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

try:
    from .arima_benchmark import (
        ARIMABenchmarkSuite,
        LABEL_ARIMA_LINEAL,
        LABEL_ARIMA_PARALELO,
    )

    ARIMA_BENCH_AVAILABLE = True
except ImportError:
    ARIMA_BENCH_AVAILABLE = False
    ARIMABenchmarkSuite = None  # type: ignore
    LABEL_ARIMA_PARALELO = "ARIMA paralelo"
    LABEL_ARIMA_LINEAL = "ARIMA lineal"


def register_benchmark_server(input, output, session, app_state):
    """Register unified benchmark server."""

    def _chart_block(title: str, plot_id: str, height: str):
        return ui.div(
            ui.tags.h6(title, class_="bench-section-title"),
            ui.output_plot(plot_id, height=height),
            class_="bench-chart-card",
        )

    @output
    @render.ui
    def fb_status_ui():
        st = app_state.get().get("fb_status", "idle")
        if st == "running":
            return ui.div(
                ui.tags.div(class_="spinner-border spinner-border-sm text-primary", role="status"),
                ui.tags.span(" Ejecutando…", class_="ms-2 text-secondary small"),
                class_="d-flex align-items-center justify-content-center py-2",
            )
        if st == "error":
            err = (app_state.get().get("fb_error") or "")[:1200]
            return ui.div(
                ui.tags.p("No se completó el benchmark.", class_="text-warning small mb-1"),
                ui.tags.pre(err, class_="bench-error-pre small"),
                class_="mt-2",
            )
        return ui.div()

    @output
    @render.ui
    def fb_results_panel():
        st = app_state.get().get("fb_status", "idle")
        placeholder = None
        if st == "idle":
            placeholder = ui.div(
                ui.tags.p(
                    "Los gráficos aparecerán aquí tras generar el benchmark.",
                    class_="bench-placeholder-text",
                ),
                class_="bench-placeholder card",
            )
        elif st == "running":
            placeholder = ui.div(
                ui.tags.div(class_="spinner-border text-primary mb-2", role="status"),
                ui.tags.p(
                    "Calculando tiempos y métricas (solo "
                    f"{LABEL_ARIMA_PARALELO} vs {LABEL_ARIMA_LINEAL}).",
                    class_="text-secondary small mb-0",
                ),
                class_="bench-running card text-center py-5",
            )

        show = st == "done"
        disp = "block" if show else "none"
        chart_stack = ui.div(
            {
                "class": "bench-charts-stack",
                "style": f"display: {disp};",
            },
            _chart_block(
                "Tiempos de ajuste (eje Y logarítmico; ver docstring en código)",
                "fb_perf_time_plot",
                "460px",
            ),
            _chart_block("Errores en holdout (RMSE, MAE, MAPE)", "fb_acc_plot", "400px"),
            _chart_block("|Error| por horizonte (holdout)", "fb_horizon_plot", "420px"),
        )

        return ui.div(
            {"class": "bench-results-root mt-3"},
            placeholder,
            chart_stack,
        )

    @reactive.Effect
    @reactive.event(input.run_full_benchmark)
    def handle_full_benchmark():
        if not BENCHMARK_AVAILABLE or not ARIMA_BENCH_AVAILABLE:
            st = app_state.get().copy()
            st["fb_status"] = "error"
            st["fb_error"] = "Dependencias de benchmark no disponibles"
            app_state.set(st)
            return

        try:
            raw_n = input.fb_n_obs()
            n_grid = [int(x.strip()) for x in raw_n.split(",") if x.strip()]
            repeats = int(input.fb_repeats())
            csv_name = (input.fb_csv() or "arima_eval_benchmark.csv").strip()
            grid_mode = str(input.fb_grid_mode())
            manual_max_p = int(input.fb_manual_max_p()) if grid_mode == "manual" else None
            manual_max_q = int(input.fb_manual_max_q()) if grid_mode == "manual" else None
        except Exception as e:
            st = app_state.get().copy()
            st["fb_status"] = "error"
            st["fb_error"] = str(e)
            app_state.set(st)
            return

        st = app_state.get().copy()
        st["fb_status"] = "running"
        st["fb_error"] = None
        app_state.set(st)

        try:
            suite = ARIMABenchmarkSuite()
            with contextlib.redirect_stdout(_io.StringIO()):
                perf = suite.run_performance_benchmark(
                    n_obs_grid=n_grid or None,
                    repeats=max(1, repeats),
                    order=(1, 1, 1),
                    grid_mode=grid_mode,
                    manual_max_p=manual_max_p,
                    manual_max_q=manual_max_q,
                )
                acc = suite.run_accuracy_benchmark(
                    csv_name=csv_name,
                    value_column=None,
                    order=(1, 1, 1),
                    test_ratio=0.2,
                    grid_mode=grid_mode,
                    manual_max_p=manual_max_p,
                    manual_max_q=manual_max_q,
                )

            fig_t = suite.build_performance_time_figure(perf)
            fig_a = suite.build_accuracy_figure(acc)

            fig_h = suite.build_error_by_horizon_figure(
                csv_name,
                (1, 1, 1),
                0.2,
                grid_mode=grid_mode,
                manual_max_p=manual_max_p,
                manual_max_q=manual_max_q,
            )
            if fig_h is None:
                fig_h, ax = plt.subplots(figsize=(5, 2))
                ax.text(0.5, 0.5, "Error por horizonte no disponible", ha="center")
                ax.axis("off")

            st2 = app_state.get().copy()
            st2["fb_status"] = "done"
            st2["fb_perf_time_fig"] = fig_t
            st2["fb_acc_fig"] = fig_a
            st2["fb_horizon_fig"] = fig_h
            st2["fb_error"] = None
            app_state.set(st2)
        except Exception as e:
            st2 = app_state.get().copy()
            st2["fb_status"] = "error"
            st2["fb_error"] = f"{e}\n{traceback.format_exc()}"
            app_state.set(st2)

    @output
    @render.plot
    def fb_perf_time_plot():
        req(app_state.get().get("fb_status") == "done")
        return app_state.get().get("fb_perf_time_fig")

    @output
    @render.plot
    def fb_acc_plot():
        req(app_state.get().get("fb_status") == "done")
        return app_state.get().get("fb_acc_fig")

    @output
    @render.plot
    def fb_horizon_plot():
        req(app_state.get().get("fb_status") == "done")
        return app_state.get().get("fb_horizon_fig")
