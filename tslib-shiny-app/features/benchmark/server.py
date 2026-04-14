# Benchmark feature server — single full run
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
        COLOR_NEUTRAL,
        COLOR_SM_REF,
        COLOR_SPARK_SM,
        COLOR_TSLIB,
        COLOR_WORKFLOW,
        LABEL_PARALLEL_WORKFLOW,
        LABEL_SPARK_STATSMODELS,
        LABEL_STATSMODELS_LOCAL,
        LABEL_TSLIB_LINEAR,
    )

    ARIMA_BENCH_AVAILABLE = True
except ImportError:
    ARIMA_BENCH_AVAILABLE = False
    ARIMABenchmarkSuite = None  # type: ignore
    LABEL_TSLIB_LINEAR = "TSLib lineal"
    LABEL_PARALLEL_WORKFLOW = "ParallelARIMAWorkflow"
    LABEL_SPARK_STATSMODELS = "Spark · statsmodels"
    LABEL_STATSMODELS_LOCAL = "statsmodels ref."
    COLOR_TSLIB = "#E07A5F"
    COLOR_WORKFLOW = "#2A9D8F"
    COLOR_SPARK_SM = "#457B9D"
    COLOR_SM_REF = "#E9C46A"
    COLOR_NEUTRAL = "#94A3B8"

_METRIC_SUMMARY_LABELS = {
    "tslib_linear": LABEL_TSLIB_LINEAR,
    "parallel_workflow": LABEL_PARALLEL_WORKFLOW,
    "spark_statsmodels": LABEL_SPARK_STATSMODELS,
    "statsmodels_local": LABEL_STATSMODELS_LOCAL,
}


def _fmt_crossover(v):
    return "—" if v is None else str(v)


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
                    "Calculando tiempos, métricas y figuras. Puede tardar varios minutos.",
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
            _chart_block("Tiempos de ajuste", "fb_perf_time_plot", "440px"),
            _chart_block("Speedup frente a TSLib lineal", "fb_perf_speedup_plot", "400px"),
            _chart_block("Errores en holdout", "fb_acc_plot", "380px"),
            _chart_block("ACF, PACF, residuos y Q-Q", "fb_diag_plot", "500px"),
            _chart_block("|Error| por horizonte — ParallelARIMAWorkflow vs referencias", "fb_horizon_plot", "400px"),
            ui.div(
                ui.tags.p(
                    "Paralelismo n_jobs en ARIMAModel; distinto del pipeline ParallelARIMAWorkflow arriba.",
                    class_="bench-hint small text-muted mb-2",
                ),
                _chart_block("Secuencial vs paralelo interno", "fb_elbow_plot", "880px"),
                class_="bench-njobs-block",
            ),
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
            nj_raw = input.fb_njobs_grid()
            nj_grid = [int(x.strip()) for x in nj_raw.split(",") if x.strip()]
            nj_repeats = int(input.fb_njobs_repeats())
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
            fig_s = suite.build_performance_speedup_figure(perf)
            fig_a = suite.build_accuracy_figure(acc)

            train_y = None
            try:
                import pandas as pd

                from .arima_benchmark import default_sampler_datasets_dir

                p = default_sampler_datasets_dir() / csv_name
                if p.exists():
                    df = pd.read_csv(p)
                    vc = None
                    for cand in ("value", "y", "passengers", "close", "temperature", "sunspots"):
                        if cand in df.columns:
                            vc = cand
                            break
                    if vc is None:
                        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                        vc = num[-1] if num else None
                    if vc:
                        y = pd.to_numeric(df[vc], errors="coerce").values.astype(float)
                        n = len(y)
                        n_test = max(1, int(round(n * 0.2)))
                        train_y = y[: (n - n_test)]
            except Exception:
                train_y = None

            fig_d = None
            if train_y is not None and len(train_y) > 20:
                fig_d = suite.build_exploratory_diagnostics_figure(train_y, (1, 1, 1))
            else:
                fig_d, ax = plt.subplots(figsize=(5, 2))
                ax.text(0.5, 0.5, "No se pudo cargar la serie de entrenamiento", ha="center")
                ax.axis("off")

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

            if nj_grid:
                runner = BenchmarkRunner(n_obs_grid=nj_grid, repeats=max(1, nj_repeats), seed=42)
                with contextlib.redirect_stdout(_io.StringIO()):
                    runner.run()
                elbow_fig = _create_elbow_plot(runner)
            else:
                fig_e, ax_e = plt.subplots(figsize=(6, 2.2))
                fig_e.patch.set_facecolor("#1a1a1a")
                ax_e.set_facecolor("#262626")
                ax_e.text(
                    0.5,
                    0.5,
                    "Indica un grid n_jobs para esta sección.",
                    ha="center",
                    va="center",
                    color="#94a3b8",
                    fontsize=11,
                )
                ax_e.axis("off")
                elbow_fig = fig_e

            lines = [
                "ARIMA fijo (1,1,1)",
                f"N mín. con speedup ≥ 1 · {LABEL_PARALLEL_WORKFLOW}: {_fmt_crossover(perf.get('crossover_workflow'))}",
                f"N mín. con speedup ≥ 1 · {LABEL_SPARK_STATSMODELS}: {_fmt_crossover(perf.get('crossover_spark_sm'))}",
                f"N mín. con speedup ≥ 1 · {LABEL_STATSMODELS_LOCAL}: {_fmt_crossover(perf.get('crossover_statsmodels_local'))}",
                "",
                "Holdout",
            ]
            for key, block in (acc.get("metrics") or {}).items():
                label = _METRIC_SUMMARY_LABELS.get(key, key)
                if "error" in block:
                    lines.append(f"  {label}: {block.get('error')}")
                else:
                    lines.append(
                        f"  {label}: RMSE {block.get('rmse'):.4f} · MAE {block.get('mae'):.4f} · "
                        f"MAPE {block.get('mape'):.2f}%"
                    )
            if nj_grid:
                lines.extend(["", "Sección n_jobs: ver gráfico anterior."])

            st2 = app_state.get().copy()
            st2["fb_status"] = "done"
            st2["fb_perf_time_fig"] = fig_t
            st2["fb_perf_speedup_fig"] = fig_s
            st2["fb_acc_fig"] = fig_a
            st2["fb_diag_fig"] = fig_d
            st2["fb_horizon_fig"] = fig_h
            st2["fb_elbow_fig"] = elbow_fig
            st2["fb_summary_text"] = "\n".join(lines)
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
    def fb_perf_speedup_plot():
        req(app_state.get().get("fb_status") == "done")
        return app_state.get().get("fb_perf_speedup_fig")

    @output
    @render.plot
    def fb_acc_plot():
        req(app_state.get().get("fb_status") == "done")
        return app_state.get().get("fb_acc_fig")

    @output
    @render.plot
    def fb_diag_plot():
        req(app_state.get().get("fb_status") == "done")
        return app_state.get().get("fb_diag_fig")

    @output
    @render.plot
    def fb_horizon_plot():
        req(app_state.get().get("fb_status") == "done")
        return app_state.get().get("fb_horizon_fig")

    @output
    @render.plot
    def fb_elbow_plot():
        req(app_state.get().get("fb_status") == "done")
        return app_state.get().get("fb_elbow_fig")

    @output
    @render.ui
    def fb_summary_ui():
        st = app_state.get().get("fb_status", "idle")
        if st != "done":
            return ui.div()
        txt = app_state.get().get("fb_summary_text")
        if not txt:
            return ui.div()
        return ui.div(
            ui.tags.h6("Resumen textual", class_="bench-section-title"),
            ui.tags.pre(txt, class_="bench-summary-pre"),
            class_="bench-summary-wrap",
        )


def _create_elbow_plot(runner):
    """Sequential vs parallel n_jobs: one row per model, legend below axes."""
    n_models = len(runner.models)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4.2 * n_models), squeeze=False)
    speedups = runner.speedups()

    for idx, model_name in enumerate(runner.models):
        ax_time = axes[idx, 0]
        ax_sp = axes[idx, 1]
        n_obs = runner.n_obs_grid
        t_seq = [runner.results[model_name][n]["sequential"] for n in n_obs]
        t_par = [runner.results[model_name][n]["parallel_all"] for n in n_obs]

        ax_time.plot(
            n_obs,
            t_seq,
            "o-",
            color=COLOR_TSLIB,
            label="Secuencial n_jobs=1",
            linewidth=2,
        )
        ax_time.plot(
            n_obs,
            t_par,
            "s-",
            color=COLOR_WORKFLOW,
            label="Paralelo n_jobs=-1",
            linewidth=2,
        )
        ax_time.set_xlabel("n_obs", fontsize=10)
        ax_time.set_ylabel("Tiempo (s)", fontsize=10)
        ax_time.set_xscale("log")
        ax_time.set_yscale("log")
        ax_time.set_title(f"{model_name} — tiempo", fontsize=11, color="white", pad=8)
        ax_time.grid(True, alpha=0.2, linestyle=":")
        ax_time.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            facecolor="#1a1a1a",
            edgecolor="#555555",
            labelcolor="white",
        )

        s_vals = [speedups[model_name][n] for n in n_obs]
        ax_sp.plot(n_obs, s_vals, "o-", color=COLOR_SPARK_SM, label="Speedup", linewidth=2)
        ax_sp.axhline(y=1.0, color=COLOR_NEUTRAL, linestyle="--", alpha=0.6, label="1×")
        ax_sp.axhline(y=1.1, color=COLOR_SM_REF, linestyle=":", linewidth=2, label="Umbral 1.1×")
        elbow = runner.elbow_threshold()[model_name]
        if elbow is not None and len(s_vals):
            ax_sp.axvline(x=elbow, color="#6b5b4f", alpha=0.35, linewidth=10, zorder=0)
            ax_sp.text(
                elbow,
                max(s_vals) * 0.85,
                " codo",
                color="#c4b5a0",
                fontsize=9,
            )
        ax_sp.set_xlabel("n_obs", fontsize=10)
        ax_sp.set_ylabel("Speedup (t_seq / t_par)", fontsize=10)
        ax_sp.set_title(f"{model_name} — speedup", fontsize=11, color="white", pad=8)
        ax_sp.set_xscale("log")
        ax_sp.grid(True, alpha=0.2, linestyle=":")
        ax_sp.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            facecolor="#1a1a1a",
            edgecolor="#555555",
            labelcolor="white",
        )

    fig.patch.set_facecolor("#1a1a1a")
    for ax in axes.flatten():
        ax.set_facecolor("#262626")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        for s in ax.spines.values():
            s.set_color("#444444")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.45)
    return fig
