# Benchmark feature server logic
from shiny import ui, reactive, render
from services.tslib_service import TSLibService
import matplotlib
import matplotlib.pyplot as plt
import io
import time
import os
import io

try:
    from .runner import BenchmarkRunner
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

try:
    from .arima_benchmark import ARIMABenchmarkSuite
    ARIMA_BENCH_AVAILABLE = True
except ImportError:
    ARIMA_BENCH_AVAILABLE = False
    ARIMABenchmarkSuite = None  # type: ignore

def register_benchmark_server(input, output, session, app_state):
    """Register benchmark server functions"""
    
    # Render execution status
    @output
    @render.ui
    def bench_execution_status():
        state = app_state.get()
        if not BENCHMARK_AVAILABLE:
            return ui.div(
                ui.tags.p("⚠️ La suite de benchmark no está disponible.", class_="text-danger mt-3")
            )
            
        status = state.get("bench_status", "idle")
        if status == "running":
            return ui.div(
                ui.tags.div(class_="spinner-border text-primary", role="status"),
                ui.tags.span(" Ejecutando benchmark... esto puede tomar varios minutos.", class_="ms-2"),
                class_="mt-3 text-center"
            )
        elif status == "error":
            error_msg = state.get("bench_error", "Error desconocido")
            return ui.div(
                ui.tags.p(f"❌ Error: {error_msg}", class_="text-danger mt-3")
            )
        elif status == "done":
            return ui.div(
                ui.tags.p("✅ Benchmark completado con éxito.", class_="text-success mt-3")
            )
        
        return ui.div()

    # Reactive button rendering for loading state
    @output
    @render.ui
    def bench_run_button_ui():
        state = app_state.get()
        status = state.get("bench_status", "idle")
        
        if status == "running":
            return ui.input_action_button(
                "run_benchmark", 
                "⏳ Procesando...", 
                class_="btn btn-primary btn-lg w-100 mt-4 disabled",
                disabled=True
            )
        
        return ui.input_action_button(
            "run_benchmark", 
            "▶️ Ejecutar Benchmark", 
            class_="btn btn-primary btn-lg w-100 mt-4"
        )
    
    # Main execution handler
    @reactive.Effect
    @reactive.event(input.run_benchmark)
    def handle_run_benchmark():
        """Run the actual benchmark asynchronously-like"""
        if not BENCHMARK_AVAILABLE:
            return
            
        # Parse inputs
        try:
            n_obs_str = input.bench_n_obs()
            n_obs_grid = [int(n.strip()) for n in n_obs_str.split(',') if n.strip()]
            if not n_obs_grid:
                raise ValueError("Se requiere al menos un tamaño de serie")
                
            repeats = int(input.bench_repeats())
            if repeats < 1:
                repeats = 1
        except Exception as e:
            new_state = app_state.get().copy()
            new_state["bench_status"] = "error"
            new_state["bench_error"] = str(e)
            app_state.set(new_state)
            return
            
        # Set running state
        new_state = app_state.get().copy()
        new_state["bench_status"] = "running"
        app_state.set(new_state)
        
        # Give UI a moment to update
        
        try:
            # Create a silent version of the runner that doesn't print to stdout
            runner = BenchmarkRunner(n_obs_grid=n_obs_grid, repeats=repeats, seed=42)
            
            # Instead of a simple `run`, we mock out prints or just let it run
            # But redirect stdout to avoid clutter
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run()
            
            # Save results to state
            results = {
                "results": runner.results,
                "elbows": runner.elbow_threshold(),
                "n_obs_grid": runner.n_obs_grid,
                "models": runner.models
            }
            
            # Generate the plot
            fig = _create_elbow_plot(runner)
            
            new_state = app_state.get().copy()
            new_state["bench_status"] = "done"
            new_state["bench_results"] = results
            new_state["bench_plot"] = fig
            app_state.set(new_state)
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            new_state = app_state.get().copy()
            new_state["bench_status"] = "error"
            new_state["bench_error"] = str(e)
            print(error_msg)
            app_state.set(new_state)

    @output
    @render.ui
    def bench_results_ui():
        """Render the results block"""
        state = app_state.get()
        status = state.get("bench_status", "idle")
        results = state.get("bench_results", None)
        
        if status == "idle":
            return ui.div(
                ui.tags.p("Configura los parámetros y haz clic en Ejecutar.", class_="text-center text-muted p-5")
            )
            
        if status == "running":
            return ui.div(
                ui.tags.div(class_="spinner-grow text-primary mt-5", role="status", style="width: 3rem; height: 3rem;"),
                ui.tags.h5("Ejecutando modelado paralelo y secuencial..."),
                ui.tags.p("Esto tomará un tiempo dependiendo de los tamaños elegidos.", class_="text-muted"),
                class_="text-center p-5"
            )
            
        if status == "error":
            return ui.div(
                ui.tags.p("Ocurrió un error al ejecutar el benchmark.", class_="text-center text-danger p-5")
            )
            
        if status == "done" and results:
            return ui.div(
                ui.output_plot("bench_elbow_plot", height="800px"),
                ui.tags.h5("Resumen", class_="mt-4"),
                ui.output_ui("bench_summary_cards")
            )
            
        return ui.div()
        
    @output
    @render.plot
    def bench_elbow_plot():
        state = app_state.get()
        fig = state.get("bench_plot")
        if fig:
            return fig
        # Fallback
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos de gráfico", ha="center")
        return fig
        
    @output
    @render.ui
    def bench_summary_cards():
        state = app_state.get()
        results = state.get("bench_results")
        if not results:
            return ui.div()
            
        elbows = results.get("elbows", {})
        
        cards = []
        for model_name, threshold in elbows.items():
            threshold_text = f"N >= {threshold}" if threshold is not None else "Pierde paralelo"
            icon = "📈" if threshold is not None else "📉"
            status_class = "border-success" if threshold is not None else "border-warning"
            
            cards.append(
                ui.div(
                    ui.div(
                        ui.div(icon, class_="metric-icon"),
                        ui.div(model_name, class_="metric-label fw-bold"),
                        ui.div(threshold_text, class_="metric-value", style="font-size: 1.2rem;"),
                        class_="metric-content"
                    ),
                    class_=f"metric-card {status_class}"
                )
            )
            
        return ui.div(*cards, class_="metrics-grid")

    @output
    @render.ui
    def arima_bench_status_ui():
        if not ARIMA_BENCH_AVAILABLE:
            return ui.div(
                ui.tags.p(
                    "Módulo de benchmark ARIMA no disponible (revisa dependencias).",
                    class_="text-warning mt-2",
                )
            )
        status = app_state.get().get("arima_bench_status", "idle")
        if status == "running":
            return ui.div(
                ui.tags.span("Ejecutando benchmark ARIMA (TSLib vs referencias)…", class_="text-info mt-2"),
                class_="mt-2",
            )
        if status == "error":
            err = app_state.get().get("arima_bench_error") or "Error"
            return ui.div(ui.tags.p(f"❌ {err}", class_="text-danger mt-2"))
        return ui.div()

    @output
    @render.ui
    def arima_bench_summary_ui():
        txt = app_state.get().get("arima_bench_summary")
        if not txt:
            return ui.div()
        return ui.div(ui.tags.pre(txt, class_="small bench-summary-pre"), class_="mt-2")

    @output
    @render.plot
    def arima_bench_perf_plot():
        fig = app_state.get().get("arima_bench_perf_fig")
        if fig is not None:
            return fig
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(
            0.5,
            0.5,
            "Ejecuta el benchmark ARIMA",
            ha="center",
            va="center",
            color="gray",
        )
        ax.axis("off")
        return fig

    @output
    @render.plot
    def arima_bench_acc_plot():
        fig = app_state.get().get("arima_bench_acc_fig")
        if fig is not None:
            return fig
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(
            0.5,
            0.5,
            "Ejecuta el benchmark ARIMA",
            ha="center",
            va="center",
            color="gray",
        )
        ax.axis("off")
        return fig

    @reactive.Effect
    @reactive.event(input.run_arima_triple_benchmark)
    def handle_arima_triple_benchmark():
        if not ARIMA_BENCH_AVAILABLE or ARIMABenchmarkSuite is None:
            st = app_state.get().copy()
            st["arima_bench_status"] = "error"
            st["arima_bench_error"] = "ARIMABenchmarkSuite no importable"
            app_state.set(st)
            return

        try:
            raw = input.bench_arima_n_obs()
            n_grid = [int(x.strip()) for x in raw.split(",") if x.strip()]
            repeats = int(input.bench_arima_repeats())
            if repeats < 1:
                repeats = 1
            csv_name = (input.bench_arima_csv() or "synthetic_arima_211.csv").strip()
        except Exception as e:
            st = app_state.get().copy()
            st["arima_bench_status"] = "error"
            st["arima_bench_error"] = str(e)
            app_state.set(st)
            return

        st = app_state.get().copy()
        st["arima_bench_status"] = "running"
        st["arima_bench_error"] = None
        app_state.set(st)

        try:
            import contextlib
            import io as _io

            suite = ARIMABenchmarkSuite()
            with contextlib.redirect_stdout(_io.StringIO()):
                perf = suite.run_performance_benchmark(
                    n_obs_grid=n_grid or None,
                    repeats=repeats,
                    order=(1, 1, 1),
                )
                acc = suite.run_accuracy_benchmark(
                    csv_name=csv_name,
                    value_column=None,
                    order=(1, 1, 1),
                    test_ratio=0.2,
                )
            fig_p = suite.build_performance_figure(perf)
            fig_a = suite.build_accuracy_figure(acc)

            lines = [
                f"N* speedup≥1 (lineal vs workflow): {perf.get('crossover_workflow')}",
                f"N* speedup≥1 (lineal vs Spark+statsmodels): {perf.get('crossover_spark_sm')}",
                f"N* speedup≥1 (lineal vs statsmodels local): {perf.get('crossover_statsmodels_local')}",
                "",
                "Métricas holdout (RMSE / MAE / MAPE):",
            ]
            for name, block in (acc.get("metrics") or {}).items():
                if "error" in block:
                    lines.append(f"  {name}: {block.get('error')}")
                else:
                    lines.append(
                        f"  {name}: RMSE={block.get('rmse'):.4f} MAE={block.get('mae'):.4f} MAPE={block.get('mape'):.2f}%"
                    )

            st2 = app_state.get().copy()
            st2["arima_bench_status"] = "done"
            st2["arima_bench_perf_fig"] = fig_p
            st2["arima_bench_acc_fig"] = fig_a
            st2["arima_bench_summary"] = "\n".join(lines)
            st2["arima_bench_error"] = None
            app_state.set(st2)
        except Exception as e:
            import traceback

            st2 = app_state.get().copy()
            st2["arima_bench_status"] = "error"
            st2["arima_bench_error"] = f"{e}\n{traceback.format_exc()}"
            app_state.set(st2)


def _create_elbow_plot(runner):
    """Recreates the runner's internal plot securely for Shiny."""
    n_models = len(runner.models)
    # Increase vertical space: 5 inches per model to avoid overlap
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models), squeeze=False)
    
    speedups = runner.speedups()
    
    for idx, model_name in enumerate(runner.models):
        ax_time = axes[idx, 0]
        ax_speedup = axes[idx, 1]
        
        n_obs = runner.n_obs_grid
        t_seq = [runner.results[model_name][n]['sequential'] for n in n_obs]
        t_par = [runner.results[model_name][n]['parallel_all'] for n in n_obs]
        
        # 1. Fit Time Plot
        ax_time.plot(n_obs, t_seq, 'o-', color='#ff6b6b', label='Secuencial (n_jobs=1)', linewidth=2)
        ax_time.plot(n_obs, t_par, 's-', color='#10ac84', label='Paralelo (n_jobs=-1)', linewidth=2)
        # ax_time.set_title(f'{model_name} - Tiempo de Ajuste', fontsize=13, pad=15)
        ax_time.set_xlabel('Tamaño de Serie (n_obs)', fontsize=10)
        ax_time.set_ylabel('Tiempo (segundos)', fontsize=10)
        ax_time.set_xscale('log')
        ax_time.grid(True, alpha=0.2, linestyle=':')
        ax_time.legend(loc='upper left', fontsize=9)
        
        # 2. Speedup Plot
        s_vals = [speedups[model_name][n] for n in n_obs]
        ax_speedup.plot(n_obs, s_vals, 'o-', color='#54a0ff', label='Speedup', linewidth=2.5)
        ax_speedup.axhline(y=1.0, color='#8395a7', linestyle='--', alpha=0.6, label='Equilibrio (1x)')
        ax_speedup.axhline(y=1.1, color='#ff9f43', linestyle=':', linewidth=2, label='Umbral (1.1x)')
        
        # Highlight elbow
        elbow = runner.elbow_threshold()[model_name]
        if elbow is not None:
            ax_speedup.axvline(x=elbow, color='#ff9f43', alpha=0.2, linewidth=12, zorder=0)
            ax_speedup.text(elbow, max(s_vals)*0.9, ' Codo', color='#ff9f43', fontweight='bold', fontsize=10)
            
        # ax_speedup.set_title(f'{model_name} - Aceleración (Speedup)', fontsize=13, pad=15)
        ax_speedup.set_xlabel('Tamaño de Serie (n_obs)', fontsize=10)
        ax_speedup.set_ylabel('Speedup (Sec / Par)', fontsize=10)
        ax_speedup.set_xscale('log')
        ax_speedup.grid(True, alpha=0.2, linestyle=':')
        ax_speedup.legend(loc='upper left', fontsize=9)
    
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes.flatten():
        ax.set_facecolor('#262626')
        ax.tick_params(colors='white', labelsize=9)
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('#444444')
        ax.spines['left'].set_color('#444444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Fix legend visual
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_color('white')
            legend.get_frame().set_facecolor('#1a1a1a')
            legend.get_frame().set_edgecolor('#444444')
            legend.get_frame().set_alpha(0.8)
            
    # Adjust layout to prevent overlap explicitly
    plt.tight_layout(pad=4.0)
    return fig
