"""Shiny app entry point: TSLib time series analysis wizard."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import time
import traceback
from pathlib import Path

from shiny import App, ui, reactive, render

_APP_STATIC = Path(__file__).resolve().parent / "static"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from components.stepper import StepperComponent
from components.layout import (
    create_app_layout,
    create_data_table,
    create_metric_card,
    create_dual_route_metric_card,
    create_form_group,
    create_file_upload_area,
)
from features.upload.ui import render_upload_ui
from features.visualization.ui import render_visualization_ui
from features.model_selection.ui import render_model_selection_ui
from features.results.ui import render_results_ui
from features.benchmark.ui import render_benchmark_ui
from features.benchmark.server import register_benchmark_server

from services.tslib_service import TSLibService, run_with_recorded_warnings
from services.evaluation_plots import (
    plot_residual_acf_plotly,
    plot_residual_hist_qq_plotly,
    plot_residuals_plotly,
    plot_standardized_residuals_plotly,
)
from config_limits import MAX_UPLOAD_FILE_BYTES
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from shinywidgets import render_plotly, output_widget


def _forecast_seq_len(seq: Any) -> int:
    """Length of a forecast sequence without bool(ndarray) (numpy-safe)."""
    if seq is None:
        return 0
    if isinstance(seq, np.ndarray):
        return int(seq.size)
    try:
        return len(seq)
    except TypeError:
        return 0


def _has_forecast_values(d: Optional[Dict[str, Any]], key: str = "forecast") -> bool:
    """True if d has a non-empty forecast field (safe when values are numpy arrays)."""
    if not d:
        return False
    return _forecast_seq_len(d.get(key)) > 0


def _as_forecast_array(d: Optional[Dict[str, Any]], key: str = "forecast") -> np.ndarray:
    """Forecast values as 1-D float array; empty if missing (avoids `arr or []`)."""
    if not d:
        return np.array([], dtype=float)
    v = d.get(key)
    if v is None:
        return np.array([], dtype=float)
    return np.asarray(v, dtype=float).ravel()


def _pct_abs_parallel_vs_linear(v_lineal: float, v_par: float) -> float:
    """Absolute relative deviation of parallel vs linear: 100 * |P - L| / |L|."""
    denom = max(abs(float(v_lineal)), 1e-12)
    return 100.0 * abs(float(v_par) - float(v_lineal)) / denom


def _plotly_forecast_layout(fig: go.Figure, title: str, *, yaxis_title: str = "Valor") -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#2d2d2d",
        font=dict(color="#ececec", size=12),
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Tiempo", gridcolor="#444444"),
        yaxis=dict(title=yaxis_title, gridcolor="#444444"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        dragmode="zoom",
        margin=dict(l=50, r=24, t=56, b=40),
    )
    return fig


def _plotly_empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color="#aaa", size=14),
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1a1a1a", plot_bgcolor="#2d2d2d", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


STEPS = [
    {
        "title": "📁 Carga de datos",
        "description": "Sube y configura tu serie temporal"
    },
    {
        "title": "📊 Exploración",
        "description": "Explora y analiza los datos"
    },
    {
        "title": "⚙️ Modelo y ejecución",
        "description": "Configura el modelo y ejecuta el análisis"
    },
    {
        "title": "📈 Resultados",
        "description": "Revisa métricas y predicciones"
    }
]

stepper = StepperComponent(STEPS)

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href=(
                "https://fonts.googleapis.com/css2?"
                "family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400"
                "&family=Playfair+Display:wght@400;600;700;800"
                "&family=JetBrains+Mono:wght@400;500"
                "&display=swap"
            ),
        ),
        ui.include_css(_APP_STATIC / "styles.css"),
        ui.tags.script(
            """
            Shiny.addCustomMessageHandler("update_preview", function(message) {
              const placeholder = document.getElementById("data_preview_placeholder");
              const content = document.getElementById("data_preview_content");
              if (placeholder && content) {
                placeholder.style.display = "none";
                content.classList.remove("d-none");
                content.classList.add("d-block");
                const rowCount = document.getElementById("row_count");
                const colCount = document.getElementById("col_count");
                const fileSize = document.getElementById("file_size");
                if (rowCount) rowCount.textContent = "Filas: " + message.rows;
                if (colCount) colCount.textContent = "Columnas: " + message.columns;
                if (fileSize) fileSize.textContent = "Tamaño: " + message.size;
              }
            });
            """
        ),
    ),

    # Main app layout with Navbar
    ui.page_navbar(
        ui.nav_panel("🧪 Asistente de Análisis", 
            create_app_layout(
                title="Análisis de Series de Tiempo",
                subtitle="Análisis avanzado con modelos de series temporales"
            ),
            ui.output_ui("stepper_navigation"),
            ui.div(
                ui.output_ui("step_content"),
                class_="container-fluid"
            )
        ),
        ui.nav_panel(
            "Series temporales paralelas",
            create_app_layout(
                title="Series temporales paralelas",
                subtitle="Benchmark y comparativas ARIMA",
            ),
            ui.div(
                render_benchmark_ui(),
                class_="container-fluid mt-4"
            )
        ),
        title="Series temporales paralelas",
        bg="var(--bg)",
        inverse=True,
    )
)

def server(input, output, session):
    """Server logic with reactive event handling."""
    app_state = reactive.Value({
        "current_step": 0,
        "data_loaded": False,
        "data_validated": False,
        "uploaded_data": None,
        "value_column": None,
        "date_column": None,
        "model_type": None,
        "model_config": {},
        "fitted_model": None,
        "forecast_results": None,
        "parallel_workflow": None,
        "parallel_forecast_results": None,
        "analysis_complete": False,
        "execution_log": [],
        "exploratory_analysis": None,
        "auto_select": True,
        "execution_metadata": {},
        "runtime_warnings": [],
        # Benchmark specific state
        "bench_status": "idle",
        "bench_plot": None,
        "bench_results": None,
        "bench_error": None,
        "arima_bench_status": "idle",
        "arima_bench_error": None,
        "arima_bench_perf_fig": None,
        "arima_bench_acc_fig": None,
        "arima_bench_summary": None,
        # Unified benchmark tab (fb_*)
        "fb_status": "idle",
        "fb_error": None,
        "fb_perf_time_fig": None,
        "fb_perf_speedup_fig": None,
        "fb_acc_fig": None,
        "fb_diag_fig": None,
        "fb_horizon_fig": None,
        "fb_elbow_fig": None,
        "fb_summary_text": None,
    })
    uploaded_dataframe = reactive.Value(None)
    
    # Register benchmark server logic
    register_benchmark_server(input, output, session, app_state)
    
    # Initialize TSLib service
    tslib_service = TSLibService()

    # ARIMA comparison colors: lineal = verde, paralelo = rosa (aligned with métricas duales)
    _COL_ARIMA_LINEAL = "#00d4aa"
    _COL_ARIMA_PARALELO = "#e879a8"

    def _style_forecast_axes(ax, fig):
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#2d2d2d")
        fig.patch.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for s in ("bottom", "left"):
            ax.spines[s].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _figure_arima_lineal_vs_paralelo_plotly() -> go.Figure:
        """Histórico + pronóstico statsmodels + pronóstico workflow Spark (Plotly: zoom, pan)."""
        state = app_state.get()
        df = uploaded_dataframe.get()
        value_col = state.get("value_column")
        forecast_results = state.get("forecast_results") or {}
        parallel_forecast_results = state.get("parallel_forecast_results") or {}
        if not state.get("analysis_complete") or df is None or value_col is None:
            return _plotly_empty_fig("Ejecuta el análisis primero")

        if pd.api.types.is_numeric_dtype(df[value_col]):
            historical = df[value_col].values
        else:
            historical = tslib_service.convert_to_numeric(df, value_col).values
        historical = np.asarray(historical, dtype=float)

        fc_lin = np.asarray(forecast_results.get("forecast", []), dtype=float).ravel()
        fc_par = np.asarray(parallel_forecast_results.get("forecast", []), dtype=float).ravel()
        lo_l = forecast_results.get("lower_bound")
        up_l = forecast_results.get("upper_bound")
        lo_p = parallel_forecast_results.get("lower_bound")
        up_p = parallel_forecast_results.get("upper_bound")

        n_hist = len(historical)
        n_lin = len(fc_lin)
        n_par = len(fc_par)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(n_hist)),
                y=historical.tolist(),
                mode="lines",
                name="Histórico",
                line=dict(color="#00d4aa", width=1.5),
            )
        )
        if n_lin:
            x_lin = list(range(n_hist, n_hist + n_lin))
            lo_a = np.asarray(lo_l, dtype=float).ravel() if lo_l is not None else None
            up_a = np.asarray(up_l, dtype=float).ravel() if up_l is not None else None
            if lo_a is not None and up_a is not None and len(lo_a) >= n_lin and len(up_a) >= n_lin:
                fig.add_trace(
                    go.Scatter(
                        x=x_lin + x_lin[::-1],
                        y=lo_a[:n_lin].tolist() + up_a[:n_lin][::-1].tolist(),
                        fill="toself",
                        fillcolor="rgba(0,212,170,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IC 95% lineal",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=x_lin,
                    y=fc_lin.tolist(),
                    mode="lines",
                    name="Pronóstico lineal",
                    line=dict(color=_COL_ARIMA_LINEAL, width=2, dash="dash"),
                )
            )
        if n_par:
            x_par = list(range(n_hist, n_hist + n_par))
            lo_b = np.asarray(lo_p, dtype=float).ravel() if lo_p is not None else None
            up_b = np.asarray(up_p, dtype=float).ravel() if up_p is not None else None
            if lo_b is not None and up_b is not None and len(lo_b) >= n_par and len(up_b) >= n_par:
                fig.add_trace(
                    go.Scatter(
                        x=x_par + x_par[::-1],
                        y=lo_b[:n_par].tolist() + up_b[:n_par][::-1].tolist(),
                        fill="toself",
                        fillcolor="rgba(232,121,168,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IC 95% paralelo",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=x_par,
                    y=fc_par.tolist(),
                    mode="lines",
                    name="Pronóstico paralelo",
                    line=dict(color=_COL_ARIMA_PARALELO, width=2, dash="dash"),
                )
            )
        return _plotly_forecast_layout(fig, "Pronóstico: lineal (verde) vs paralelo (rosa)")

    # Stepper header renderer
    @render.ui
    def stepper_header():
        """Render stepper header reactively"""
        current_step = app_state.get()["current_step"]
        return ui.div(
            ui.div(
                ui.tags.h2(
                    STEPS[current_step]["title"],
                    class_="stepper-title"
                ),
                ui.div(
                    f"Paso {current_step + 1} de {len(STEPS)}",
                    class_="stepper-progress"
                ),
                ui.output_ui("spark_status_header_ui"),
                class_="stepper-header"
            ),
            class_="stepper-container"
        )

    @render.ui
    def spark_status_header_ui():
        """Mini status badge for Spark availability in header."""
        state = app_state.get()
        model_type = state.get("model_type")
        if model_type not in ["AR", "MA", "ARMA", "ARIMA"]:
            return ui.div()
        status = tslib_service.get_spark_parallel_status(model_type=model_type)
        cls = "status-success" if status.get("available") else "status-warning"
        text = "Spark paralelo: disponible" if status.get("available") else "Spark paralelo: no disponible"
        return ui.div(text, class_=f"status-indicator {cls}")
    
    # Stepper navigation renderer
    @render.ui
    def stepper_navigation():
        """Render stepper navigation reactively"""
        current_step = app_state.get()["current_step"]
        state = app_state.get()
        
        # Check if we can proceed to next step
        can_proceed = validate_current_step(current_step, state)
        
        # Previous button
        prev_button = None
        if current_step > 0:
            prev_button = ui.input_action_button(
                "prev_step",
                "← Anterior",
                class_="btn btn-secondary"
            )
        
        # Next button
        next_button = None
        if current_step < len(STEPS) - 1:
            if can_proceed:
                next_button = ui.input_action_button(
                    "next_step",
                    "Siguiente →",
                    class_="btn btn-primary"
                )
            else:
                # Disabled button - use HTML button with disabled attribute
                next_button = ui.tags.button(
                    "Siguiente →",
                    type="button",
                    class_="btn btn-primary",
                    disabled=True,
                    title="Completa los requisitos del paso actual para continuar"
                )
        
        return ui.div(
            ui.div(
                prev_button if prev_button else ui.div(),
                class_="d-flex"
            ),
            ui.div(
                next_button if next_button else ui.div(),
                class_="d-flex"
            ),
            class_="stepper-navigation"
        )
    
    # Step content renderer
    @render.ui
    def step_content():
        """Render content for current step"""
        current_step = app_state.get()["current_step"]
        state = app_state.get()
        
        if current_step == 0:
            return render_upload_ui()
        elif current_step == 1:
            return render_visualization_ui()
        elif current_step == 2:
            auto_select_value = state.get("auto_select", True)
            return render_model_selection_ui(
                auto_select_value=auto_select_value,
            )
        elif current_step == 3:
            return render_results_ui()
        else:
            return ui.div("Paso no válido", class_="alert alert-danger")
    
    @render.ui
    def model_type_select():
        """Render model type select with state hydration."""
        state = app_state.get()
        current = state.get("model_type")
        try:
            current_input = input.model_type() if hasattr(input, 'model_type') else None
        except Exception:
            current_input = None
        selected_value = current_input or current or "__none__"
        return ui.input_select(
            "model_type",
            "",
            choices={
                "__none__": "— Selecciona un modelo —",
                "AR": "AR - Autoregresivo",
                "MA": "MA - Media Móvil",
                "ARMA": "ARMA - Combinado",
                "ARIMA": "ARIMA - Integrado"
            },
            selected=selected_value
        )
    
    @render.ui
    def upload_area_ui():
        """Render upload area only when no data is loaded"""
        state = app_state.get()
        if state.get("data_loaded"):
            return ui.div()
        
        return ui.div(
            create_file_upload_area(
                input_id="file_upload",
                label="Seleccionar archivo",
                accept=".csv,.xlsx,.xls"
            ),
            ui.div(
                ui.tags.p("Formatos soportados CSV, Excel (.xlsx, .xls)", class_="text-muted"),
                class_="mt-2"
            )
        )
    
    @render.ui
    def data_preview_ui():
        """Render data preview after upload"""
        df = uploaded_dataframe.get()
        state = app_state.get()
        
        if df is None:
            return ui.div(
                ui.tags.p("No hay datos cargados", class_="text-muted text-center"),
                class_="data-preview-empty"
            )
        
        preview_df = df.head(10)
        table = create_data_table(
            preview_df.to_numpy().tolist(),
            headers=list(preview_df.columns)
        )
        file_info = state.get("uploaded_data") or {}
        file_size_kb = None
        if file_info and file_info.get("size") is not None:
            file_size_kb = f"{file_info['size'] / 1024:.1f} KB"
        
        return ui.div(
            ui.div(
                # Summary cards aligned at the same level
                create_metric_card(
                    f"{file_info.get('rows', '0')}", 
                    "Filas", 
                    "🧾"
                ),
                create_metric_card(
                    f"{file_info.get('columns', '0')}", 
                    "Columnas", 
                    "📊"
                ),
                create_metric_card(
                    f"{file_size_kb if file_size_kb else '—'}", 
                    "Tamaño", 
                    "💾"
                ),
                class_="metrics-grid"
            ),
            ui.div(
                ui.tags.h5("Vista previa de datos"),
                class_="mt-2"
            ),
            ui.div(
                table,
                class_="table-preview"
            ),
            class_="data-preview-content"
        )
    
    @render.ui
    def date_column_select():
        """Render date column select with state hydration to avoid resets"""
        df = uploaded_dataframe.get()
        state = app_state.get()
        
        if df is None:
            return ui.div()
        
        all_cols = list(df.columns)
        
        # Get current value from state
        current = state.get("date_column")
        
        # If input already has a value (during same session), prefer it
        try:
            current_input = input.date_column() if hasattr(input, 'date_column') else None
        except Exception:
            current_input = None
        
        # Detect datetime column for default
        datetime_col = tslib_service.detect_datetime_column(df)
        default_value = datetime_col if datetime_col else "(Ninguna)"
        
        # Use current input, then state, then default
        selected_value = current_input or (current if current else default_value)
        
        return ui.input_select(
            "date_column",
            "",
            choices=["(Ninguna)"] + all_cols,
            selected=selected_value
        )
    
    @render.ui
    def value_column_select():
        """Render value column select with state hydration to avoid resets"""
        df = uploaded_dataframe.get()
        state = app_state.get()
        
        if df is None:
            return ui.div()
        
        # Get numeric columns
        numeric_cols = tslib_service.get_numeric_columns(df)
        
        if not numeric_cols:
            return ui.div()
        
        # Get current value from state
        current = state.get("value_column")
        
        # If input already has a value (during same session), prefer it
        try:
            current_input = input.value_column() if hasattr(input, 'value_column') else None
        except Exception:
            current_input = None
        
        # Default to first numeric column
        default_value = numeric_cols[0] if numeric_cols else None
        
        # Use current input, then state, then default
        selected_value = current_input or (current if current else default_value)
        
        return ui.input_select(
            "value_column",
            "",
            choices=numeric_cols,
            selected=selected_value
        )
    
    @render.ui
    def column_selection_ui():
        """Render column selection UI after data is loaded"""
        df = uploaded_dataframe.get()
        state = app_state.get()
        
        if df is None:
            return ui.div()
        
        # Get numeric columns
        numeric_cols = tslib_service.get_numeric_columns(df)
        
        if not numeric_cols:
            return ui.div(
                ui.tags.p("⚠️ No se encontraron columnas numéricas en el archivo", class_="text-warning"),
                class_="mt-3"
            )
        
        return ui.div(
            ui.tags.h5("Configuración de columnas:", class_="mt-4"),
            ui.div(
                ui.div(
                    create_form_group(
                        label="Columna de Fecha/Tiempo",
                        control=ui.output_ui("date_column_select"),
                        help_text="Columna para el eje X en gráficos"
                    ),
                    class_="col-md-6"
                ),
                ui.div(
                    create_form_group(
                        label="Columna de Valores",
                        control=ui.output_ui("value_column_select"),
                        help_text="Selecciona la columna con los valores de la serie temporal"
                    ),
                    class_="col-md-6"
                ),
                class_="row"
            ),
            ui.div(
                ui.input_action_button("validate_data", "✓ Validar Datos", class_="btn btn-primary"),
                class_="mt-3"
            ),
            ui.output_ui("validation_results_ui"),
            class_="mt-3"
        )
    
    @render.ui
    def validation_results_ui():
        """Show validation results"""
        state = app_state.get()
        validation = state.get("validation_report", {})
        
        if not validation:
            return ui.div()
        
        messages = validation.get("messages", [])
        warnings = validation.get("warnings", [])
        runtime_warnings = validation.get("runtime_warnings", []) or []
        is_valid = validation.get("valid", False)
        quality = (validation.get("quality_report", {}) or {}).get("quality_metrics", {}) or {}

        result_class = "status-success" if is_valid else "status-warning"
        quality_rows = []
        if quality:
            completeness = float(quality.get("completeness", 0.0)) * 100
            validity = float(quality.get("validity", 0.0)) * 100
            consistency = float(quality.get("consistency", 0.0))
            quality_rows = [
                ["Completitud", f"{completeness:.2f}%"],
                ["Validez", f"{validity:.2f}%"],
                ["Consistencia", f"{consistency:.4f}"],
            ]

        return ui.div(
            ui.div(
                *[ui.div(msg, class_=f"status-indicator {result_class}") for msg in messages],
                *[ui.div(f"⚠️ {warn}", class_="status-indicator status-warning") for warn in warnings],
                *(
                    [
                        ui.tags.h6("Avisos del motor (Python / librerías)", class_="mt-3 mb-2"),
                        *[
                            ui.div(f"⚠ {rw}", class_="status-indicator status-warning")
                            for rw in runtime_warnings
                        ],
                    ]
                    if runtime_warnings
                    else []
                ),
                class_="mt-3"
            ),
            ui.div(
                ui.tags.h6("Calidad de datos", class_="mt-3"),
                create_data_table(quality_rows, headers=["Métrica", "Valor"]) if quality_rows else ui.div(),
                class_="mt-2"
            ) if quality_rows else ui.div(),
        )
    
    # Visualization renders
    @render.plot
    def time_series_plot():
        """Render time series plot"""
        df = uploaded_dataframe.get()
        state = app_state.get()
        value_col = state.get("value_column")
        date_col = state.get("date_column")
        
        if df is None or value_col is None:
            # Return empty plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'Carga datos primero', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Convert to numeric if needed
        if pd.api.types.is_numeric_dtype(df[value_col]):
            y_values = df[value_col]
        else:
            y_values = tslib_service.convert_to_numeric(df, value_col)
        
        # Handle missing values for plotting (forward fill)
        y_values_plot = y_values.copy()
        if y_values_plot.isna().any():
            # Use interpolation for missing values
            mask = y_values_plot.isna()
            if mask.any() and not mask.all():
                indices = np.arange(len(y_values_plot))
                y_values_plot[mask] = np.interp(indices[mask], indices[~mask], y_values_plot[~mask])
            else:
                y_values_plot = y_values_plot.fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        if date_col and date_col != "(Ninguna)" and date_col in df.columns:
            x = pd.to_datetime(df[date_col], errors='coerce')
            ax.plot(x, y_values_plot, linewidth=1.5, color='#00d4aa')
            ax.set_xlabel('Fecha')
        else:
            ax.plot(y_values_plot, linewidth=1.5, color='#00d4aa')
            ax.set_xlabel('Índice')
        
        ax.set_ylabel(value_col)
        ax.set_title('Serie Temporal', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#2d2d2d')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    @render.ui
    def statistics_cards():
        """Render statistics cards"""
        df = uploaded_dataframe.get()
        state = app_state.get()
        value_col = state.get("value_column")
        
        if df is None or value_col is None:
            return ui.div(
                ui.tags.p("Selecciona una columna de valores primero", class_="text-muted"),
            )
        
        # Convert to numeric if needed
        if pd.api.types.is_numeric_dtype(df[value_col]):
            data = df[value_col].values
        else:
            data = tslib_service.convert_to_numeric(df, value_col).values
        
        # Handle missing values (stats function already handles NaN)
        stats = tslib_service.calculate_basic_stats(data)
        
        return ui.div(
            create_metric_card(f"{stats['mean']:.2f}", "Media", "📊"),
            create_metric_card(f"{stats['std']:.2f}", "Desv. Estándar", "📏"),
            create_metric_card(f"{stats['min']:.2f}", "Mínimo", "⬇️"),
            create_metric_card(f"{stats['max']:.2f}", "Máximo", "⬆️"),
            class_="metrics-grid"
        )
    
    @render.plot
    def acf_plot():
        """Render ACF plot"""
        state = app_state.get()
        analysis = state.get("exploratory_analysis")
        
        if analysis is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'Valida los datos primero', ha='center', va='center', color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        acf_values = analysis.get("acf", [])
        
        # Check if ACF values are empty or invalid
        if not acf_values or len(acf_values) == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'ACF no disponible\n(puede requerir más datos)', 
                   ha='center', va='center', color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        fig, ax = plt.subplots(figsize=(6, 3))
        
        # Get data length from state for confidence intervals
        df = uploaded_dataframe.get()
        value_col = state.get("value_column")
        n_obs = len(df[value_col]) if df is not None and value_col else 100
        
        ax.stem(range(len(acf_values)), acf_values, linefmt='#00d4aa', markerfmt='o', basefmt=' ')
        ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
        
        # Add confidence intervals if we have enough data
        if n_obs > 0:
            conf_level = 1.96 / np.sqrt(n_obs)
            ax.axhline(y=conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title('Autocorrelación (ACF)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#2d2d2d')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    @render.plot
    def pacf_plot():
        """Render PACF plot"""
        state = app_state.get()
        analysis = state.get("exploratory_analysis")
        
        if analysis is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'Valida los datos primero', ha='center', va='center', color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        pacf_values = analysis.get("pacf", [])
        
        # Check if PACF values are empty or invalid
        if not pacf_values or len(pacf_values) == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'PACF no disponible\n(puede requerir más datos)', 
                   ha='center', va='center', color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        fig, ax = plt.subplots(figsize=(6, 3))
        
        # Get data length from state for confidence intervals
        df = uploaded_dataframe.get()
        value_col = state.get("value_column")
        n_obs = len(df[value_col]) if df is not None and value_col else 100
        
        ax.stem(range(len(pacf_values)), pacf_values, linefmt='#0099cc', markerfmt='o', basefmt=' ')
        ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
        
        # Add confidence intervals if we have enough data
        if n_obs > 0:
            conf_level = 1.96 / np.sqrt(n_obs)
            ax.axhline(y=conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('PACF')
        ax.set_title('Autocorrelación Parcial (PACF)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#2d2d2d')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    @render.ui
    def acf_pacf_debug():
        """Placeholder for ACF/PACF debug readout (optional)."""
        return ui.div()

    @render.ui
    def exploration_notes_ui():
        """Short hint; exploratory detail is in plots and statistics (no heuristic text banners)."""
        state = app_state.get()
        validation = state.get("validation_report", {}) or {}
        analysis = state.get("exploratory_analysis", {}) or {}

        if not validation and not analysis:
            return ui.tags.p("Valida los datos para ver notas de exploración.", class_="text-muted")

        return ui.tags.p(
            "Usa la serie, estadísticas y ACF/PACF de esta pantalla para interpretar la estructura.",
            class_="text-muted",
        )

    # Model selection renders
    @render.ui
    def model_description():
        """Show model description based on selection"""
        # Try to get from input first, then fallback to state
        if hasattr(input, 'model_type'):
            try:
                model_type = input.model_type()
                if not model_type:
                    model_type = app_state.get().get("model_type", None)
            except:
                model_type = app_state.get().get("model_type", None)
        else:
            model_type = app_state.get().get("model_type", None)
        
        descriptions = {
            "AR": "Modelo Autoregresivo: El valor actual depende de valores pasados. Útil para series con persistencia.",
            "MA": "Media Móvil: El valor actual depende de errores pasados. Útil para modelar shocks transitorios.",
            "ARMA": "Combinación AR + MA: Para series estacionarias con estructura compleja.",
            "ARIMA": "ARMA Integrado: Incluye diferenciación para series no estacionarias con tendencia."
        }
        
        return ui.div(
            ui.tags.p(descriptions.get(model_type, ""), class_="text-muted"),
            class_="mt-2 model-description"
        )
    
    @render.ui
    def manual_parameters_ui():
        """Show manual parameter inputs based on model type and auto_select"""
        auto_select = input.auto_select() if hasattr(input, 'auto_select') else True
        
        if auto_select:
            return ui.div(
                ui.tags.p("Los parámetros se seleccionarán automáticamente", class_="text-muted"),
                class_="mb-3"
            )
        
        model_type = input.model_type() if hasattr(input, 'model_type') else None
        if not model_type:
            model_type = app_state.get().get("model_type", None)
        if not model_type:
            return ui.div(
                ui.tags.p("Selecciona un tipo de modelo para configurar parámetros.", class_="text-muted"),
                class_="mb-3"
            )
        
        if model_type == "AR":
            return ui.div(
                create_form_group(
                    label="Orden AR (p)",
                    control=ui.input_numeric("p_order", "", value=1, min=0, max=10),
                    help_text="Número de términos autorregresivos"
                ),
                class_="mb-3"
            )
        elif model_type == "MA":
            return ui.div(
                create_form_group(
                    label="Orden MA (q)",
                    control=ui.input_numeric("q_order", "", value=1, min=0, max=10),
                    help_text="Número de términos de media móvil"
                ),
                class_="mb-3"
            )
        elif model_type == "ARMA":
            return ui.div(
                ui.div(
                    create_form_group(
                        label="Orden AR (p)",
                        control=ui.input_numeric("p_order", "", value=1, min=0, max=10),
                        help_text="Términos autorregresivos"
                    ),
                    class_="col-md-6"
                ),
                ui.div(
                    create_form_group(
                        label="Orden MA (q)",
                        control=ui.input_numeric("q_order", "", value=1, min=0, max=10),
                        help_text="Términos de media móvil"
                    ),
                    class_="col-md-6"
                ),
                class_="row mb-3"
            )
        elif model_type == "ARIMA":
            return ui.div(
                ui.div(
                    create_form_group(
                        label="Orden AR (p)",
                        control=ui.input_numeric("p_order", "", value=1, min=0, max=10),
                        help_text="Términos autorregresivos"
                    ),
                    class_="col-md-4"
                ),
                ui.div(
                    create_form_group(
                        label="Diferenciación (d)",
                        control=ui.input_numeric("d_order", "", value=1, min=0, max=3),
                        help_text="Orden de diferenciación"
                    ),
                    class_="col-md-4"
                ),
                ui.div(
                    create_form_group(
                        label="Orden MA (q)",
                        control=ui.input_numeric("q_order", "", value=1, min=0, max=10),
                        help_text="Términos de media móvil"
                    ),
                    class_="col-md-4"
                ),
                class_="row mb-3"
            )
        
        return ui.div()

    # Navigation event handlers
    @reactive.effect
    @reactive.event(input.next_step)
    def handle_next_step():
        """Handle next step button click"""
        current_state = app_state.get()
        current_step = current_state["current_step"]
        
        # Validate current step before proceeding
        if validate_current_step(current_step, current_state):
            if current_step < len(STEPS) - 1:
                new_state = current_state.copy()
                new_state["current_step"] = current_step + 1
                app_state.set(new_state)
                stepper.current_step = current_step + 1
    
    @reactive.effect
    @reactive.event(input.prev_step)
    def handle_prev_step():
        """Handle previous step button click"""
        current_state = app_state.get()
        current_step = current_state["current_step"]
        
        if current_step > 0:
            new_state = current_state.copy()
            new_state["current_step"] = current_step - 1
            app_state.set(new_state)
            stepper.current_step = current_step - 1
    
    # Execution renders
    @render.ui
    def execution_summary():
        """Show execution configuration summary"""
        state = app_state.get()
        model_type = state.get("model_type", "N/A")
        value_col = state.get("value_column", "N/A")
        auto_select = input.auto_select() if hasattr(input, 'auto_select') else True
        exec_meta = state.get("execution_metadata", {}) or {}
        order_text = exec_meta.get("effective_order")
        parallel_backend = exec_meta.get("parallel_backend")
        
        return ui.div(
            ui.tags.p(f"Modelo: {model_type}"),
            ui.tags.p(f"Columna de datos: {value_col}"),
            ui.tags.p(f"Auto-selección: {'Sí' if auto_select else 'No'}"),
            ui.tags.p(f"Orden efectivo: {order_text}") if order_text else None,
            ui.tags.p(f"Backend paralelo (Spark): {parallel_backend}") if parallel_backend else None,
            class_="text-muted"
        )

    @render.ui
    def spark_parallel_status_ui():
        """Show Spark parallel availability before execution."""
        state = app_state.get()
        model_type = state.get("model_type")
        if model_type not in ["AR", "MA", "ARMA", "ARIMA"]:
            return ui.div()
        status = tslib_service.get_spark_parallel_status(model_type=model_type)
        badge_class = "status-success" if status.get("available") else "status-warning"
        prefix = "✓" if status.get("available") else "⚠️"
        return ui.div(
            ui.div(f"{prefix} {status.get('message')}", class_=f"status-indicator {badge_class}"),
            class_="mt-2"
        )
    
    @render.ui
    def execution_status_ui():
        """Show execution status"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return ui.div()
        
        return ui.div(
            ui.div("✓ Análisis completado exitosamente", class_="status-indicator status-success"),
            class_="mt-3"
        )
    
    @render.ui
    def execution_log():
        """Show execution log"""
        state = app_state.get()
        log_entries = state.get("execution_log", [])
        
        if not log_entries:
            return ui.div(
                ui.tags.p("El log aparecerá cuando inicies el análisis", class_="text-muted")
            )
        
        return ui.div(
            *[ui.div(f"• {entry}", class_="progress-step") for entry in log_entries],
            class_="progress-list"
        )
    
    # Results renders
    @render.ui
    def model_info_ui():
        """Show model information"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return ui.div(
                ui.tags.p("Ejecuta el análisis primero", class_="text-muted")
            )
        
        model_type = state.get("model_type", "N/A")
        fitted_model = state.get("fitted_model")
        
        if fitted_model and hasattr(fitted_model, 'order'):
            order = fitted_model.order
            if isinstance(order, tuple):
                order_str = f"{order}"
            else:
                order_str = f"({order})"
        else:
            order_str = "N/A"
        
        return ui.div(
            ui.tags.p(f"Tipo de Modelo: {model_type}"),
            ui.tags.p(f"Orden: {order_str}"),
            class_="text-muted"
        )

    @render.ui
    def runtime_warnings_ui():
        """Show Python / Spark / TSLib warnings captured during execution."""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return ui.div()
        msgs = state.get("runtime_warnings") or []
        if not msgs:
            return ui.div(
                ui.tags.p(
                    "No se registraron avisos del motor en la última ejecución.",
                    class_="text-muted",
                )
            )
        return ui.div(
            ui.tags.p(
                "Mensajes emitidos por librerías (p. ej. estacionariedad, APIs de Spark). "
                "Revisarlos no implica fallo del análisis.",
                class_="text-muted small mb-2",
            ),
            *[
                ui.div(f"⚠ {m}", class_="status-indicator status-warning mb-1")
                for m in msgs
            ],
        )
    
    @render.ui
    def metrics_cards():
        """Render metrics cards (ARIMA + Spark: dual lineal/paralelo en la misma tarjeta y leyenda de color)."""

        def _fmt_float_metric(x):
            if x is None:
                return "N/A"
            try:
                return f"{float(x):.2f}"
            except (TypeError, ValueError):
                return "N/A"

        def _norm_order_par(s):
            if not s or s == "N/A":
                return "N/A"
            t = str(s).strip()
            if t.upper().startswith("ARIMA"):
                rest = t[5:].strip()
                return rest if rest else t
            return t

        state = app_state.get()
        if not state.get("analysis_complete"):
            return ui.div(
                ui.tags.p("Las métricas aparecerán después de ejecutar el análisis", class_="text-muted")
            )

        fitted_model = state.get("fitted_model")
        if not fitted_model:
            return ui.div(ui.tags.p("No hay métricas disponibles", class_="text-muted"))

        metrics_lin = tslib_service.get_model_metrics(fitted_model)
        em = state.get("execution_metadata") or {}
        t_lin = em.get("time_linear_fit_s")
        t_par = em.get("time_parallel_fit_s")
        _mt = state.get("model_type")
        parallel_workflow = state.get("parallel_workflow")

        dual = _mt in ("ARIMA", "AR") and parallel_workflow is not None

        if dual:
            metrics_par = tslib_service.get_parallel_model_metrics(parallel_workflow, model_type=_mt)
            legend = ui.div(
                {"class": "metrics-palette-key mb-2"},
                ui.tags.span("Lineal", class_="metrics-palette-swatch metrics-palette-swatch--lineal"),
                ui.tags.span("Paralelo", class_="metrics-palette-swatch metrics-palette-swatch--paralelo"),
            )
            cards = [
                create_dual_route_metric_card(
                    "AIC",
                    "📊",
                    _fmt_float_metric(metrics_lin.get("aic")),
                    _fmt_float_metric(metrics_par.get("aic")),
                ),
                create_dual_route_metric_card(
                    "BIC",
                    "📊",
                    _fmt_float_metric(metrics_lin.get("bic")),
                    _fmt_float_metric(metrics_par.get("bic")),
                ),
                create_dual_route_metric_card(
                    "Orden",
                    "⚙️",
                    str(metrics_lin.get("order", "N/A")),
                    _norm_order_par(metrics_par.get("order")),
                ),
            ]
            return ui.div(legend, ui.div(*cards, class_="metrics-grid"))

        cards = [
            create_metric_card(
                _fmt_float_metric(metrics_lin.get("aic")),
                "AIC",
                "📊",
            ),
            create_metric_card(
                _fmt_float_metric(metrics_lin.get("bic")),
                "BIC",
                "📊",
            ),
            create_metric_card(
                str(metrics_lin.get("order", "N/A")),
                "Orden",
                "⚙️",
            ),
        ]
        if t_lin is not None:
            cards.append(
                create_metric_card(
                    f"{float(t_lin):.3f}",
                    "Tiempo de ajuste (s)",
                    "⏱",
                )
            )

        return ui.div(*cards, class_="metrics-grid")
    
    @render_plotly
    def execution_timing_plot():
        """Bar chart: linear vs parallel wall time + Spark warm-up + task DF distribution."""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return _plotly_empty_fig("Ejecuta el análisis primero")
        em = state.get("execution_metadata") or {}
        mt = state.get("model_type")
        t_lin = em.get("time_linear_fit_s")
        t_par = em.get("time_parallel_fit_s")
        st = em.get("spark_timing") or {}

        labels: List[str] = []
        values: List[float] = []
        colors: List[str] = []

        if mt in ("ARIMA", "AR"):
            if t_lin is not None:
                labels.append("Lineal (statsmodels, alineado)")
                values.append(float(t_lin))
                colors.append("#e879a8")
            if t_par is not None:
                labels.append("Paralelo (workflow total)")
                values.append(float(t_par))
                colors.append("#2A9D8F")
            wu = st.get("executor_warmup_s")
            dd = st.get("tasks_dataframe_distribute_s")
            if wu is not None:
                labels.append("Spark: warm-up ejecutores")
                values.append(float(wu))
                colors.append("#6c757d")
            if dd is not None:
                labels.append("Spark: distribución DF de tareas")
                values.append(float(dd))
                colors.append("#adb5bd")
        else:
            if t_lin is not None:
                labels.append("Lineal (TSLib)")
                values.append(float(t_lin))
                colors.append("#e879a8")
            if t_par is not None:
                labels.append("Paralelo (Spark)")
                values.append(float(t_par))
                colors.append("#2A9D8F")

        if not values:
            return _plotly_empty_fig("Sin tiempos registrados")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.4f} s" for v in values],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#2d2d2d",
            font=dict(color="#ececec", size=11),
            title=dict(text="Tiempos de ejecución (segundos)", x=0.5),
            yaxis=dict(title="s", gridcolor="#444444", rangemode="tozero"),
            xaxis=dict(tickangle=-28),
            margin=dict(l=50, r=24, t=56, b=140),
            dragmode="zoom",
            showlegend=False,
        )
        return fig
    
    @render_plotly
    def forecast_plot():
        """Interactive forecast plot (Plotly: zoom, pan)."""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return _plotly_empty_fig("Ejecuta el análisis primero")

        df = uploaded_dataframe.get()
        value_col = state.get("value_column")
        forecast_results = state.get("forecast_results")
        parallel_forecast_results = state.get("parallel_forecast_results")
        mt = state.get("model_type")

        if (
            mt in ("ARIMA", "AR")
            and forecast_results
            and parallel_forecast_results
            and _has_forecast_values(forecast_results)
            and _has_forecast_values(parallel_forecast_results)
        ):
            return _figure_arima_lineal_vs_paralelo_plotly()

        if not forecast_results or df is None:
            return _plotly_empty_fig("No hay pronóstico disponible")

        if pd.api.types.is_numeric_dtype(df[value_col]):
            historical = df[value_col].values
        else:
            historical = tslib_service.convert_to_numeric(df, value_col).values
        historical = np.asarray(historical, dtype=float)

        forecast = np.asarray(forecast_results.get("forecast", []), dtype=float).ravel()
        lower = forecast_results.get("lower_bound")
        upper = forecast_results.get("upper_bound")
        n_hist = len(historical)
        n_fore = len(forecast)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(n_hist)),
                y=historical.tolist(),
                mode="lines",
                name="Histórico",
                line=dict(color="#00d4aa", width=1.5),
            )
        )
        fx = list(range(n_hist, n_hist + n_fore))
        lo = np.asarray(lower, dtype=float).ravel() if lower is not None else None
        up = np.asarray(upper, dtype=float).ravel() if upper is not None else None
        if lo is not None and up is not None and len(lo) >= n_fore and len(up) >= n_fore:
            fig.add_trace(
                go.Scatter(
                    x=fx + fx[::-1],
                    y=lo[:n_fore].tolist() + up[:n_fore][::-1].tolist(),
                    fill="toself",
                    fillcolor="rgba(0,153,204,0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="IC 95%",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=forecast.tolist(),
                mode="lines",
                name="Pronóstico",
                line=dict(color="#0099cc", width=1.5, dash="dash"),
            )
        )
        return _plotly_forecast_layout(fig, "Pronóstico de serie temporal")
    
    @render.ui
    def forecast_table_ui():
        """Render forecast table"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return ui.div(ui.tags.p("Ejecuta el análisis primero", class_="text-muted"))
        
        forecast_results = state.get("forecast_results")
        if not forecast_results:
            return ui.div(ui.tags.p("No hay pronóstico disponible", class_="text-muted"))
        
        forecast = forecast_results.get('forecast', [])
        lower = forecast_results.get('lower_bound')
        upper = forecast_results.get('upper_bound')
        pr = state.get("parallel_forecast_results") or {}
        pfc = pr.get("forecast", [])

        if (
            state.get("model_type") in ("ARIMA", "AR")
            and _forecast_seq_len(pfc) > 0
            and _forecast_seq_len(forecast) > 0
        ):
            table_data = []
            n = min(len(forecast), len(pfc))
            for i in range(n):
                vl = float(forecast[i])
                vp = float(pfc[i])
                pct = _pct_abs_parallel_vs_linear(vl, vp)
                row = [
                    f"t+{i+1}",
                    f"{vl:.4f}",
                    f"{vp:.4f}",
                    f"{abs(vl - vp):.4f}",
                    f"{pct:.4f}",
                ]
                table_data.append(row)
            headers = ["Paso", "Lineal (statsmodels)", "Paralelo (Spark)", "|L−P|", "% |P−L|/|L|"]
            return create_data_table(table_data, headers=headers)

        table_data = []
        for i, val in enumerate(forecast, 1):
            row = [f"t+{i}", f"{val:.4f}"]
            if lower is not None and upper is not None:
                row.extend([f"{lower[i-1]:.4f}", f"{upper[i-1]:.4f}"])
            table_data.append(row)
        
        headers = ["Paso", "Pronóstico"]
        if lower is not None:
            headers.extend(["Límite Inferior", "Límite Superior"])
        
        return create_data_table(table_data, headers=headers)

    @render_plotly
    def forecast_diff_horizon_plot():
        """|lineal − paralelo| por horizonte (solo ARIMA con ambas rutas), interactivo."""
        state = app_state.get()
        if not state.get("analysis_complete") or state.get("model_type") not in ("ARIMA", "AR"):
            return _plotly_empty_fig("Solo aplica con ARIMA/AR y ambas rutas")
        fr = state.get("forecast_results") or {}
        pr = state.get("parallel_forecast_results") or {}
        a = _as_forecast_array(fr)
        b = _as_forecast_array(pr)
        n = min(len(a), len(b))
        if n == 0:
            return _plotly_empty_fig("Sin datos comparables")
        d = np.abs(a[:n] - b[:n])
        x = np.arange(1, n + 1)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=x,
                    y=d,
                    marker=dict(color=_COL_ARIMA_LINEAL, line=dict(color="#444")),
                    name="|L−P|",
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#2d2d2d",
            font=dict(color="#ececec"),
            title=dict(text="Diferencia entre pronósticos por paso", x=0.5),
            xaxis=dict(title="Horizonte (t+h)", gridcolor="#444"),
            yaxis=dict(title="|lineal − paralelo|", gridcolor="#444"),
            showlegend=False,
            dragmode="zoom",
            margin=dict(l=50, r=20, t=50, b=40),
        )
        return fig
    
    @render_plotly
    def residuals_plot():
        """Residuals vs time (interactive)."""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return _plotly_empty_fig("Ejecuta el análisis primero")
        fitted_model = state.get("fitted_model")
        if not fitted_model or not hasattr(fitted_model, "get_residuals"):
            return _plotly_empty_fig("No hay residuos disponibles")
        residuals = fitted_model.get_residuals()
        return plot_residuals_plotly(np.asarray(residuals, dtype=float))
    
    @render_plotly
    def residuals_acf_plot():
        """ACF of residuals (interactive)."""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return _plotly_empty_fig("Ejecuta el análisis primero")
        fitted_model = state.get("fitted_model")
        if not fitted_model or not hasattr(fitted_model, "get_residuals"):
            return _plotly_empty_fig("No hay residuos disponibles")
        residuals = fitted_model.get_residuals()
        try:
            return plot_residual_acf_plotly(np.asarray(residuals, dtype=float))
        except Exception as e:
            return _plotly_empty_fig(f"Error ACF: {e}")

    @render_plotly
    def eval_residual_hist_qq_plot():
        state = app_state.get()
        if not state.get("analysis_complete"):
            return _plotly_empty_fig("Ejecuta el análisis primero")
        fitted_model = state.get("fitted_model")
        if not fitted_model or not hasattr(fitted_model, "get_residuals"):
            return _plotly_empty_fig("Sin residuos")
        r = fitted_model.get_residuals()
        return plot_residual_hist_qq_plotly(np.asarray(r, dtype=float))

    @render_plotly
    def eval_standardized_residuals_plot():
        state = app_state.get()
        if not state.get("analysis_complete"):
            return _plotly_empty_fig("Ejecuta el análisis primero")
        fitted_model = state.get("fitted_model")
        if not fitted_model or not hasattr(fitted_model, "get_residuals"):
            return _plotly_empty_fig("Sin residuos")
        r = np.asarray(fitted_model.get_residuals(), dtype=float)
        return plot_standardized_residuals_plotly(r)

    @render.ui
    def linear_model_title():
        """Show linear model title (statsmodels for ARIMA, TSLib for AR/MA/ARMA)."""
        mt = app_state.get().get("model_type")
        label = (
            "Modelo lineal (statsmodels)"
            if mt in ("ARIMA", "AR")
            else "Modelo lineal (TSLib)"
        )
        return ui.tags.h4(label, class_="mb-3")
    
    @render.ui
    def parallel_model_section():
        """Render parallel model section"""
        state = app_state.get()
        model_type = state.get("model_type")
        parallel_workflow = state.get("parallel_workflow")
        
        # Check if parallel model was executed
        if parallel_workflow is None:
            state = app_state.get()
            execution_log = state.get("execution_log", [])
            # Find error messages related to parallel model
            error_messages = [log for log in execution_log if "Error" in log or "error" in log or "⚠" in log]
            
            error_text = "El modelo paralelo no está disponible. Puede que haya ocurrido un error durante la ejecución."
            if error_messages:
                error_text += f"\n\nÚltimos mensajes de error:\n" + "\n".join(error_messages[-3:])  # Show last 3 error messages
            
            return ui.div(
                ui.tags.p(error_text, class_="text-muted"),
                ui.tags.p("Revisa los logs en la consola para más detalles.", class_="text-muted", style="font-size: 0.9em;"),
                class_="mb-4"
            )
        
        return ui.div(
            # Parallel model info
            ui.div(
                ui.tags.h5("Información del modelo paralelo:"),
                ui.output_ui("parallel_model_info_ui"),
                class_="mb-4"
            ),
            # Forecast distance metrics (L vs P); AIC/BIC/orden están en «Métricas de evaluación»
            ui.div(
                ui.tags.h5("Comparación de pronósticos (lineal vs paralelo):"),
                ui.output_ui("parallel_metrics_cards"),
                class_="mb-4"
            ),
            ui.div(
                ui.tags.h5("Pronóstico paralelo (Spark)"),
                ui.tags.p(
                    "Con ARIMA y ambas rutas, el gráfico coincide con «Pronóstico» arriba (verde = lineal, rosa = paralelo).",
                    class_="text-muted small mb-2",
                ),
                output_widget("parallel_forecast_plot", height="420px"),
                class_="mb-4"
            ),
            ui.div(
                ui.tags.h5("Detalle tabla paralela"),
                ui.output_ui("parallel_forecast_table_ui"),
                class_="mb-4"
            )
        )
    
    @render.ui
    def parallel_model_info_ui():
        """Show parallel model information"""
        state = app_state.get()
        parallel_workflow = state.get("parallel_workflow")
        model_type = state.get("model_type", "N/A")
        
        if not parallel_workflow:
            return ui.div(ui.tags.p("No hay información disponible", class_="text-muted"))

        # Get metrics
        metrics = tslib_service.get_parallel_model_metrics(parallel_workflow, model_type=model_type)
        order = metrics.get('order', 'N/A')
        backend = metrics.get("backend", "N/A")
        backend_desc = "Spark distribuido." if backend == "spark" else "N/A"
        lines = [
            ui.tags.p(f"Tipo de modelo: {model_type} (paralelo)"),
            ui.tags.p(f"Orden: {order}"),
            ui.tags.p(f"Backend: {backend} — {backend_desc}"),
            ui.tags.p(
                "Tiempos detallados (lineal, paralelo, warm-up Spark, distribución DF): gráfico de barras arriba.",
                class_="small",
            ),
        ]

        return ui.div(*lines, class_="text-muted")
    
    @render.ui
    def parallel_metrics_cards():
        """ARIMA: solo distancias entre pronósticos L vs P (AIC/BIC/orden van en métricas principales)."""
        state = app_state.get()
        parallel_workflow = state.get("parallel_workflow")
        model_type = state.get("model_type", "N/A")
        
        if not parallel_workflow:
            return ui.div(ui.tags.p("No hay métricas disponibles", class_="text-muted"))
        
        metrics = tslib_service.get_parallel_model_metrics(parallel_workflow, model_type=model_type)
        em = state.get("execution_metadata") or {}
        comparison = em.get("forecast_comparison") or {}

        cards = []

        if model_type in ("ARIMA", "AR"):
            if comparison:
                cards.append(
                    create_metric_card(
                        f"{comparison.get('mae_diff', 0):.4f}",
                        "MAE |L−P|",
                        "↔️",
                    )
                )
                cards.append(
                    create_metric_card(
                        f"{comparison.get('rmse_diff', 0):.4f}",
                        "RMSE (L vs P)",
                        "↔️",
                    )
                )
                if comparison.get("mape_diff") is not None:
                    cards.append(
                        create_metric_card(
                            f"{comparison.get('mape_diff', 0):.2f}",
                            "MAPE* dif.",
                            "↔️",
                        )
                    )
            if not cards:
                return ui.div(
                    ui.tags.p(
                        "No hay métricas de comparación hasta que existan ambos pronósticos.",
                        class_="text-muted",
                    )
                )
            return ui.div(*cards, class_="metrics-grid")

        if metrics.get("order"):
            cards.append(
                create_metric_card(
                    metrics.get("order", "N/A"),
                    "Orden",
                    "⚙️",
                )
            )
        if metrics.get("aic") is not None:
            cards.append(
                create_metric_card(
                    f"{float(metrics['aic']):.2f}",
                    "AIC",
                    "📊",
                )
            )
        if metrics.get("bic") is not None:
            cards.append(
                create_metric_card(
                    f"{float(metrics['bic']):.2f}",
                    "BIC",
                    "📊",
                )
            )
        if metrics.get("mae") is not None:
            cards.append(
                create_metric_card(
                    f"{float(metrics['mae']):.4f}",
                    "MAE",
                    "📊",
                )
            )
        if metrics.get("rmse") is not None:
            cards.append(
                create_metric_card(
                    f"{float(metrics['rmse']):.4f}",
                    "RMSE",
                    "📊",
                )
            )
        if metrics.get("mape") is not None:
            cards.append(
                create_metric_card(
                    f"{float(metrics['mape']):.2f}",
                    "MAPE",
                    "📊",
                )
            )

        if not cards:
            return ui.div(ui.tags.p("No hay métricas disponibles", class_="text-muted"))

        return ui.div(*cards, class_="metrics-grid")
    
    @render_plotly
    def parallel_forecast_plot():
        """Misma vista que el pronóstico principal cuando hay lineal + paralelo (ARIMA), Plotly."""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return _plotly_empty_fig("Ejecuta el análisis primero")

        if (
            state.get("model_type") in ("ARIMA", "AR")
            and state.get("forecast_results")
            and state.get("parallel_forecast_results")
            and _has_forecast_values(state.get("forecast_results"))
            and _has_forecast_values(state.get("parallel_forecast_results"))
        ):
            return _figure_arima_lineal_vs_paralelo_plotly()

        df = uploaded_dataframe.get()
        value_col = state.get("value_column")
        parallel_forecast_results = state.get("parallel_forecast_results")

        if not parallel_forecast_results or df is None:
            return _plotly_empty_fig("No hay pronóstico paralelo disponible")

        if pd.api.types.is_numeric_dtype(df[value_col]):
            historical = df[value_col].values
        else:
            historical = tslib_service.convert_to_numeric(df, value_col).values
        if np.any(np.isnan(historical)):
            mask = np.isnan(historical)
            indices = np.arange(len(historical))
            if np.any(~mask):
                historical[mask] = np.interp(indices[mask], indices[~mask], historical[~mask])
            else:
                historical = np.zeros_like(historical)

        forecast = np.asarray(parallel_forecast_results.get("forecast", []), dtype=float).ravel()
        lower = parallel_forecast_results.get("lower_bound")
        upper = parallel_forecast_results.get("upper_bound")
        n_hist = len(historical)
        n_f = len(forecast)
        fx = list(range(n_hist, n_hist + n_f))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(n_hist)),
                y=np.asarray(historical, dtype=float).tolist(),
                mode="lines",
                name="Histórico",
                line=dict(color="#00d4aa", width=1.5),
            )
        )
        lo = np.asarray(lower, dtype=float).ravel() if lower is not None else None
        up = np.asarray(upper, dtype=float).ravel() if upper is not None else None
        if lo is not None and up is not None and len(lo) >= n_f and len(up) >= n_f:
            fig.add_trace(
                go.Scatter(
                    x=fx + fx[::-1],
                    y=lo[:n_f].tolist() + up[:n_f][::-1].tolist(),
                    fill="toself",
                    fillcolor="rgba(42,157,143,0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="IC 95%",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=forecast.tolist(),
                mode="lines",
                name="Pronóstico paralelo",
                line=dict(color=_COL_ARIMA_PARALELO, width=1.5, dash="dash"),
            )
        )
        return _plotly_forecast_layout(fig, "Pronóstico — ruta paralela")
    
    @render.ui
    def parallel_forecast_table_ui():
        """Render parallel forecast table"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return ui.div(ui.tags.p("Ejecuta el análisis primero", class_="text-muted"))

        if (
            state.get("model_type") in ("ARIMA", "AR")
            and _has_forecast_values(state.get("forecast_results"))
            and _has_forecast_values(state.get("parallel_forecast_results"))
        ):
            return ui.div(
                ui.tags.p(
                    "La tabla comparativa lineal / paralelo / |L−P| está en la sección «Valores del pronóstico» arriba.",
                    class_="text-muted small",
                )
            )
        
        parallel_forecast_results = state.get("parallel_forecast_results")
        if not parallel_forecast_results:
            return ui.div(ui.tags.p("No hay pronóstico paralelo disponible", class_="text-muted"))
        
        forecast = parallel_forecast_results.get('forecast', [])
        lower = parallel_forecast_results.get('lower_bound')
        upper = parallel_forecast_results.get('upper_bound')
        
        # Create table data
        table_data = []
        for i, val in enumerate(forecast, 1):
            row = [f"t+{i}", f"{val:.4f}"]
            if lower is not None and upper is not None:
                row.extend([f"{lower[i-1]:.4f}", f"{upper[i-1]:.4f}"])
            table_data.append(row)
        
        headers = ["Paso", "Pronóstico"]
        if lower is not None:
            headers.extend(["Límite Inferior", "Límite Superior"])
        
        return create_data_table(table_data, headers=headers)
    
    # File upload handler
    @reactive.effect
    @reactive.event(input.file_upload)
    def handle_file_upload():
        """Handle file upload"""
        file_info = input.file_upload()
        if not file_info:
            uploaded_dataframe.set(None)
            new_state = app_state.get().copy()
            new_state["data_loaded"] = False
            new_state["uploaded_data"] = None
            app_state.set(new_state)
            return
        
        file_metadata = file_info[0]
        temp_path = file_metadata.get("datapath")
        file_name = file_metadata.get("name", "dataset")
        file_size = file_metadata.get("size")

        if file_size is not None and file_size > MAX_UPLOAD_FILE_BYTES:
            uploaded_dataframe.set(None)
            new_state = app_state.get().copy()
            new_state["data_loaded"] = False
            new_state["uploaded_data"] = None
            app_state.set(new_state)
            limit_mb = MAX_UPLOAD_FILE_BYTES // (1024 * 1024)
            ui.notification_show(
                f"El archivo supera el límite de {limit_mb} MB. Reduce el tamaño o divide los datos.",
                type="error",
                duration=8,
            )
            return

        try:
            if file_name.lower().endswith(".csv"):
                df = pd.read_csv(temp_path)
            elif file_name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(temp_path)
            else:
                raise ValueError("Formato de archivo no soportado")
        except Exception as exc:
            uploaded_dataframe.set(None)
            new_state = app_state.get().copy()
            new_state["data_loaded"] = False
            new_state["uploaded_data"] = None
            app_state.set(new_state)
            ui.notification_show(
                f"No fue posible cargar el archivo: {exc}",
                type="error",
                duration=5
            )
            return
        
        uploaded_dataframe.set(df)
        
        new_state = app_state.get().copy()
        new_state["data_loaded"] = not df.empty
        new_state["uploaded_data"] = {
            "filename": file_name,
            "size": file_size,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        }
        app_state.set(new_state)
    
    # Column selection handler
    @reactive.effect
    @reactive.event(input.value_column)
    def handle_value_column_change():
        """Handle value column selection"""
        if not hasattr(input, 'value_column'):
            return
        
        value_col = input.value_column()
        new_state = app_state.get().copy()
        new_state["value_column"] = value_col
        new_state["data_validated"] = False  # Reset validation
        app_state.set(new_state)
    
    @reactive.effect
    @reactive.event(input.date_column)
    def handle_date_column_change():
        """Handle date column selection"""
        if not hasattr(input, 'date_column'):
            return
        
        date_col = input.date_column()
        new_state = app_state.get().copy()
        new_state["date_column"] = date_col if date_col != "(Ninguna)" else None
        app_state.set(new_state)
    
    # Data validation handler
    @reactive.effect
    @reactive.event(input.validate_data)
    def handle_validate_data():
        """Handle data validation"""
        df = uploaded_dataframe.get()
        state = app_state.get()
        value_col = state.get("value_column")
        
        if df is None or value_col is None:
            ui.notification_show(
                "Selecciona una columna de valores primero",
                type="warning",
                duration=3
            )
            return
        
        try:
            validation_result, w_val = run_with_recorded_warnings(
                lambda: tslib_service.validate_data(df, value_col)
            )

            exploratory = None
            w_exp: list[str] = []
            if validation_result.get("valid"):
                if pd.api.types.is_numeric_dtype(df[value_col]):
                    data = df[value_col].values
                else:
                    data = tslib_service.convert_to_numeric(df, value_col).values

                exploratory, w_exp = run_with_recorded_warnings(
                    lambda: tslib_service.get_exploratory_analysis(
                        data,
                        validation_report=validation_result.get("quality_report", {}),
                    )
                )
            rw = list(dict.fromkeys(w_val + w_exp))
            validation_result = {**validation_result, "runtime_warnings": rw}

            new_state = state.copy()
            new_state["data_validated"] = validation_result["valid"]
            new_state["validation_report"] = validation_result
            new_state["exploratory_analysis"] = exploratory
            app_state.set(new_state)

        except Exception as e:
            ui.notification_show(
                f"Error en validación: {str(e)}",
                type="error",
                duration=5
            )
    
    # Model type change handler
    @reactive.effect
    @reactive.event(input.model_type)
    def handle_model_type_change():
        """Handle model type change and update state"""
        if not hasattr(input, 'model_type'):
            return
        
        try:
            model_type = input.model_type()
            # Normalize empty selection to None
            if model_type is not None and model_type not in ["", "__none__"]:
                new_state = app_state.get().copy()
                new_state["model_type"] = model_type
                # Clear parallel model results when model type changes
                new_state["parallel_workflow"] = None
                new_state["parallel_forecast_results"] = None
                app_state.set(new_state)
            else:
                # Clear model_type if user selects placeholder
                new_state = app_state.get().copy()
                new_state["model_type"] = None
                new_state["parallel_workflow"] = None
                new_state["parallel_forecast_results"] = None
                app_state.set(new_state)
        except Exception as e:
            # If there's an error getting the value, don't update state
            pass
    
    @reactive.effect
    @reactive.event(input.auto_select)
    def handle_auto_select_change():
        """Handle auto_select switch change and persist in state"""
        if not hasattr(input, 'auto_select'):
            return
        
        try:
            auto_select_value = input.auto_select()
            new_state = app_state.get().copy()
            new_state["auto_select"] = auto_select_value
            app_state.set(new_state)
        except Exception as e:
            # If there's an error getting the value, don't update state
            pass

    # Model execution handler
    @reactive.effect
    @reactive.event(input.start_execution)
    def handle_start_execution():
        """Handle model execution start"""
        df = uploaded_dataframe.get()
        state = app_state.get()
        value_col = state.get("value_column")
        model_type = state.get("model_type", "ARIMA")
        
        if df is None or value_col is None:
            ui.notification_show(
                "No hay datos cargados",
                type="error",
                duration=3
            )
            return
        
        if not state.get("data_validated"):
            ui.notification_show(
                "Valida los datos primero",
                type="warning",
                duration=3
            )
            return
        
        try:
            # Get data and convert to numeric if needed
            if pd.api.types.is_numeric_dtype(df[value_col]):
                data = df[value_col].values
            else:
                data = tslib_service.convert_to_numeric(df, value_col).values

            quality_report = (state.get("validation_report", {}) or {}).get("quality_report", {})
            stationarity = tslib_service.get_stationarity_guidance(quality_report)
            if model_type in ["AR", "MA", "ARMA"] and stationarity.get("recommended_stationary_only_block"):
                msg = (
                    "La serie muestra señal de no estacionariedad (tendencia). "
                    f"Para {model_type} usa primero diferenciación o selecciona ARIMA."
                )
                new_state = app_state.get().copy()
                new_state["execution_log"].append(f"⚠ {msg}")
                app_state.set(new_state)
                ui.notification_show(msg, type="warning", duration=6)
                return
            
            # Get model parameters
            auto_select = input.auto_select() if hasattr(input, 'auto_select') else True
            forecast_steps = input.forecast_steps() if hasattr(input, 'forecast_steps') else 10
            include_conf = input.include_confidence() if hasattr(input, 'include_confidence') else True
            
            # Determine order based on model type and auto_select
            if auto_select:
                order = None  # Will be auto-selected
            else:
                if model_type == "AR":
                    p = input.p_order() if hasattr(input, 'p_order') else 1
                    order = (p,)
                elif model_type == "MA":
                    q = input.q_order() if hasattr(input, 'q_order') else 1
                    order = (q,)
                elif model_type == "ARMA":
                    p = input.p_order() if hasattr(input, 'p_order') else 1
                    q = input.q_order() if hasattr(input, 'q_order') else 1
                    order = (p, q)
                elif model_type == "ARIMA":
                    p = input.p_order() if hasattr(input, 'p_order') else 1
                    d = input.d_order() if hasattr(input, 'd_order') else 1
                    q = input.q_order() if hasattr(input, 'q_order') else 1
                    order = (p, d, q)
            
            runtime_msgs: list[str] = []
            new_state = state.copy()
            exec_log = [
                "Iniciando análisis...",
                f"Modelo seleccionado: {model_type}",
            ]
            if model_type == "ARIMA":
                exec_log.append(
                    "Secuencia ARIMA: paralelo Spark (workflow 11 pasos) → lineal statsmodels (mismo orden cuando Spark OK)."
                )
            elif model_type == "AR":
                exec_log.append(
                    "Secuencia AR: paralelo Spark (ParallelARWorkflow) → lineal statsmodels ARIMA(p,0,0) alineado."
                )
            else:
                exec_log.append("Ajustando modelo...")
            new_state["execution_log"] = exec_log
            new_state["runtime_warnings"] = []
            app_state.set(new_state)

            def _spark_placeholder_order(mt: str, ord_: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
                if ord_ is not None:
                    return ord_
                if mt in ["AR", "MA"]:
                    return (1,)
                if mt == "ARMA":
                    return (1, 1)
                return (1, 1, 1)

            placeholder_order = _spark_placeholder_order(model_type, order)

            if model_type == "ARIMA":
                parallel_workflow = None
                parallel_forecast_results = None
                time_parallel_fit_s = None
                new_state = app_state.get().copy()
                new_state["execution_log"].append(
                    "Ajustando ARIMA paralelo (Spark, ParallelARIMAWorkflow, 11 pasos)..."
                )
                app_state.set(new_state)
                try:
                    t_par0 = time.perf_counter()
                    parallel_workflow, parallel_forecast_results = tslib_service.fit_parallel_model_spark(
                        data=data,
                        model_type=model_type,
                        order=placeholder_order,
                        steps=forecast_steps,
                        validation_report=quality_report,
                    )
                    time_parallel_fit_s = float(time.perf_counter() - t_par0)
                    new_state = app_state.get().copy()
                    backend = getattr(parallel_workflow, "backend_", "desconocido")
                    new_state["execution_log"].append(f"✓ ARIMA paralelo completado (backend: {backend})")
                    app_state.set(new_state)
                except Exception as e:
                    error_msg = f"Ruta paralela Spark (ARIMA): {type(e).__name__}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    new_state = app_state.get().copy()
                    new_state["execution_log"].append(f"⚠ {error_msg}")
                    app_state.set(new_state)

                if parallel_forecast_results:
                    runtime_msgs.extend(
                        parallel_forecast_results.get("parallel_runtime_warnings") or []
                    )

                if parallel_workflow is not None and getattr(parallel_workflow, "order_", None) is not None:
                    sm_order = tuple(int(x) for x in parallel_workflow.order_)
                    new_state = app_state.get().copy()
                    new_state["execution_log"].append(
                        f"Orden statsmodels alineado con workflow paralelo: ARIMA{sm_order}"
                    )
                    app_state.set(new_state)
                elif order is not None and len(order) == 3:
                    sm_order = tuple(int(x) for x in order)
                else:
                    sm_order = (1, 1, 1)
                    runtime_msgs.append(
                        "ARIMA lineal: orden (1,1,1) por defecto (sin orden del workflow ni manual completo)."
                    )

                new_state = app_state.get().copy()
                new_state["execution_log"].append("Ajustando ARIMA lineal (statsmodels)...")
                app_state.set(new_state)

                t_lin0 = time.perf_counter()

                def _fit_linear_statsmodels():
                    if parallel_workflow is not None and getattr(
                        parallel_workflow, "working_data_", None
                    ) is not None:
                        return tslib_service.fit_statsmodels_arima_aligned_to_workflow(
                            parallel_workflow
                        )
                    return tslib_service.fit_statsmodels_arima(
                        np.asarray(data, dtype=float), sm_order
                    )

                fitted_model, w_fit = run_with_recorded_warnings(_fit_linear_statsmodels)
                time_linear_fit_s = float(time.perf_counter() - t_lin0)
                time_statsmodels_fit_s = time_linear_fit_s
                runtime_msgs.extend(w_fit)

                effective_order = str(fitted_model.order)
                new_state = app_state.get().copy()
                new_state["execution_log"].append(f"Orden efectivo lineal (statsmodels): {effective_order}")
                app_state.set(new_state)

                new_state = app_state.get().copy()
                new_state["execution_log"].append("Generando pronóstico (statsmodels)...")
                app_state.set(new_state)

                forecast_results, w_fc = run_with_recorded_warnings(
                    lambda: tslib_service.get_forecast(
                        model=fitted_model,
                        steps=forecast_steps,
                        return_conf_int=include_conf,
                    )
                )
                runtime_msgs.extend(w_fc)

            elif model_type == "AR":
                parallel_workflow = None
                parallel_forecast_results = None
                time_parallel_fit_s = None
                new_state = app_state.get().copy()
                new_state["execution_log"].append(
                    "Ajustando AR paralelo (Spark, ParallelARWorkflow)..."
                )
                app_state.set(new_state)
                try:
                    t_par0 = time.perf_counter()
                    parallel_workflow, parallel_forecast_results = tslib_service.fit_parallel_model_spark(
                        data=data,
                        model_type=model_type,
                        order=placeholder_order,
                        steps=forecast_steps,
                        validation_report=quality_report,
                    )
                    time_parallel_fit_s = float(time.perf_counter() - t_par0)
                    new_state = app_state.get().copy()
                    backend = getattr(parallel_workflow, "backend_", "desconocido")
                    new_state["execution_log"].append(f"✓ AR paralelo completado (backend: {backend})")
                    app_state.set(new_state)
                except Exception as e:
                    error_msg = f"Ruta paralela Spark (AR): {type(e).__name__}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    new_state = app_state.get().copy()
                    new_state["execution_log"].append(f"⚠ {error_msg}")
                    app_state.set(new_state)

                if parallel_forecast_results:
                    runtime_msgs.extend(
                        parallel_forecast_results.get("parallel_runtime_warnings") or []
                    )

                if parallel_workflow is not None and getattr(parallel_workflow, "order_", None) is not None:
                    po = parallel_workflow.order_
                    sm_order = (int(po[0]), 0, 0)
                    new_state = app_state.get().copy()
                    new_state["execution_log"].append(
                        f"Orden statsmodels alineado: ARIMA{sm_order} (AR clásico sobre serie de trabajo)"
                    )
                    app_state.set(new_state)
                elif order is not None and len(order) >= 1:
                    sm_order = (int(order[0]), 0, 0)
                else:
                    sm_order = (1, 0, 0)
                    runtime_msgs.append(
                        "AR lineal: orden (1,0,0) por defecto (sin orden del workflow ni p manual)."
                    )

                new_state = app_state.get().copy()
                new_state["execution_log"].append("Ajustando AR lineal (statsmodels, ARIMA(p,0,0))...")
                app_state.set(new_state)

                t_lin0 = time.perf_counter()

                def _fit_linear_ar_statsmodels():
                    if parallel_workflow is not None and getattr(
                        parallel_workflow, "working_data_", None
                    ) is not None:
                        return tslib_service.fit_statsmodels_ar_aligned_to_parallel_ar_workflow(
                            parallel_workflow
                        )
                    return tslib_service.fit_statsmodels_arima(
                        np.asarray(data, dtype=float), sm_order
                    )

                fitted_model, w_fit = run_with_recorded_warnings(_fit_linear_ar_statsmodels)
                time_linear_fit_s = float(time.perf_counter() - t_lin0)
                time_statsmodels_fit_s = time_linear_fit_s
                runtime_msgs.extend(w_fit)

                effective_order = str(fitted_model.order)
                new_state = app_state.get().copy()
                new_state["execution_log"].append(f"Orden efectivo lineal (statsmodels): {effective_order}")
                app_state.set(new_state)

                new_state = app_state.get().copy()
                new_state["execution_log"].append("Generando pronóstico (statsmodels)...")
                app_state.set(new_state)

                forecast_results, w_fc = run_with_recorded_warnings(
                    lambda: tslib_service.get_forecast(
                        model=fitted_model,
                        steps=forecast_steps,
                        return_conf_int=include_conf,
                    )
                )
                runtime_msgs.extend(w_fc)

            else:
                t_lin0 = time.perf_counter()
                fitted_model, w_fit = run_with_recorded_warnings(
                    lambda: tslib_service.fit_model(
                        data=data,
                        model_type=model_type,
                        order=placeholder_order,
                        auto_select=auto_select,
                        validation_report=quality_report,
                    )
                )
                time_linear_fit_s = float(time.perf_counter() - t_lin0)
                runtime_msgs.extend(w_fit)

                time_statsmodels_fit_s = None

                effective_order = None
                if hasattr(fitted_model, "order"):
                    effective_order = str(fitted_model.order)
                    new_state = app_state.get().copy()
                    new_state["execution_log"].append(f"Orden efectivo ajustado: {effective_order}")
                    app_state.set(new_state)

                new_state = app_state.get().copy()
                new_state["execution_log"].append("Generando pronóstico (modelo lineal TSLib)...")
                app_state.set(new_state)

                forecast_results, w_fc = run_with_recorded_warnings(
                    lambda: tslib_service.get_forecast(
                        model=fitted_model,
                        steps=forecast_steps,
                        return_conf_int=include_conf,
                    )
                )
                runtime_msgs.extend(w_fc)

                parallel_workflow = None
                parallel_forecast_results = None
                new_state = app_state.get().copy()
                new_state["execution_log"].append(f"Ajustando modelo paralelo {model_type} (Spark)...")
                app_state.set(new_state)
                time_parallel_fit_s = None
                try:
                    t_par0 = time.perf_counter()
                    parallel_workflow, parallel_forecast_results = tslib_service.fit_parallel_model_spark(
                        data=data,
                        model_type=model_type,
                        order=placeholder_order,
                        steps=forecast_steps,
                        validation_report=quality_report,
                    )
                    time_parallel_fit_s = float(time.perf_counter() - t_par0)
                    new_state = app_state.get().copy()
                    backend = getattr(parallel_workflow, "backend_", "desconocido")
                    new_state["execution_log"].append(f"✓ Modelo paralelo completado (backend: {backend})")
                    app_state.set(new_state)
                except Exception as e:
                    error_msg = f"Ruta paralela Spark no disponible: {type(e).__name__}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    new_state = app_state.get().copy()
                    new_state["execution_log"].append(f"⚠ {error_msg}")
                    app_state.set(new_state)

                if parallel_forecast_results:
                    runtime_msgs.extend(
                        parallel_forecast_results.get("parallel_runtime_warnings") or []
                    )
            runtime_msgs = list(dict.fromkeys(runtime_msgs))

            # Update state with results
            new_state = app_state.get().copy()
            new_state["fitted_model"] = fitted_model
            new_state["forecast_results"] = forecast_results
            new_state["parallel_workflow"] = parallel_workflow
            new_state["parallel_forecast_results"] = parallel_forecast_results
            new_state["runtime_warnings"] = runtime_msgs
            linear_fc = forecast_results.get("forecast", []) if forecast_results else []
            parallel_fc = parallel_forecast_results.get("forecast", []) if parallel_forecast_results else []
            forecast_comparison = tslib_service.compare_forecasts(linear_fc, parallel_fc)
            spark_timing_meta: Dict[str, Any] = {}
            if parallel_workflow is not None:
                spark_timing_meta = tslib_service.get_workflow_spark_timing(parallel_workflow)
            new_state["execution_metadata"] = {
                "effective_order": effective_order,
                "parallel_backend": getattr(parallel_workflow, "backend_", None) if parallel_workflow else None,
                "forecast_comparison": forecast_comparison,
                "runtime_warnings": runtime_msgs,
                "time_linear_fit_s": time_linear_fit_s,
                "time_parallel_fit_s": time_parallel_fit_s,
                "time_statsmodels_fit_s": time_statsmodels_fit_s,
                "spark_timing": spark_timing_meta,
            }
            new_state["analysis_complete"] = True
            for msg in runtime_msgs:
                new_state["execution_log"].append(f"⚠ Aviso: {msg}")
            new_state["execution_log"].append("✓ Análisis completado")
            app_state.set(new_state)
            
            ui.notification_show(
                "✓ Análisis completado exitosamente",
                type="message",
                duration=3
            )
            
        except Exception as e:
            new_state = app_state.get().copy()
            new_state["execution_log"].append(f"✗ Error: {str(e)}")
            app_state.set(new_state)
            
            ui.notification_show(
                f"Error en ejecución: {str(e)}",
                type="error",
                duration=5
            )
    
    def validate_current_step(step: int, state: dict) -> bool:
        """Validate if current step can proceed to next"""
        if step == 0:  # Upload step
            # Require data loaded, column selected, and validated
            return (state.get("data_loaded", False) and 
                    state.get("value_column") is not None and 
                    state.get("data_validated", False))
        elif step == 1:  # Visualization step
            # Require validated data
            return state.get("data_validated", False)
        elif step == 2:  # Model + Execution step
            # Require analysis complete
            return state.get("analysis_complete", False)
        elif step == 3:  # Results step
            # Require analysis complete
            return state.get("analysis_complete", False)
        return True
    
    # Reactive UI updates based on state
    @reactive.effect
    def update_ui_state():
        """Update UI elements based on app state"""
        state = app_state.get()
        
        # Update step indicators
        if state["data_loaded"]:
            # Show data preview elements
            pass
        
        if state["analysis_complete"]:
            # Show results elements
            pass

# Create the Shiny app
app = App(app_ui, server)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
