"""Shiny app entry point: TSLib time series analysis wizard."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import traceback

from shiny import App, ui, reactive, render

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from components.stepper import StepperComponent
from components.layout import create_app_layout, create_data_table, create_metric_card, create_form_group, create_file_upload_area
from features.upload.ui import render_upload_ui
from features.visualization.ui import render_visualization_ui
from features.model_selection.ui import render_model_selection_ui
from features.results.ui import render_results_ui
from features.benchmark.ui import render_benchmark_ui
from features.benchmark.server import register_benchmark_server

from services.tslib_service import TSLibService, run_with_recorded_warnings
from services.evaluation_plots import (
    plot_residual_histogram_and_qq,
    plot_standardized_residuals,
)
from config_limits import MAX_UPLOAD_FILE_BYTES

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
        ui.tags.style(
            """
            /* TSLib Shiny App - Dark Professional Theme */
            :root {
              --bg-primary: #1a1a1a;
              --bg-secondary: #2d2d2d;
              --bg-tertiary: #3a3a3a;
              --bg-card: #252525;
              --bg-hover: #404040;
              --text-primary: #ffffff;
              --text-secondary: #b3b3b3;
              --text-muted: #cccccc;
              --accent-primary: #00d4aa;
              --accent-secondary: #0099cc;
              --accent-danger: #ff6b6b;
              --accent-warning: #ffd93d;
              --accent-success: #6bcf7f;
              --border-color: #404040;
              --border-light: #555555;
              --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.3);
              --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.4);
              --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.5);
              --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              --font-size-xs: 0.75rem;
              --font-size-sm: 0.875rem;
              --font-size-base: 1rem;
              --font-size-lg: 1.125rem;
              --font-size-xl: 1.25rem;
              --font-size-2xl: 1.5rem;
              --font-size-3xl: 1.875rem;
              --spacing-xs: 0.25rem;
              --spacing-sm: 0.5rem;
              --spacing-md: 1rem;
              --spacing-lg: 1.5rem;
              --spacing-xl: 2rem;
              --spacing-2xl: 3rem;
              --radius-sm: 0.25rem;
              --radius-md: 0.5rem;
              --radius-lg: 0.75rem;
              --radius-xl: 1rem;
            }

            body {
              font-family: var(--font-family);
              background-color: var(--bg-primary);
              color: var(--text-primary);
              line-height: 1.6;
              margin: 0;
              padding: 0;
              min-height: auto;
            }
            
            .main-container {
              background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
              padding: var(--spacing-sm) 0;
            }
            
            .app-header {
              background-color: var(--bg-secondary);
              border-bottom: 1px solid var(--border-color);
              padding: var(--spacing-lg) var(--spacing-xl);
              box-shadow: var(--shadow-sm);
            }
            
            .app-title {
              font-size: var(--font-size-2xl);
              font-weight: 700;
              color: var(--accent-primary);
              margin: 0;
            }
            
            .app-subtitle {
              font-size: var(--font-size-sm);
              color: var(--text-secondary);
              margin: var(--spacing-xs) 0 0 0;
            }
            
            .stepper-container {
              background-color: var(--bg-card);
              border-radius: var(--radius-lg);
              padding: var(--spacing-lg);
              margin: var(--spacing-md);
              box-shadow: var(--shadow-md);
            }
            
            .stepper-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: var(--spacing-xl);
              padding-bottom: var(--spacing-lg);
              border-bottom: 1px solid var(--border-color);
            }
            
            .stepper-title {
              font-size: var(--font-size-xl);
              font-weight: 600;
              color: var(--text-primary);
              margin: 0;
            }
            
            .stepper-progress {
              font-size: var(--font-size-sm);
              color: var(--text-secondary);
              background-color: var(--bg-tertiary);
              padding: var(--spacing-sm) var(--spacing-md);
              border-radius: var(--radius-md);
            }
            
            .stepper-content {
              min-height: 200px;
              padding: var(--spacing-md) 0;
            }
            
            .stepper-navigation {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-top: var(--spacing-xl);
              padding-top: var(--spacing-lg);
              border-top: 1px solid var(--border-color);
              padding-bottom: var(--spacing-sm);
              padding-left: var(--spacing-md);
              padding-right: var(--spacing-md);
            }
            
            .btn {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              padding: var(--spacing-sm) var(--spacing-lg);
              border: none;
              border-radius: var(--radius-md);
              font-size: var(--font-size-sm);
              font-weight: 500;
              text-decoration: none;
              cursor: pointer;
              transition: all 0.2s ease;
              min-width: 100px;
            }
            
            .btn-primary {
              background-color: var(--accent-primary);
              color: var(--bg-primary);
            }
            
            .btn-primary:hover:not(:disabled) {
              background-color: #00b894;
              transform: translateY(-1px);
              box-shadow: var(--shadow-md);
            }
            
            .btn-secondary {
              background-color: var(--bg-tertiary);
              color: var(--text-primary);
              border: 1px solid var(--border-color);
            }
            
            .btn-secondary:hover:not(:disabled) {
              background-color: var(--bg-hover);
              border-color: var(--border-light);
            }
            
            /* Disabled button styles */
            .btn:disabled,
            .btn.disabled {
              opacity: 0.5;
              cursor: not-allowed !important;
              pointer-events: none;
              transform: none !important;
            }
            
            .btn:disabled:hover,
            .btn.disabled:hover {
              background-color: inherit;
              transform: none;
              box-shadow: none;
            }
            
            .card {
              background-color: var(--bg-card);
              border: 1px solid var(--border-color);
              border-radius: var(--radius-lg);
              padding: var(--spacing-lg);
              margin-bottom: var(--spacing-lg);
              box-shadow: var(--shadow-sm);
              transition: all 0.2s ease;
            }
            
            .card:hover {
              box-shadow: var(--shadow-md);
              border-color: var(--border-light);
            }
            
            .card-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: var(--spacing-md);
              padding-bottom: var(--spacing-sm);
              border-bottom: 1px solid var(--border-color);
            }
            
            .card-title {
              font-size: var(--font-size-lg);
              font-weight: 600;
              color: var(--text-primary);
              margin: 0;
            }
            
            .card-subtitle {
              font-size: var(--font-size-sm);
              color: var(--text-secondary);
              margin: var(--spacing-xs) 0 0 0;
            }
            
            .file-upload-area {
              border: 2px dashed var(--border-color);
              border-radius: var(--radius-lg);
              padding: var(--spacing-2xl);
              text-align: center;
              background-color: var(--bg-tertiary);
              transition: all 0.2s ease;
              cursor: pointer;
            }
            
            .file-upload-area:hover {
              border-color: var(--accent-primary);
              background-color: var(--bg-hover);
            }
            
            .file-upload-icon {
              font-size: var(--font-size-3xl);
              color: var(--accent-primary);
              margin-bottom: var(--spacing-md);
            }
            
            .file-upload-text {
              font-size: var(--font-size-lg);
              color: var(--text-primary);
              margin-bottom: var(--spacing-sm);
            }
            
            .file-upload-hint {
              font-size: var(--font-size-sm);
              color: var(--text-secondary);
            }
            
            .metrics-grid {
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: var(--spacing-lg);
              margin-bottom: var(--spacing-xl);
            }
            
            .metric-card {
              background-color: var(--bg-card);
              border: 1px solid var(--border-color);
              border-radius: var(--radius-lg);
              padding: var(--spacing-lg);
              text-align: center;
              box-shadow: var(--shadow-sm);
            }
            
            .metric-value {
              font-size: var(--font-size-2xl);
              font-weight: 700;
              color: var(--accent-primary);
              margin-bottom: var(--spacing-xs);
            }
            
            .metric-label {
              font-size: var(--font-size-sm);
              color: var(--text-secondary);
              margin: 0;
            }
            
            .d-none { display: none; }
            .d-block { display: block; }
            .d-flex { display: flex; }
            .d-grid { display: grid; }
            
            .text-center { text-align: center; }
            .text-left { text-align: left; }
            .text-right { text-align: right; }
            
            .text-muted { 
              color: var(--text-secondary) !important; 
            }
            
            /* Ensure all paragraph text is visible */
            p.text-muted {
              color: var(--text-secondary) !important;
            }
            
            /* Model description specific styling */
            .model-description p.text-muted,
            .mt-2 p.text-muted,
            .mb-3 p.text-muted {
              color: var(--text-secondary) !important;
            }
            
            /* Override for regular paragraphs that should be primary */
            .card-body p:not(.text-muted),
            .card p:not(.text-muted) {
              color: var(--text-primary) !important;
            }
            
            .mt-1 { margin-top: var(--spacing-xs); }
            .mt-2 { margin-top: var(--spacing-sm); }
            .mt-3 { margin-top: var(--spacing-md); }
            .mt-4 { margin-top: var(--spacing-lg); }
            .mt-5 { margin-top: var(--spacing-xl); }
            
            .mb-1 { margin-bottom: var(--spacing-xs); }
            .mb-2 { margin-bottom: var(--spacing-sm); }
            .mb-3 { margin-bottom: var(--spacing-md); }
            .mb-4 { margin-bottom: var(--spacing-lg); }
            .mb-5 { margin-bottom: var(--spacing-xl); }
            
            .gap-2 { gap: var(--spacing-sm); }
            .gap-3 { gap: var(--spacing-md); }
            
            .justify-center { justify-content: center; }
            .justify-between { justify-content: space-between; }
            .align-center { align-items: center; }
            
            /* Form elements styling */
            .form-label {
              color: var(--text-primary) !important;
              font-weight: 500;
              margin-bottom: var(--spacing-xs);
              display: block;
            }
            
            .form-help {
              color: var(--text-secondary) !important;
              font-size: var(--font-size-xs);
              margin-top: var(--spacing-xs);
            }
            
            /* Ensure all labels are visible */
            label {
              color: var(--text-primary) !important;
            }
            
            /* Ensure all small text is visible */
            small {
              color: var(--text-secondary) !important;
            }
            
            /* Ensure all text elements are visible */
            h1, h2, h3, h4, h5, h6 {
              color: var(--text-primary) !important;
            }
            
            p, div, span {
              color: var(--text-primary) !important;
            }
            
            /* Specific styling for execution step */
            .execution-status, .estimated-time, .progress-step {
              color: var(--text-secondary) !important;
            }
            
            /* Data preview styling */
            #data_preview_placeholder {
              color: var(--text-secondary) !important;
            }
            
            /* Table styling - ensure all text is visible */
            table, .data-table {
              color: var(--text-primary) !important;
            }
            
            /* Data preview card wrapper for better spacing */
            .table-preview {
              background-color: var(--bg-secondary);
              border: 1px solid var(--border-color);
              border-radius: var(--radius-lg);
              padding: var(--spacing-md);
              margin-top: var(--spacing-sm);
              box-shadow: var(--shadow-sm);
              overflow: auto;
            }
            
            table th,
            table td,
            .data-table th,
            .data-table td {
              color: var(--text-primary) !important;
              border-color: var(--border-color) !important;
              padding: 12px 16px; /* add comfortable cell padding */
            }
            
            table thead th,
            .data-table thead th {
              background-color: var(--bg-secondary) !important;
              color: var(--text-primary) !important;
              position: sticky; /* keep header visible when scrolling */
              top: 0;
              z-index: 1;
            }
            
            table tbody td,
            .data-table tbody td {
              color: var(--text-secondary) !important;
            }
            
            table tbody tr:hover td,
            .data-table tbody tr:hover td {
              color: var(--text-primary) !important;
            }
            
            /* Subtle row separators */
            table tbody tr,
            .data-table tbody tr {
              border-bottom: 1px solid var(--border-color);
            }
            table tbody tr:last-child,
            .data-table tbody tr:last-child {
              border-bottom: none;
            }
            
            /* Shiny specific table elements */
            .shiny-table th,
            .shiny-table td {
              color: var(--text-primary) !important;
            }
            
            /* Shiny radio buttons container */
            .shiny-input-radiogroup label,
            .shiny-input-radiogroup .shiny-options-group label {
              color: var(--text-primary) !important;
            }
            
            .shiny-input-radiogroup input[type="radio"] {
              accent-color: var(--accent-primary);
              margin-right: var(--spacing-xs);
            }
            
            /* Shiny checkbox container */
            .shiny-input-checkboxgroup label {
              color: var(--text-primary) !important;
            }
            
            .shiny-input-checkboxgroup input[type="checkbox"] {
              accent-color: var(--accent-primary);
              margin-right: var(--spacing-xs);
            }
            
            /* Shiny select input */
            .shiny-input-select select {
              background-color: var(--bg-tertiary) !important;
              color: var(--text-primary) !important;
            }
            
            /* Shiny switch/checkbox input */
            .shiny-input-container label {
              color: var(--text-primary) !important;
            }
            
            /* Ensure all Shiny output text is visible */
            .shiny-text-output,
            .shiny-html-output {
              color: var(--text-primary) !important;
            }
            
            /* Shiny notification styling */
            .shiny-notification {
              background-color: var(--bg-card) !important;
              color: var(--text-primary) !important;
              border: 1px solid var(--border-color) !important;
            }
            
            /* Input styling */
            input[type="number"], input[type="text"], input[type="email"], input[type="password"], 
            select, textarea {
              background-color: var(--bg-tertiary) !important;
              border: 1px solid var(--border-color);
              border-radius: var(--radius-md);
              color: var(--text-primary) !important;
              padding: var(--spacing-sm) var(--spacing-md);
              font-size: var(--font-size-sm);
            }
            
            input[type="number"]:hover, input[type="text"]:hover, input[type="email"]:hover, 
            input[type="password"]:hover, select:hover, textarea:hover {
              background-color: var(--bg-hover) !important;
              border-color: var(--border-light);
              color: var(--text-primary) !important;
            }
            
            input[type="number"]:focus, input[type="text"]:focus, input[type="email"]:focus, 
            input[type="password"]:focus, select:focus, textarea:focus {
              outline: none !important;
              border-color: var(--accent-primary) !important;
              box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2) !important;
              background-color: var(--bg-hover) !important;
              color: var(--text-primary) !important;
            }
            
            /* Select dropdown styling */
            select {
              background-color: var(--bg-tertiary) !important;
              color: var(--text-primary) !important;
              background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23ffffff' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
              background-repeat: no-repeat;
              background-position: right var(--spacing-sm) center;
              padding-right: var(--spacing-xl);
              appearance: none;
              -webkit-appearance: none;
              -moz-appearance: none;
            }
            
            select option {
              background-color: var(--bg-tertiary);
              color: var(--text-primary);
              padding: var(--spacing-sm);
            }
            
            select option:checked {
              background-color: var(--accent-primary);
              color: var(--bg-primary);
            }
            
            /* Radio buttons and checkboxes styling */
            .form-check {
              display: flex;
              align-items: center;
              margin-bottom: var(--spacing-sm);
            }
            
            .form-check-input {
              background-color: var(--bg-tertiary);
              border-color: var(--border-color);
              width: 1.25em;
              height: 1.25em;
              margin-right: var(--spacing-sm);
              cursor: pointer;
            }
            
            .form-check-input:checked {
              background-color: var(--accent-primary);
              border-color: var(--accent-primary);
            }
            
            .form-check-input:focus {
              outline: none;
              box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2);
            }
            
            .form-check-label {
              color: var(--text-primary) !important;
              margin-left: 0;
              cursor: pointer;
              font-size: var(--font-size-sm);
            }
            
            /* Radio buttons specific */
            input[type="radio"] {
              accent-color: var(--accent-primary);
            }
            
            input[type="radio"]:checked {
              background-color: var(--accent-primary);
            }
            
            /* Checkboxes specific */
            input[type="checkbox"] {
              accent-color: var(--accent-primary);
            }
            
            /* Switch styling */
            .form-switch .form-check-input {
              width: 2.5em;
              height: 1.25em;
              background-color: var(--bg-tertiary);
              border-color: var(--border-color);
            }
            
            .form-switch .form-check-input:checked {
              background-color: var(--accent-primary);
              border-color: var(--accent-primary);
            }
            
            .form-switch .form-check-label {
              color: var(--text-primary) !important;
            }
            
            /* Slider styling */
            input[type="range"] {
              background: var(--bg-tertiary);
              border-radius: var(--radius-md);
            }
            
            input[type="range"]::-webkit-slider-thumb {
              background: var(--accent-primary);
              border-radius: 50%;
            }
            
            input[type="range"]::-moz-range-thumb {
              background: var(--accent-primary);
              border-radius: 50%;
            }
            
            /* Status indicators */
            .status-indicator {
              padding: var(--spacing-xs) var(--spacing-sm);
              border-radius: var(--radius-sm);
              font-size: var(--font-size-sm);
              font-weight: 500;
            }
            
            .status-info {
              background-color: rgba(0, 153, 204, 0.2);
              color: var(--accent-secondary);
              border: 1px solid var(--accent-secondary);
            }
            
            .status-success {
              background-color: rgba(107, 207, 127, 0.2);
              color: var(--accent-success);
              border: 1px solid var(--accent-success);
            }
            
            .status-warning {
              background-color: rgba(255, 217, 61, 0.2);
              color: var(--accent-warning);
              border: 1px solid var(--accent-warning);
            }
            
            .status-danger {
              background-color: rgba(255, 107, 107, 0.2);
              color: var(--accent-danger);
              border: 1px solid var(--accent-danger);
            }
            
            /* Progress steps */
            .progress-step {
              color: var(--text-secondary);
              font-size: var(--font-size-sm);
              padding: var(--spacing-xs) 0;
              border-left: 2px solid var(--border-color);
              padding-left: var(--spacing-sm);
              margin-bottom: var(--spacing-xs);
            }
            
            .progress-list {
              margin-top: var(--spacing-sm);
            }
            
            /* Button variants */
            .btn-lg {
              padding: var(--spacing-md) var(--spacing-xl);
              font-size: var(--font-size-lg);
            }
            
            .btn-warning {
              background-color: var(--accent-warning);
              color: var(--bg-primary);
            }
            
            .btn-warning:hover:not(:disabled) {
              background-color: #e6c200;
              transform: translateY(-1px);
              box-shadow: var(--shadow-md);
            }
            
            .btn-danger {
              background-color: var(--accent-danger);
              color: var(--bg-primary);
            }
            
            .btn-danger:hover:not(:disabled) {
              background-color: #ff5252;
              transform: translateY(-1px);
              box-shadow: var(--shadow-md);
            }
            
            /* File upload styling */
            .file-input-hidden {
              display: none !important;
            }
            
            .file-upload-wrapper {
              position: relative;
            }
            
            .file-upload-area {
              border: 2px dashed var(--border-color);
              border-radius: var(--radius-lg);
              padding: var(--spacing-2xl);
              text-align: center;
              background-color: var(--bg-tertiary);
              transition: all 0.2s ease;
              cursor: pointer;
              position: relative;
            }
            
            .file-upload-area:hover {
              border-color: var(--accent-primary);
              background-color: var(--bg-hover);
            }
            
            .file-upload-area:active {
              transform: scale(0.98);
            }
            </style>
            <script>
            // Handle custom messages from server
            Shiny.addCustomMessageHandler("update_preview", function(message) {
              // Hide placeholder and show content
              const placeholder = document.getElementById("data_preview_placeholder");
              const content = document.getElementById("data_preview_content");
              
              if (placeholder && content) {
                placeholder.style.display = "none";
                content.classList.remove("d-none");
                content.classList.add("d-block");
                
                // Update file info
                const rowCount = document.getElementById("row_count");
                const colCount = document.getElementById("col_count");
                const fileSize = document.getElementById("file_size");
                
                if (rowCount) rowCount.textContent = "Filas: " + message.rows;
                if (colCount) colCount.textContent = "Columnas: " + message.columns;
                if (fileSize) fileSize.textContent = "Tamaño: " + message.size;
              }
            });
            </script>
            """
        )
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
        ui.nav_panel("🚀 Benchmark",
            create_app_layout(
                title="Suite de Benchmark Paralelo",
                subtitle="Comparativa de tiempos de ajuste para todos los modelos"
            ),
            ui.div(
                render_benchmark_ui(),
                class_="container-fluid mt-4"
            )
        ),
        title="TSLib",
        bg="var(--bg-secondary)",
        inverse=True
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
        """Explain exploratory findings in plain Spanish."""
        state = app_state.get()
        validation = state.get("validation_report", {}) or {}
        quality = validation.get("quality_report", {}) or {}
        analysis = state.get("exploratory_analysis", {}) or {}
        meta = analysis.get("meta", {}) or {}

        if not validation and not analysis:
            return ui.tags.p("Valida los datos para ver notas de exploración.", class_="text-muted")

        notes = []

        recs = quality.get("recommendations", [])
        if any("Seasonal patterns detected" in str(r) for r in recs):
            notes.append(
                "Señal estacional detectada por picos de autocorrelación en lags típicos "
                "(4, 7, 12, 24) con umbral aproximado |ACF| > 0.3."
            )
        if any("Trend detected" in str(r) for r in recs):
            notes.append(
                "Señal de tendencia detectada con ajuste lineal simple "
                "(R² por encima del umbral interno del validador)."
            )

        if not notes:
            return ui.tags.p(
                "No se detectaron señales relevantes adicionales en la exploración.",
                class_="text-muted",
            )

        return ui.div(
            *[ui.div(f"• {n}", class_="progress-step") for n in notes],
            class_="progress-list",
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
            ui.tags.p(f"Backend paralelo ARIMA: {parallel_backend}") if parallel_backend else None,
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
        """Render metrics cards"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return ui.div(
                ui.tags.p("Las métricas aparecerán después de ejecutar el análisis", class_="text-muted")
            )
        
        fitted_model = state.get("fitted_model")
        if not fitted_model:
            return ui.div(ui.tags.p("No hay métricas disponibles", class_="text-muted"))
        
        metrics = tslib_service.get_model_metrics(fitted_model)
        
        return ui.div(
            create_metric_card(
                f"{metrics.get('aic', 0):.2f}" if metrics.get('aic') else "N/A",
                "AIC",
                "📊"
            ),
            create_metric_card(
                f"{metrics.get('bic', 0):.2f}" if metrics.get('bic') else "N/A",
                "BIC",
                "📊"
            ),
            create_metric_card(
                metrics.get('order', 'N/A'),
                "Orden",
                "⚙️"
            ),
            class_="metrics-grid"
        )
    
    @render.plot
    def forecast_plot():
        """Render forecast plot"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'Ejecuta el análisis primero', ha='center', va='center')
            ax.axis('off')
            return fig
        
        df = uploaded_dataframe.get()
        value_col = state.get("value_column")
        forecast_results = state.get("forecast_results")
        
        if not forecast_results or df is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'No hay pronóstico disponible', ha='center', va='center')
            ax.axis('off')
            return fig
        
        # Plot historical data + forecast
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Convert to numeric if needed
        if pd.api.types.is_numeric_dtype(df[value_col]):
            historical = df[value_col].values
        else:
            historical = tslib_service.convert_to_numeric(df, value_col).values
        
        historical = np.asarray(historical, dtype=float)

        forecast = forecast_results.get('forecast', [])
        lower = forecast_results.get('lower_bound')
        upper = forecast_results.get('upper_bound')
        
        n_hist = len(historical)
        n_fore = len(forecast)
        
        # Plot historical
        ax.plot(range(n_hist), historical, label='Histórico', color='#00d4aa', linewidth=1.5)
        
        # Plot forecast
        forecast_x = range(n_hist, n_hist + n_fore)
        ax.plot(forecast_x, forecast, label='Pronóstico', color='#0099cc', linewidth=1.5, linestyle='--')
        
        # Plot confidence intervals if available
        if lower is not None and upper is not None:
            ax.fill_between(forecast_x, lower, upper, alpha=0.3, color='#0099cc', label='IC 95%')
        
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Valor')
        ax.set_title('Pronóstico de Serie Temporal', fontsize=12, fontweight='bold')
        ax.legend(loc='best', facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
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
    
    @render.plot
    def residuals_plot():
        """Render residuals plot"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'Ejecuta el análisis primero', ha='center', va='center')
            ax.axis('off')
            return fig
        
        fitted_model = state.get("fitted_model")
        if not fitted_model or not hasattr(fitted_model, 'get_residuals'):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'No hay residuos disponibles', ha='center', va='center')
            ax.axis('off')
            return fig
        
        residuals = fitted_model.get_residuals()
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(residuals, color='#00d4aa', linewidth=1, alpha=0.8)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Residuos')
        ax.set_title('Residuos del Modelo', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#2d2d2d')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        plt.tight_layout()
        return fig
    
    @render.plot
    def residuals_acf_plot():
        """Render residuals ACF plot"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'Ejecuta el análisis primero', ha='center', va='center', color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        fitted_model = state.get("fitted_model")
        if not fitted_model or not hasattr(fitted_model, 'get_residuals'):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'No hay residuos disponibles', ha='center', va='center', color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        residuals = fitted_model.get_residuals()
        try:
            from tslib.core.acf_pacf import ACFCalculator
            acf_calc = ACFCalculator()
            acf_result = acf_calc.calculate(residuals)
            if isinstance(acf_result, tuple) and len(acf_result) == 2:
                lags, acf_values = acf_result
            else:
                acf_values = acf_result

            if isinstance(acf_values, np.ndarray):
                pass
            elif isinstance(acf_values, list):
                acf_values = np.array(acf_values)
            else:
                acf_values = np.array(list(acf_values)) if hasattr(acf_values, "__iter__") else np.array([])

            if len(acf_values) > 20:
                acf_values = acf_values[:20]

            if acf_values is None or len(acf_values) == 0:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.text(0.5, 0.5, "ACF no disponible", ha="center", va="center", color="white")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                ax.set_facecolor("#2d2d2d")
                fig.patch.set_facecolor("#1a1a1a")
                return fig

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.stem(range(len(acf_values)), acf_values, linefmt="#00d4aa", markerfmt="o", basefmt=" ")
            ax.axhline(y=0, color="white", linestyle="-", linewidth=0.5)
            if len(residuals) > 0:
                conf_level = 1.96 / np.sqrt(len(residuals))
                ax.axhline(y=conf_level, color="red", linestyle="--", linewidth=1, alpha=0.7)
                ax.axhline(y=-conf_level, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel("Lag")
            ax.set_ylabel("ACF")
            ax.set_title("ACF de Residuos", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#2d2d2d")
            fig.patch.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            return fig
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, f"Error al calcular ACF:\n{str(e)}", ha="center", va="center", color="white", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_facecolor("#2d2d2d")
            fig.patch.set_facecolor("#1a1a1a")
            return fig

    @render.plot
    def eval_residual_hist_qq_plot():
        state = app_state.get()
        if not state.get("analysis_complete"):
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "Ejecuta el análisis primero", ha="center", va="center", color="white")
            ax.axis("off")
            return fig
        fitted_model = state.get("fitted_model")
        if not fitted_model or not hasattr(fitted_model, "get_residuals"):
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "Sin residuos", ha="center", va="center", color="white")
            ax.axis("off")
            return fig
        r = fitted_model.get_residuals()
        return plot_residual_histogram_and_qq(np.asarray(r, dtype=float))

    @render.plot
    def eval_standardized_residuals_plot():
        state = app_state.get()
        if not state.get("analysis_complete"):
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "Ejecuta el análisis primero", ha="center", va="center", color="white")
            ax.axis("off")
            return fig
        fitted_model = state.get("fitted_model")
        if not fitted_model or not hasattr(fitted_model, "get_residuals"):
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "Sin residuos", ha="center", va="center", color="white")
            ax.axis("off")
            return fig
        r = np.asarray(fitted_model.get_residuals(), dtype=float)
        return plot_standardized_residuals(r)

    @render.ui
    def linear_model_title():
        """Show linear model title."""
        return ui.tags.h4("Modelo lineal", class_="mb-3")
    
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
            # Parallel metrics
            ui.div(
                ui.tags.h5("Métricas de evaluación (paralelo):"),
                ui.output_ui("parallel_metrics_cards"),
                class_="mb-4"
            ),
            # Parallel forecast plot
            ui.div(
                ui.tags.h5("Pronóstico (paralelo):"),
                ui.output_plot("parallel_forecast_plot", height="400px"),
                class_="mb-4"
            ),
            # Parallel forecast table
            ui.div(
                ui.tags.h5("Valores del pronóstico (paralelo):"),
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
        
        return ui.div(
            ui.tags.p(f"Tipo de Modelo: {model_type} Paralelo"),
            ui.tags.p(f"Orden: {order}"),
            ui.tags.p(f"Backend: {backend}"),
            ui.tags.p(f"Detalle backend: {backend_desc}"),
            class_="text-muted"
        )
    
    @render.ui
    def parallel_metrics_cards():
        """Render parallel model metrics cards"""
        state = app_state.get()
        parallel_workflow = state.get("parallel_workflow")
        model_type = state.get("model_type", "N/A")
        
        if not parallel_workflow:
            return ui.div(ui.tags.p("No hay métricas disponibles", class_="text-muted"))
        
        metrics = tslib_service.get_parallel_model_metrics(parallel_workflow, model_type=model_type)
        comparison = (state.get("execution_metadata", {}) or {}).get("forecast_comparison", {})
        
        cards = []
        
        # Order card
        if metrics.get('order'):
            cards.append(create_metric_card(
                metrics.get('order', 'N/A'),
                "Orden",
                "⚙️"
            ))
        
        # MAE card
        if metrics.get('mae') is not None:
            cards.append(create_metric_card(
                f"{metrics.get('mae', 0):.4f}",
                "MAE",
                "📊"
            ))
        
        # RMSE card
        if metrics.get('rmse') is not None:
            cards.append(create_metric_card(
                f"{metrics.get('rmse', 0):.4f}",
                "RMSE",
                "📊"
            ))
        
        # MAPE card
        if metrics.get('mape') is not None:
            cards.append(create_metric_card(
                f"{metrics.get('mape', 0):.4f}",
                "MAPE",
                "📊"
            ))
        if comparison:
            cards.append(create_metric_card(
                f"{comparison.get('mae_diff', 0):.4f}",
                "MAE Dif. L/P",
                "↔️"
            ))
            cards.append(create_metric_card(
                f"{comparison.get('rmse_diff', 0):.4f}",
                "RMSE Dif. L/P",
                "↔️"
            ))
        
        if not cards:
            return ui.div(ui.tags.p("No hay métricas disponibles", class_="text-muted"))
        
        return ui.div(*cards, class_="metrics-grid")
    
    @render.plot
    def parallel_forecast_plot():
        """Render parallel forecast plot"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'Ejecuta el análisis primero', ha='center', va='center', color='white')
            ax.axis('off')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        df = uploaded_dataframe.get()
        value_col = state.get("value_column")
        parallel_forecast_results = state.get("parallel_forecast_results")
        
        if not parallel_forecast_results or df is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'No hay pronóstico paralelo disponible', ha='center', va='center', color='white')
            ax.axis('off')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#1a1a1a')
            return fig
        
        # Plot historical data + forecast
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Convert to numeric if needed
        if pd.api.types.is_numeric_dtype(df[value_col]):
            historical = df[value_col].values
        else:
            historical = tslib_service.convert_to_numeric(df, value_col).values
        
        # Handle missing values in historical data (forward fill for plotting)
        if np.any(np.isnan(historical)):
            mask = np.isnan(historical)
            indices = np.arange(len(historical))
            if np.any(~mask):
                historical[mask] = np.interp(indices[mask], indices[~mask], historical[~mask])
            else:
                historical = np.zeros_like(historical)
        
        forecast = parallel_forecast_results.get('forecast', [])
        lower = parallel_forecast_results.get('lower_bound')
        upper = parallel_forecast_results.get('upper_bound')
        
        n_hist = len(historical)
        n_fore = len(forecast)
        
        # Plot historical
        ax.plot(range(n_hist), historical, label='Histórico', color='#00d4aa', linewidth=1.5)
        
        # Plot forecast
        forecast_x = range(n_hist, n_hist + n_fore)
        ax.plot(forecast_x, forecast, label='Pronóstico (Paralelo)', color='#ff6b6b', linewidth=1.5, linestyle='--')
        
        # Plot confidence intervals if available
        if lower is not None and upper is not None:
            ax.fill_between(forecast_x, lower, upper, alpha=0.3, color='#ff6b6b', label='IC 95%')
        
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Valor')
        ax.set_title('Pronóstico de Serie Temporal (Ruta Paralela)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
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
    def parallel_forecast_table_ui():
        """Render parallel forecast table"""
        state = app_state.get()
        if not state.get("analysis_complete"):
            return ui.div(ui.tags.p("Ejecuta el análisis primero", class_="text-muted"))
        
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
            new_state["execution_log"] = [
                "Iniciando análisis...",
                f"Modelo seleccionado: {model_type}",
                "Ajustando modelo...",
            ]
            new_state["runtime_warnings"] = []
            app_state.set(new_state)

            fitted_model, w_fit = run_with_recorded_warnings(
                lambda: tslib_service.fit_model(
                    data=data,
                    model_type=model_type,
                    order=order if order else (1,) if model_type in ["AR", "MA"] else (1, 1) if model_type == "ARMA" else (1, 1, 1),
                    auto_select=auto_select,
                    validation_report=quality_report,
                )
            )
            runtime_msgs.extend(w_fit)
            effective_order = None
            if hasattr(fitted_model, "order"):
                effective_order = str(fitted_model.order)
                new_state = app_state.get().copy()
                new_state["execution_log"].append(f"Orden efectivo ajustado: {effective_order}")
                app_state.set(new_state)
            
            # Update log
            new_state = app_state.get().copy()
            new_state["execution_log"].append("Generando pronóstico (modelo lineal)...")
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
            try:
                parallel_workflow, parallel_forecast_results = tslib_service.fit_parallel_model_spark(
                    data=data,
                    model_type=model_type,
                    order=order if order else (1,) if model_type in ["AR", "MA"] else (1, 1) if model_type == "ARMA" else (1, 1, 1),
                    steps=forecast_steps,
                    validation_report=quality_report,
                )
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
            new_state["execution_metadata"] = {
                "effective_order": effective_order,
                "parallel_backend": getattr(parallel_workflow, "backend_", None) if parallel_workflow else None,
                "forecast_comparison": forecast_comparison,
                "runtime_warnings": runtime_msgs,
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
