"""
Matplotlib helpers for model evaluation (residuals, forecasts, comparisons).
Plotly variants for interactive Shiny dashboards (zoom/pan).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _plotly_dark_layout(fig, title: str, *, yaxis_title: str = "", xaxis_title: str = "") -> None:
    import plotly.graph_objects as go

    assert isinstance(fig, go.Figure)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#2d2d2d",
        font=dict(color="#ececec", size=11),
        title=dict(text=title, x=0.5),
        xaxis=dict(title=xaxis_title or "Índice", gridcolor="#444444"),
        yaxis=dict(title=yaxis_title, gridcolor="#444444"),
        margin=dict(l=50, r=24, t=56, b=48),
        dragmode="zoom",
        hovermode="x unified",
    )


def plot_residuals_plotly(residuals: np.ndarray, title: str = "Residuos del modelo") -> Any:
    import plotly.graph_objects as go

    r = np.asarray(residuals, dtype=float).ravel()
    x = np.arange(len(r))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=r, mode="lines", name="Residuos", line=dict(color="#00d4aa", width=1))
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b")
    _plotly_dark_layout(fig, title, yaxis_title="Residuo", xaxis_title="Tiempo")
    return fig


def plot_residual_acf_plotly(residuals: np.ndarray, max_lag: int = 20, title: str = "ACF de residuos") -> Any:
    import plotly.graph_objects as go

    r = np.asarray(residuals, dtype=float).ravel()
    try:
        from tslib.core.acf_pacf import ACFCalculator

        acf_calc = ACFCalculator(max_lags=max_lag, n_jobs=1)
        acf_result = acf_calc.calculate(r)
        if isinstance(acf_result, tuple) and len(acf_result) == 2:
            _, acf_values = acf_result
        else:
            acf_values = acf_result
        v = np.asarray(acf_values, dtype=float).ravel()[:max_lag]
    except Exception:
        v = np.array([], dtype=float)
    if len(v) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="ACF no disponible",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#aaa"),
        )
        _plotly_dark_layout(fig, title)
        return fig
    lags = np.arange(len(v))
    conf = 1.96 / np.sqrt(len(r)) if len(r) > 0 else 0.05
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=v, name="ACF", marker_color="#00d4aa"))
    fig.add_hline(y=conf, line_dash="dash", line_color="#ff6b6b", opacity=0.7)
    fig.add_hline(y=-conf, line_dash="dash", line_color="#ff6b6b", opacity=0.7)
    _plotly_dark_layout(fig, title, yaxis_title="ACF", xaxis_title="Lag")
    return fig


def plot_residual_hist_qq_plotly(residuals: np.ndarray) -> Any:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    r = np.asarray(residuals, dtype=float).ravel()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histograma de residuos", "Q-Q (normal)"))
    fig.add_trace(
        go.Histogram(x=r, nbinsx=min(30, max(10, len(r) // 5)), name="Hist", marker_color="#0099cc"),
        row=1,
        col=1,
    )
    std_r = float(np.std(r)) if len(r) else 0.0
    degenerate = len(r) < 3 or std_r < 1e-15 or (float(np.max(r)) - float(np.min(r))) < 1e-15
    if degenerate:
        fig.add_annotation(
            text="Q–Q no aplicable",
            xref="x2",
            yref="y2",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#aaa"),
        )
    else:
        r_sorted = np.sort(r)
        n = len(r_sorted)
        p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        theoretical = stats.norm.ppf(p)
        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=r_sorted,
                mode="markers",
                name="Q-Q",
                marker=dict(color="#00d4aa", size=5),
            ),
            row=1,
            col=2,
        )
        t_q = stats.norm.ppf([0.25, 0.75])
        s_q = np.percentile(r, [25, 75])
        slope = (s_q[1] - s_q[0]) / (t_q[1] - t_q[0]) if abs(t_q[1] - t_q[0]) > 1e-12 else 1.0
        intercept = s_q[0] - slope * t_q[0]
        tx = np.array([float(theoretical.min()), float(theoretical.max())])
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=slope * tx + intercept,
                mode="lines",
                name="referencia",
                line=dict(color="#888", width=1),
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#2d2d2d",
        font=dict(color="#ececec"),
        title=dict(text="Histograma y Q-Q de residuos", x=0.5),
        margin=dict(l=50, r=24, t=56, b=48),
        dragmode="zoom",
        showlegend=False,
    )
    return fig


def plot_standardized_residuals_plotly(residuals: np.ndarray, sigma: Optional[float] = None) -> Any:
    import plotly.graph_objects as go

    r = np.asarray(residuals, dtype=float).ravel()
    s = float(sigma) if sigma is not None else float(np.std(r)) or 1.0
    z = r / s
    x = np.arange(len(z))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=z, mode="lines", line=dict(color="#feca57", width=1), name="z"))
    fig.add_hline(y=2, line_dash="dot", line_color="#ff6b6b")
    fig.add_hline(y=-2, line_dash="dot", line_color="#ff6b6b")
    fig.add_hline(y=0, line_color="white", opacity=0.5)
    _plotly_dark_layout(fig, "Residuos estandarizados (±2σ)", yaxis_title="z", xaxis_title="Índice")
    return fig


def style_dark_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#2d2d2d")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, alpha=0.3)
    for s in ax.spines.values():
        s.set_color("#666666")


def plot_residuals_vs_time(
    residuals: np.ndarray,
    title: str = "Residuos vs tiempo",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    r = np.asarray(residuals, dtype=float).ravel()
    ax.plot(r, color="#00d4aa", lw=1)
    ax.axhline(0, color="#ff6b6b", ls="--", lw=1)
    ax.set_xlabel("Índice")
    ax.set_ylabel("Residuo")
    ax.set_title(title, fontsize=11)
    style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def plot_residual_histogram_and_qq(residuals: np.ndarray) -> plt.Figure:
    r = np.asarray(residuals, dtype=float).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axh, axq = axes
    axh.hist(r, bins=min(30, max(10, len(r) // 5)), color="#0099cc", edgecolor="#1a1a1a", alpha=0.85)
    axh.set_title("Histograma de residuos")
    axh.set_xlabel("Residuo")
    axh.set_ylabel("Frecuencia")

    std_r = float(np.std(r)) if len(r) else 0.0
    degenerate = len(r) < 3 or std_r < 1e-15 or (float(np.max(r)) - float(np.min(r))) < 1e-15
    if degenerate:
        axq.text(
            0.5,
            0.5,
            "Varianza nula o muestra corta;\nQ–Q no aplicable.",
            ha="center",
            va="center",
            transform=axq.transAxes,
            fontsize=10,
            color="white",
        )
        axq.set_xticks([])
        axq.set_yticks([])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            stats.probplot(r, dist="norm", plot=axq)
    axq.set_title("Q-Q (normal)")
    axq.get_lines()[0].set_markerfacecolor("#00d4aa")
    axq.get_lines()[0].set_markeredgecolor("#00d4aa")

    for ax in axes:
        style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def plot_standardized_residuals(residuals: np.ndarray, sigma: Optional[float] = None) -> plt.Figure:
    r = np.asarray(residuals, dtype=float).ravel()
    s = float(sigma) if sigma is not None else float(np.std(r)) or 1.0
    z = r / s
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(z, color="#feca57", lw=1)
    ax.axhline(2, color="#ff6b6b", ls=":", lw=1)
    ax.axhline(-2, color="#ff6b6b", ls=":", lw=1)
    ax.axhline(0, color="white", ls="-", lw=0.5)
    ax.set_title("Residuos estandarizados (±2σ)")
    ax.set_xlabel("Índice")
    ax.set_ylabel("z")
    style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def plot_forecast_vs_actual(
    historical: np.ndarray,
    forecast: Sequence[float],
    actual_holdout: Optional[np.ndarray] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    title: str = "Pronóstico vs real",
) -> plt.Figure:
    hist = np.asarray(historical, dtype=float).ravel()
    fc = np.asarray(list(forecast), dtype=float).ravel()
    n = len(hist)
    h = len(fc)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(n), hist, label="Histórico", color="#00d4aa", lw=1.5)
    fx = range(n, n + h)
    ax.plot(fx, fc, label="Pronóstico", color="#0099cc", lw=1.5, ls="--")
    if actual_holdout is not None:
        ah = np.asarray(actual_holdout, dtype=float).ravel()
        m = min(h, len(ah))
        ax.plot(list(fx)[:m], ah[:m], label="Real (holdout)", color="#ff9f43", lw=1.5)
    if lower is not None and upper is not None:
        lo = np.asarray(list(lower), dtype=float)
        up = np.asarray(list(upper), dtype=float)
        ax.fill_between(list(fx), lo[:h], up[:h], alpha=0.25, color="#0099cc", label="IC")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.legend(loc="best", facecolor="#2d2d2d", edgecolor="white", labelcolor="white")
    style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def plot_error_by_horizon(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric: str = "mae",
) -> plt.Figure:
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    m = min(len(a), len(p))
    a, p = a[:m], p[:m]
    errs = np.abs(a - p)
    if metric.lower() == "rmse":
        errs = (a - p) ** 2
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(1, m + 1), errs, color="#54a0ff", edgecolor="#1a1a1a")
    ax.set_xlabel("Horizonte h")
    ax.set_ylabel("Error" if metric.lower() == "mae" else "Error²")
    ax.set_title(f"Error por horizonte ({metric.upper()})")
    style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def plot_fan_chart(
    forecast: Sequence[float],
    lower_50: Sequence[float],
    upper_50: Sequence[float],
    lower_80: Sequence[float],
    upper_80: Sequence[float],
    historical_tail: Optional[np.ndarray] = None,
) -> plt.Figure:
    fc = np.asarray(list(forecast), dtype=float).ravel()
    h = len(fc)
    base = 0
    if historical_tail is not None:
        tail = np.asarray(historical_tail, dtype=float).ravel()
        base = len(tail)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(base), tail, color="#00d4aa", lw=1.5, label="Histórico (cola)")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(base, base + h)
    ax.fill_between(x, lower_80[:h], upper_80[:h], alpha=0.2, color="#5f27cd", label="80%")
    ax.fill_between(x, lower_50[:h], upper_50[:h], alpha=0.35, color="#54a0ff", label="50%")
    ax.plot(x, fc, color="white", lw=2, label="Mediana / punto")
    ax.set_title("Fan chart (pronóstico)")
    ax.legend(loc="best", facecolor="#2d2d2d", edgecolor="white", labelcolor="white")
    style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def plot_method_metric_bars(
    labels: List[str],
    values: List[float],
    ylabel: str = "RMSE",
    title: str = "Comparación por método",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=["#ff6b6b", "#10ac84", "#54a0ff", "#feca57"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def plot_aicc_heatmap(matrix: np.ndarray, p_labels: List[int], q_labels: List[int]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(q_labels)))
    ax.set_xticklabels(q_labels)
    ax.set_yticks(range(len(p_labels)))
    ax.set_yticklabels(p_labels)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title("AICc (menor es mejor)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def plot_window_rank_boxplot(
    data_by_pq: Dict[str, Sequence[float]],
    title: str = "Rango AICc por ventana",
) -> plt.Figure:
    labels = list(data_by_pq.keys())
    data = [np.asarray(data_by_pq[k], dtype=float) for k in labels]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_ylabel("Rango (ventanas)")
    ax.set_title(title)
    style_dark_axes(ax)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig
