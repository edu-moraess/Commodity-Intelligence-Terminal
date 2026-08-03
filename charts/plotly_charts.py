"""
Charts — Biblioteca Plotly com Tema Escuro Institucional
============================================================
Funções fábrica que retornam `go.Figure` prontas para `st.plotly_chart`.
Centralizar aqui evita duplicar `update_layout` em cada página e garante
consistência visual (fonte, grid, cores) em todo o terminal.
"""

from __future__ import annotations
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config.settings import THEME

# MARGENS AJUSTADAS para eliminar espaço preto à direita
_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=THEME["surface"],
    plot_bgcolor=THEME["surface"],
    font=dict(
        color=THEME["text"],
        family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        size=12,
    ),
    margin=dict(l=36, r=24, t=52, b=36),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=THEME["surface_alt"] if "surface_alt" in THEME else THEME["surface"],
        bordercolor=THEME["border"],
        font=dict(family="Inter, sans-serif", size=12, color=THEME["text"]),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        orientation="h",
        y=1.10,
        x=0,
        font=dict(size=11, color=THEME["text_muted"]),
    ),
    xaxis=dict(
        gridcolor="rgba(38,38,44,0.55)",
        zerolinecolor="rgba(38,38,44,0.8)",
        showline=True,
        linecolor=THEME["border"],
        tickfont=dict(size=11, color=THEME["text_muted"]),
        title_font=dict(size=11, color=THEME["text_muted"]),
    ),
    yaxis=dict(
        gridcolor="rgba(38,38,44,0.55)",
        zerolinecolor="rgba(38,38,44,0.8)",
        showline=True,
        linecolor=THEME["border"],
        tickfont=dict(size=11, color=THEME["text_muted"]),
        title_font=dict(size=11, color=THEME["text_muted"]),
        side="right",
    ),
)


def _apply_theme(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    layout = dict(_LAYOUT_DEFAULTS)
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=14, color=THEME["text"], family="Inter, sans-serif"),
            x=0.0,
            xanchor="left",
            pad=dict(b=8),
        )
    fig.update_layout(**layout, height=height)
    # Linhas um pouco mais espessas por padrão (quando aplicável)
    for tr in fig.data:
        if hasattr(tr, "line") and tr.line is not None and getattr(tr.line, "width", None) in (None, 1, 1.0):
            try:
                tr.line.width = 2.0
            except Exception:
                pass
    return fig


def candlestick_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=THEME["positive"], decreasing_line_color=THEME["negative"],
        name="OHLC",
    )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_theme(fig, title)


def line_chart(series_dict: dict[str, pd.Series], title: str = "", y_title: str = "") -> go.Figure:
    fig = go.Figure()
    palette = [THEME["accent"], THEME["positive"], THEME["warning"], THEME["negative"],
               THEME["chart_extra_1"], THEME["chart_extra_2"]]
    for i, (name, s) in enumerate(series_dict.items()):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name,
                                  line=dict(color=palette[i % len(palette)], width=1.8)))
    fig.update_yaxes(title_text=y_title)
    return _apply_theme(fig, title)


def correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Matriz de Correlação") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale=[[0, THEME["negative"]], [0.5, THEME["surface"]], [1, THEME["positive"]]],
        zmid=0, zmin=-1, zmax=1,
        text=corr_matrix.round(2).values, texttemplate="%{text}",
        colorbar=dict(title="ρ"),
    ))
    return _apply_theme(fig, title, height=max(360, 28 * len(corr_matrix)))


def fan_chart(fan_df: pd.DataFrame, last_price: float, last_date: pd.Timestamp,
              title: str = "Forecast — Cenários Probabilísticos") -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(fan_df.index) + list(fan_df.index[::-1]),
        y=list(fan_df["p90"]) + list(fan_df["p10"][::-1]),
        fill="toself", fillcolor="rgba(201,162,39,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Intervalo P10–P90", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=list(fan_df.index) + list(fan_df.index[::-1]),
        y=list(fan_df["p75"]) + list(fan_df["p25"][::-1]),
        fill="toself", fillcolor="rgba(201,162,39,0.22)",
        line=dict(color="rgba(0,0,0,0)"), name="Intervalo P25–P75", showlegend=True,
    ))
    fig.add_trace(go.Scatter(x=fan_df.index, y=fan_df["p50"], mode="lines",
                              line=dict(color=THEME["accent"], width=2.6), name="Mediana (Base)"))
    fig.add_trace(go.Scatter(x=[last_date], y=[last_price], mode="markers",
                              marker=dict(color=THEME["text"], size=8), name="Preço Atual"))
    return _apply_theme(fig, title, height=480)


def bar_chart(categories: list[str], values: list[float], title: str = "",
              positive_negative: bool = True) -> go.Figure:
    colors = [THEME["positive"] if v >= 0 else THEME["negative"] for v in values] if positive_negative else THEME["accent"]
    fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors)])
    return _apply_theme(fig, title)


def radar_chart(categories: list[str], values: list[float], name: str = "", title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                                   fill="toself", name=name, line=dict(color=THEME["accent"])))
    fig.update_layout(polar=dict(
        bgcolor=THEME["surface"],
        radialaxis=dict(gridcolor=THEME["border"], color=THEME["text_muted"]),
        angularaxis=dict(gridcolor=THEME["border"], color=THEME["text"]),
    ))
    return _apply_theme(fig, title)


def treemap_chart(labels: list[str], parents: list[str], values: list[float],
                   title: str = "") -> go.Figure:
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        marker=dict(colorscale=[[0, THEME["negative"]], [0.5, THEME["surface"]], [1, THEME["positive"]]]),
        textfont=dict(color=THEME["text"]),
    ))
    return _apply_theme(fig, title, height=480)


def histogram_chart(values, title: str = "", x_title: str = "") -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=values, marker_color=THEME["accent"], opacity=0.85)])
    fig.update_xaxes(title_text=x_title)
    return _apply_theme(