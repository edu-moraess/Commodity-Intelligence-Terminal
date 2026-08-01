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

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=THEME["surface"],
    plot_bgcolor=THEME["surface"],
    font=dict(color=THEME["text"], family="Inter, -apple-system, sans-serif", size=12),
    margin=dict(l=40, r=20, t=48, b=32),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08, x=0),
    xaxis=dict(gridcolor=THEME["border"], zerolinecolor=THEME["border"]),
    yaxis=dict(gridcolor=THEME["border"], zerolinecolor=THEME["border"]),
)


def _apply_theme(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    layout = dict(_LAYOUT_DEFAULTS)
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color=THEME["text"]))
    fig.update_layout(**layout, height=height)
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
    palette = [THEME["accent"], THEME["positive"], THEME["warning"], THEME["negative"], "#9b8afb", "#f7c948"]
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
        fill="toself", fillcolor="rgba(63,177,206,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Intervalo P10–P90", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=list(fan_df.index) + list(fan_df.index[::-1]),
        y=list(fan_df["p75"]) + list(fan_df["p25"][::-1]),
        fill="toself", fillcolor="rgba(63,177,206,0.22)",
        line=dict(color="rgba(0,0,0,0)"), name="Intervalo P25–P75", showlegend=True,
    ))
    fig.add_trace(go.Scatter(x=fan_df.index, y=fan_df["p50"], mode="lines",
                              line=dict(color=THEME["accent"], width=2.4), name="Mediana (Base)"))
    fig.add_trace(go.Scatter(x=[last_date], y=[last_price], mode="markers",
                              marker=dict(color=THEME["text"], size=8), name="Preço Atual"))
    return _apply_theme(fig, title, height=420)


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
    return _apply_theme(fig, title, height=420)


def histogram_chart(values, title: str = "", x_title: str = "") -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=values, marker_color=THEME["accent"], opacity=0.85)])
    fig.update_xaxes(title_text=x_title)
    return _apply_theme(fig, title)


def var_breach_chart(returns: pd.Series, var_series: pd.Series, breaches: pd.Series,
                      title: str = "Backtesting de VaR — Retornos vs. Limite Previsto") -> go.Figure:
    """Gráfico institucional padrão de backtest de VaR: retorno diário
    (barras), limite de VaR previsto (linha, invertido para o mesmo eixo
    de perdas) e marcadores nos dias de breach (exceção)."""
    idx = var_series.index
    rets_aligned = returns.reindex(idx)

    fig = go.Figure()
    bar_colors = [THEME["negative"] if b else THEME["text_muted"] for b in breaches.reindex(idx).fillna(False)]
    fig.add_trace(go.Bar(x=idx, y=rets_aligned, name="Retorno diário",
                          marker_color=bar_colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=idx, y=-var_series, mode="lines", name="− VaR previsto",
                              line=dict(color=THEME["warning"], width=1.6, dash="dot")))

    breach_dates = breaches[breaches].index.intersection(idx)
    if len(breach_dates) > 0:
        fig.add_trace(go.Scatter(
            x=breach_dates, y=rets_aligned.reindex(breach_dates), mode="markers",
            name=f"Exceções ({len(breach_dates)})",
            marker=dict(color=THEME["negative"], size=7, symbol="x", line=dict(width=1, color="white")),
        ))
    fig.update_yaxes(title_text="Retorno diário", tickformat=".1%")
    return _apply_theme(fig, title, height=420)


def regime_price_chart(close: pd.Series, viterbi_states: pd.Series,
                        title: str = "Preço com Regimes de Volatilidade (HMM)") -> go.Figure:
    """Preço com faixas verticais sombreadas indicando o regime (0 = baixa
    vol, 1 = alta vol) identificado pelo Viterbi — leitura visual imediata
    de quando o mercado esteve "calmo" vs "estressado"."""
    fig = go.Figure()

    states = viterbi_states.reindex(close.index).ffill().bfill()
    shapes = []
    if len(states) > 0:
        run_start = states.index[0]
        run_state = states.iloc[0]
        for i in range(1, len(states)):
            if states.iloc[i] != run_state:
                if run_state == 1:
                    shapes.append(dict(
                        type="rect", xref="x", yref="paper",
                        x0=run_start, x1=states.index[i], y0=0, y1=1,
                        fillcolor=THEME["negative"], opacity=0.12, line_width=0,
                    ))
                run_start = states.index[i]
                run_state = states.iloc[i]
        if run_state == 1:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=run_start, x1=states.index[-1], y0=0, y1=1,
                fillcolor=THEME["negative"], opacity=0.12, line_width=0,
            ))

    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines",
                              line=dict(color=THEME["accent"], width=1.8), name="Preço"))
    layout = dict(_LAYOUT_DEFAULTS)
    layout["shapes"] = shapes
    layout["title"] = dict(text=title, font=dict(size=15, color=THEME["text"]))
    fig.update_layout(**layout, height=420)
    return fig


def regime_probability_chart(state_probs: pd.DataFrame,
                              title: str = "Probabilidade Suavizada de Regime") -> go.Figure:
    """Área empilhada com P(baixa vol) e P(alta vol) ao longo do tempo —
    complementa o regime_price_chart mostrando a confiança do modelo,
    não só a classificação binária (Viterbi)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=state_probs.index, y=state_probs["prob_baixa_vol"], mode="lines",
        name="P(Baixa Volatilidade)", stackgroup="one",
        line=dict(width=0.5, color=THEME["positive"]), fillcolor="rgba(62,207,142,0.55)",
    ))
    fig.add_trace(go.Scatter(
        x=state_probs.index, y=state_probs["prob_alta_vol"], mode="lines",
        name="P(Alta Volatilidade)", stackgroup="one",
        line=dict(width=0.5, color=THEME["negative"]), fillcolor="rgba(229,72,77,0.55)",
    ))
    fig.update_yaxes(title_text="Probabilidade", range=[0, 1], tickformat=".0%")
    return _apply_theme(fig, title, height=280)


def spread_chart(spread: pd.Series, mean: float, std: float,
                  title: str = "Spread do Par (Engle-Granger)") -> go.Figure:
    """Spread estático (resíduo da regressão de cointegração) com a média
    histórica e bandas de ±1σ/±2σ — leitura visual de quando o spread
    está "esticado" o suficiente para sinalizar reversão à média."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, mode="lines",
                              line=dict(color=THEME["accent"], width=1.6), name="Spread"))
    for k, dash, opacity in [(1, "dot", 0.5), (2, "dash", 0.35)]:
        fig.add_hline(y=mean + k * std, line=dict(color=THEME["warning"], width=1, dash=dash), opacity=opacity)
        fig.add_hline(y=mean - k * std, line=dict(color=THEME["warning"], width=1, dash=dash), opacity=opacity)
    fig.add_hline(y=mean, line=dict(color=THEME["text_muted"], width=1))
    return _apply_theme(fig, title, height=360)


def kalman_beta_chart(beta_series: pd.Series, static_beta: float | None = None,
                       title: str = "Hedge Ratio Dinâmico (Kalman Filter)") -> go.Figure:
    """Hedge ratio (beta) variando no tempo via Kalman Filter, com
    referência opcional ao hedge ratio estático (Engle-Granger, full
    sample) para comparação direta."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta_series.index, y=beta_series.values, mode="lines",
                              line=dict(color=THEME["accent"], width=1.8), name="β (Kalman, dinâmico)"))
    if static_beta is not None:
        fig.add_hline(y=static_beta, line=dict(color=THEME["warning"], width=1.4, dash="dash"))
        fig.add_annotation(text=f"β estático (EG) = {static_beta:.3f}", xref="paper", x=0.01, y=static_beta,
                            yref="y", showarrow=False, font=dict(color=THEME["warning"], size=11))
    return _apply_theme(fig, title, height=340)


def zscore_chart(z_score: pd.Series, title: str = "Z-Score do Spread — Sinal de Pairs Trading") -> go.Figure:
    """Z-score do spread do Kalman Filter com zonas de entrada (|z|>2) e
    saída (z≈0) — o sinal clássico de arbitragem estatística em pairs."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z_score.index, y=z_score.values, mode="lines",
                              line=dict(color=THEME["accent"], width=1.6), name="Z-Score"))
    fig.add_hrect(y0=2, y1=max(float(np.nanmax(z_score.values)) if len(z_score) else 3, 3),
                  fillcolor=THEME["negative"], opacity=0.08, line_width=0)
    fig.add_hrect(y0=min(float(np.nanmin(z_score.values)) if len(z_score) else -3, -3), y1=-2,
                  fillcolor=THEME["positive"], opacity=0.08, line_width=0)
    fig.add_hline(y=0, line=dict(color=THEME["text_muted"], width=1))
    fig.add_hline(y=2, line=dict(color=THEME["negative"], width=1, dash="dot"), opacity=0.6)
    fig.add_hline(y=-2, line=dict(color=THEME["positive"], width=1, dash="dot"), opacity=0.6)
    return _apply_theme(fig, title, height=340)


def risk_return_scatter(df: pd.DataFrame, title: str = "Risco vs. Retorno") -> go.Figure:
    """Bubble chart institucional padrão: eixo X = volatilidade anualizada,
    eixo Y = Sharpe (ou retorno), tamanho da bolha = |momentum|, cor por
    setor. `df` precisa ter as colunas: name, sector, vol, sharpe, momentum.
    """
    sector_colors = {
        "Energia": THEME["warning"], "Metais": THEME["accent"],
        "Agricultura": THEME["positive"], "Brasil": "#9b8afb",
    }
    fig = go.Figure()
    for sector in df["sector"].unique():
        sub = df[df["sector"] == sector]
        fig.add_trace(go.Scatter(
            x=sub["vol"], y=sub["sharpe"], mode="markers+text",
            text=sub["name"], textposition="top center",
            textfont=dict(size=9, color=THEME["text_muted"]),
            name=sector,
            marker=dict(
                size=(sub["momentum"].abs() * 300).clip(lower=8, upper=40),
                color=sector_colors.get(sector, THEME["accent"]),
                opacity=0.75, line=dict(width=1, color=THEME["surface"]),
            ),
        ))
    fig.add_hline(y=0, line=dict(color=THEME["text_muted"], width=1, dash="dot"), opacity=0.5)
    fig.update_xaxes(title_text="Volatilidade Anualizada", tickformat=".0%")
    fig.update_yaxes(title_text="Sharpe Ratio")
    return _apply_theme(fig, title, height=460)