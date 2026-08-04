"""
Charts — Estilo macro terminal (Azuria / Bloomberg slide)
==========================================================
Fundo preto puro, sem grid, series de alto contraste e anotacoes
internas (ultimo preco, extremos, notas de leitura).

NAO altera calculos — apenas apresentacao visual.

BUGFIX (título x nota sobrepostos): antes, o título do gráfico
(`layout["title"]`) e a nota automática (`_note()`, uma annotation
solta em xref/yref="paper", y=0.98) ocupavam a MESMA região no canto
superior-esquerdo — resultado: duas caixas de texto competindo pelo
mesmo espaço, uma "riscando" a outra visualmente. Corrigido fundindo
título + nota num único bloco de título com subtítulo (`<br>` +
`<span>` menor e mais claro logo abaixo do título principal) — o
Plotly reserva o espaço verticalmente sozinho, então não existe mais
como colidir, em nenhuma altura de gráfico. Padrão usado por
dashboards financeiros reais (título em negrito + linha de contexto
em cinza logo abaixo, como uma legenda única).
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config.settings import THEME

_C = {
    "bg": "#000000",
    "paper": "#000000",
    "text": "#f0f0f0",
    "muted": "#8a8a8a",
    "gold": "#f5d76e",
    "blue": "#4da6ff",
    "red": "#ff4d5a",
    "white": "#ffffff",
    "teal": "#2dd4bf",
    "orange": "#f0a030",
    "violet": "#a78bfa",
    "rose": "#fb7185",
    "green": "#00c853",
}

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=_C["paper"],
    plot_bgcolor=_C["bg"],
    font=dict(color=_C["text"], family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
    margin=dict(l=48, r=56, t=64, b=40),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#111111", bordercolor="#333333",
                    font=dict(family="Inter, sans-serif", size=12, color=_C["text"]), align="left"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, orientation="h", y=1.14, x=0,
                font=dict(size=11, color=_C["muted"]), itemsizing="constant", tracegroupgap=10),
    xaxis=dict(showgrid=False, zeroline=False, showline=False,
               tickfont=dict(size=11, color=_C["muted"]), title_font=dict(size=11, color=_C["muted"]),
               showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
               spikecolor="rgba(255,255,255,0.25)", spikedash="dot"),
    yaxis=dict(showgrid=False, zeroline=False, showline=False, side="right",
               tickfont=dict(size=11, color=_C["muted"]), title_font=dict(size=11, color=_C["muted"]),
               showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
               spikecolor="rgba(255,255,255,0.25)", spikedash="dot"),
    transition=dict(duration=200, easing="cubic-in-out"),
)


def _title_block(title: Optional[str], subtitle: Optional[str]) -> Optional[dict]:
    """Monta o objeto de título único (título + subtítulo opcional na
    mesma peça, via <br> + <span>). Substitui o par título/annotation
    solta que colidia visualmente."""
    if not title and not subtitle:
        return None
    text = f"<b>{title}</b>" if title else ""
    if subtitle:
        sep = "<br>" if title else ""
        text += f"{sep}<span style='font-size:11px;color:{_C['muted']}'>{subtitle}</span>"
    return dict(text=text, font=dict(size=15, color=_C["text"], family="Inter, sans-serif"),
                x=0.0, xanchor="left", y=0.97, yanchor="top", pad=dict(b=12, t=4))


def _apply_theme(fig: go.Figure, title: Optional[str] = None, height: int = 440,
                  subtitle: Optional[str] = None) -> go.Figure:
    layout = dict(_LAYOUT_DEFAULTS)
    title_block = _title_block(title, subtitle)
    if title_block:
        layout["title"] = title_block
    fig.update_layout(**layout, height=height, uirevision="cit")
    for tr in fig.data:
        if getattr(tr, "type", None) == "scatter" and hasattr(tr, "line") and tr.line is not None:
            w = getattr(tr.line, "width", None)
            if w in (None, 1, 1.0, 1.6, 1.8, 2.0):
                try:
                    tr.line.width = 2.0
                except Exception:
                    pass
        elif getattr(tr, "type", None) == "bar":
            try:
                if getattr(tr, "marker", None) is not None:
                    tr.marker.line = dict(width=0)
            except Exception:
                pass
    return fig


def _label_last(fig: go.Figure, series: pd.Series, color: str, fmt: str = "{:.2f}") -> None:
    if series is None or len(series) == 0:
        return
    s = series.dropna()
    if len(s) == 0:
        return
    fig.add_trace(go.Scatter(
        x=[s.index[-1]], y=[float(s.iloc[-1])], mode="markers+text",
        marker=dict(size=7, color=color, line=dict(width=1.5, color=_C["bg"])),
        text=[fmt.format(float(s.iloc[-1]))], textposition="middle right",
        textfont=dict(size=11, color=color, family="JetBrains Mono, monospace"),
        showlegend=False, hoverinfo="skip",
    ))


def candlestick_chart(df: pd.DataFrame, title: str = "", note: str = "") -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=_C["green"], decreasing_line_color=_C["red"],
        increasing_fillcolor=_C["green"], decreasing_fillcolor=_C["red"], name="OHLC",
    )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    if not note and len(df) > 0:
        last = float(df["Close"].iloc[-1])
        chg = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100 if len(df) > 1 else 0
        note = "Close {:.2f}  ·  periodo {:+.1f}%".format(last, chg)
    fig = _apply_theme(fig, title, subtitle=note)
    return fig


def line_chart(series_dict: Dict[str, pd.Series], title: str = "", y_title: str = "",
               note: str = "", mark_extremes: bool = False) -> go.Figure:
    fig = go.Figure()
    palette = [_C["gold"], _C["blue"], _C["green"], _C["teal"],
               _C["orange"], _C["violet"], _C["red"], _C["rose"]]
    first = None
    for i, (name, s) in enumerate(series_dict.items()):
        color = palette[i % len(palette)]
        if first is None:
            first = s
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=2.0),
            hovertemplate="%{y:.2f}<extra>%{fullData.name}</extra>",
        ))
        _label_last(fig, s, color)
    fig.update_yaxes(title_text=y_title)
    if not note and first is not None and len(first.dropna()) > 1:
        s0 = list(series_dict.values())[0].dropna()
        chg = (float(s0.iloc[-1]) / float(s0.iloc[0]) - 1.0) * 100
        note = "Serie principal em {} de {:+.1f}% no periodo".format(
            "alta" if chg >= 0 else "baixa", chg)
    fig = _apply_theme(fig, title, subtitle=note)
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Matriz de Correlacao",
                        note: str = "") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale=[[0, _C["red"]], [0.5, "#1a1a1a"], [1, _C["green"]]],
        zmid=0, zmin=-1, zmax=1, text=corr_matrix.round(2).values, texttemplate="%{text}",
        textfont=dict(size=11, color=_C["text"]),
        colorbar=dict(title="rho", tickfont=dict(color=_C["muted"])),
    ))
    if not note:
        cm = corr_matrix.copy()
        np.fill_diagonal(cm.values, 0.0)
        if cm.size > 0:
            flat = cm.abs().stack()
            if len(flat):
                pair = flat.idxmax()
                note = "Maior |corr|: {} x {} = {:+.2f}".format(pair[0], pair[1], cm.loc[pair])
    fig = _apply_theme(fig, title, height=max(360, 28 * len(corr_matrix)), subtitle=note)
    return fig


def fan_chart(fan_df: pd.DataFrame, last_price: float, last_date: pd.Timestamp,
              title: str = "Forecast — Cenarios Probabilisticos", note: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fan_df.index) + list(fan_df.index[::-1]),
        y=list(fan_df["p90"]) + list(fan_df["p10"][::-1]),
        fill="toself", fillcolor="rgba(245,215,110,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="P10-P90", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=list(fan_df.index) + list(fan_df.index[::-1]),
        y=list(fan_df["p75"]) + list(fan_df["p25"][::-1]),
        fill="toself", fillcolor="rgba(245,215,110,0.20)",
        line=dict(color="rgba(0,0,0,0)"), name="P25-P75", showlegend=True,
    ))
    fig.add_trace(go.Scatter(x=fan_df.index, y=fan_df["p50"], mode="lines",
                              line=dict(color=_C["gold"], width=2.4), name="Mediana (Base)"))
    fig.add_trace(go.Scatter(x=[last_date], y=[last_price], mode="markers",
                              marker=dict(color=_C["white"], size=9, line=dict(width=1, color=_C["gold"])),
                              name="Preco Atual"))
    if not note:
        med = float(fan_df["p50"].iloc[-1]) if len(fan_df) else last_price
        upside = (med / last_price - 1.0) * 100 if last_price else 0
        note = "Mediana terminal {:+.1f}% vs preco atual · leia o lag, nao o ruido".format(upside)
    fig = _apply_theme(fig, title, height=480, subtitle=note)
    return fig


def bar_chart(categories: List[str], values: List[float], title: str = "",
              positive_negative: bool = True, note: str = "") -> go.Figure:
    colors = ([_C["green"] if v >= 0 else _C["red"] for v in values]
              if positive_negative else _C["gold"])
    fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors)])
    fig = _apply_theme(fig, title, subtitle=note)
    return fig


def radar_chart(categories: List[str], values: List[float], name: str = "",
                title: str = "", note: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", name=name, line=dict(color=_C["gold"], width=2),
        fillcolor="rgba(245,215,110,0.18)",
    ))
    fig.update_layout(polar=dict(
        bgcolor=_C["bg"],
        radialaxis=dict(gridcolor="#222222", color=_C["muted"], showline=False),
        angularaxis=dict(gridcolor="#222222", color=_C["text"]),
    ))
    fig = _apply_theme(fig, title, subtitle=note)
    return fig


def treemap_chart(labels: List[str], parents: List[str], values: List[float],
                   title: str = "", note: str = "") -> go.Figure:
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        marker=dict(colorscale=[[0, _C["red"]], [0.5, "#1a1a1a"], [1, _C["green"]]]),
        textfont=dict(color=_C["text"]),
    ))
    fig = _apply_theme(fig, title, height=480, subtitle=note)
    return fig


def histogram_chart(values, title: str = "", x_title: str = "", note: str = "") -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=values, marker_color=_C["gold"], opacity=0.85)])
    fig.update_xaxes(title_text=x_title)
    if not note:
        arr = np.asarray(list(values), dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            note = "Media {:+.3f}  ·  desvio {:.3f}".format(float(np.mean(arr)), float(np.std(arr)))
    fig = _apply_theme(fig, title, height=400, subtitle=note)
    return fig


def var_breach_chart(returns: pd.Series, var_series: pd.Series, breaches: pd.Series,
                      title: str = "Backtesting de VaR — Retornos vs. Limite Previsto",
                      note: str = "") -> go.Figure:
    idx = var_series.index
    rets_aligned = returns.reindex(idx)
    fig = go.Figure()
    bar_colors = [_C["red"] if b else _C["muted"] for b in breaches.reindex(idx).fillna(False)]
    fig.add_trace(go.Bar(x=idx, y=rets_aligned, name="Retorno diario", marker_color=bar_colors, opacity=0.75))
    fig.add_trace(go.Scatter(x=idx, y=-var_series, mode="lines", name="- VaR previsto",
                              line=dict(color=_C["orange"], width=1.8, dash="dot")))
    breach_dates = breaches[breaches].index.intersection(idx)
    n_breach = len(breach_dates)
    if n_breach > 0:
        fig.add_trace(go.Scatter(
            x=breach_dates, y=rets_aligned.reindex(breach_dates), mode="markers",
            name="Excecoes ({})".format(n_breach),
            marker=dict(color=_C["red"], size=8, symbol="x", line=dict(width=1, color=_C["white"])),
        ))
    fig.update_yaxes(title_text="Retorno diario", tickformat=".1%")
    if not note:
        rate = (n_breach / max(len(idx), 1)) * 100
        note = "{} excecoes ({:.1f}% dos dias) · VaR deve conter ~95%".format(n_breach, rate)
    fig = _apply_theme(fig, title, height=420, subtitle=note)
    return fig


def regime_price_chart(close: pd.Series, viterbi_states: pd.Series,
                        title: str = "Preco com Regimes de Volatilidade (HMM)",
                        note: str = "") -> go.Figure:
    fig = go.Figure()
    states = viterbi_states.reindex(close.index).ffill().bfill()
    shapes = []
    if len(states) > 0:
        run_start, run_state = states.index[0], states.iloc[0]
        for i in range(1, len(states)):
            if states.iloc[i] != run_state:
                if run_state == 1:
                    shapes.append(dict(type="rect", xref="x", yref="paper",
                                       x0=run_start, x1=states.index[i], y0=0, y1=1,
                                       fillcolor=_C["red"], opacity=0.12, line_width=0))
                run_start, run_state = states.index[i], states.iloc[i]
        if run_state == 1:
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=run_start, x1=states.index[-1], y0=0, y1=1,
                               fillcolor=_C["red"], opacity=0.12, line_width=0))
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines",
                              line=dict(color=_C["gold"], width=2.0), name="Preco"))
    _label_last(fig, close, _C["gold"])
    fig = _apply_theme(fig, title, height=420,
                        subtitle=note or "Faixas vermelhas = regime de alta vol (HMM) · preco reage com lag")
    fig.update_layout(shapes=shapes)
    return fig


def regime_probability_chart(state_probs: pd.DataFrame,
                              title: str = "Probabilidade Suavizada de Regime",
                              note: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=state_probs.index, y=state_probs["prob_baixa_vol"], mode="lines",
        name="P(Baixa Vol)", stackgroup="one",
        line=dict(width=0.5, color=_C["green"]), fillcolor="rgba(0,200,83,0.40)",
    ))
    fig.add_trace(go.Scatter(
        x=state_probs.index, y=state_probs["prob_alta_vol"], mode="lines",
        name="P(Alta Vol)", stackgroup="one",
        line=dict(width=0.5, color=_C["red"]), fillcolor="rgba(255,77,90,0.40)",
    ))
    fig.update_yaxes(title_text="Probabilidade", range=[0, 1], tickformat=".0%")
    if not note:
        last_hv = float(state_probs["prob_alta_vol"].iloc[-1]) if len(state_probs) else 0
        note = "P(alta vol) atual = {:.0%} · regime e caminho, nao ponto".format(last_hv)
    fig = _apply_theme(fig, title, height=300, subtitle=note)
    return fig


def spread_chart(spread: pd.Series, mean: float, std: float,
                  title: str = "Spread do Par (Engle-Granger)", note: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, mode="lines",
                              line=dict(color=_C["blue"], width=2.0), name="Spread"))
    for k, dash, opacity in [(1, "dot", 0.55), (2, "dash", 0.4)]:
        fig.add_hline(y=mean + k * std, line=dict(color=_C["gold"], width=1, dash=dash), opacity=opacity)
        fig.add_hline(y=mean - k * std, line=dict(color=_C["gold"], width=1, dash=dash), opacity=opacity)
    fig.add_hline(y=mean, line=dict(color=_C["muted"], width=1))
    _label_last(fig, spread, _C["blue"], fmt="{:.3f}")
    if not note:
        last = float(spread.dropna().iloc[-1]) if len(spread.dropna()) else mean
        z = (last - mean) / std if std else 0
        note = "Spread atual z = {:+.2f} sigma · bandas +/-1s / +/-2s em dourado".format(z)
    fig = _apply_theme(fig, title, height=380, subtitle=note)
    return fig


def kalman_beta_chart(beta_series: pd.Series, static_beta: Optional[float] = None,
                       title: str = "Hedge Ratio Dinamico (Kalman Filter)",
                       note: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta_series.index, y=beta_series.values, mode="lines",
                              line=dict(color=_C["blue"], width=2.0), name="beta Kalman"))
    if static_beta is not None:
        fig.add_hline(y=static_beta, line=dict(color=_C["gold"], width=1.4, dash="dash"))
        fig.add_annotation(text="beta estatico (EG) = {:.3f}".format(static_beta),
                           xref="paper", x=0.01, y=static_beta, yref="y",
                           showarrow=False, font=dict(color=_C["gold"], size=11))
    _label_last(fig, beta_series, _C["blue"], fmt="{:.3f}")
    fig = _apply_theme(fig, title, height=380,
                        subtitle=note or "beta dinamico vs estatico · hedge ratio muda com o regime")
    return fig


def zscore_chart(z_score: pd.Series,
                  title: str = "Z-Score do Spread — Sinal de Pairs Trading",
                  note: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z_score.index, y=z_score.values, mode="lines",
                              line=dict(color=_C["white"], width=1.8), name="Z-Score"))
    ymax = max(float(np.nanmax(z_score.values)) if len(z_score) else 3, 3)
    ymin = min(float(np.nanmin(z_score.values)) if len(z_score) else -3, -3)
    fig.add_hrect(y0=2, y1=ymax, fillcolor=_C["red"], opacity=0.08, line_width=0)
    fig.add_hrect(y0=ymin, y1=-2, fillcolor=_C["green"], opacity=0.08, line_width=0)
    fig.add_hline(y=0, line=dict(color=_C["muted"], width=1))
    fig.add_hline(y=2, line=dict(color=_C["red"], width=1, dash="dot"), opacity=0.7)
    fig.add_hline(y=-2, line=dict(color=_C["green"], width=1, dash="dot"), opacity=0.7)
    _label_last(fig, z_score, _C["white"], fmt="{:+.2f}")
    if not note:
        last = float(z_score.dropna().iloc[-1]) if len(z_score.dropna()) else 0
        if last >= 2:
            note = "Z = {:+.2f} · zona de venda do spread (|z|>2)".format(last)
        elif last <= -2:
            note = "Z = {:+.2f} · zona de compra do spread (|z|>2)".format(last)
        else:
            note = "Z = {:+.2f} · dentro da banda · sem sinal extremo".format(last)
    fig = _apply_theme(fig, title, height=380, subtitle=note)
    return fig


def risk_return_scatter(df: pd.DataFrame, title: str = "Risco vs. Retorno",
                         note: str = "") -> go.Figure:
    sector_colors = {"Energia": _C["orange"], "Metais": _C["gold"],
                     "Agricultura": _C["green"], "Brasil": _C["blue"]}
    fig = go.Figure()
    for sector in df["sector"].unique():
        sub = df[df["sector"] == sector]
        fig.add_trace(go.Scatter(
            x=sub["vol"], y=sub["sharpe"], mode="markers+text",
            text=sub["name"], textposition="top center",
            textfont=dict(size=10, color=_C["muted"]), name=sector,
            marker=dict(size=(sub["momentum"].abs() * 300).clip(lower=10, upper=42),
                        color=sector_colors.get(sector, _C["gold"]),
                        opacity=0.85, line=dict(width=1, color=_C["bg"])),
        ))
    fig.add_hline(y=0, line=dict(color=_C["muted"], width=1, dash="dot"), opacity=0.5)
    fig.update_xaxes(title_text="Volatilidade Anualizada", tickformat=".0%")
    fig.update_yaxes(title_text="Sharpe Ratio")
    if not note and len(df) and "sharpe" in df.columns:
        best = df.loc[df["sharpe"].idxmax()]
        note = "Melhor Sharpe: {} ({:+.2f}) · tamanho = |momentum|".format(
            best["name"], float(best["sharpe"]))
    fig = _apply_theme(fig, title, height=480, subtitle=note)
    return fig


def efficient_frontier_chart(frontier: pd.DataFrame, method_points: list,
                              title: str = "Fronteira Eficiente (anualizada)",
                              note: str = "") -> go.Figure:
    """Fronteira eficiente (linha) + pontos de cada método de otimização
    (Max Sharpe, Min Variance, Risk Parity, ...) sobrepostos, com cores
    100% da paleta _C do arquivo (nada de azul padrão do Plotly)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["volatility"], y=frontier["return"], mode="lines",
        name="Fronteira", line=dict(color=_C["gold"], width=2.2),
    ))
    palette = [_C["green"], _C["red"], _C["blue"], _C["teal"], _C["orange"], _C["violet"]]
    for i, pt in enumerate(method_points):
        fig.add_trace(go.Scatter(
            x=[pt["vol"]], y=[pt["ret"]], mode="markers+text",
            name=pt["method"], text=[pt["method"]], textposition="top center",
            textfont=dict(size=9, color=_C["muted"]),
            marker=dict(size=11, color=palette[i % len(palette)],
                        line=dict(width=1.2, color=_C["bg"])),
        ))
    fig.update_xaxes(title_text="Volatilidade Anualizada", tickformat=".0%")
    fig.update_yaxes(title_text="Retorno Esperado Anualizado", tickformat=".0%")
    fig = _apply_theme(fig, title, height=480, subtitle=note)
    return fig
