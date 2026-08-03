"""
Ticker Tape — Fita de Cotações Estilo Bloomberg/Investing.com
==================================================================
Renderiza uma fita horizontal rolante com preço + variação de um
conjunto de ativos, no topo de cada página. Puramente CSS (@keyframes),
sem JS externo — funciona em qualquer navegador sem dependências.

Usa `st.markdown(..., unsafe_allow_html=True)` uma única vez; a
animação continua rodando no cliente sem re-render do Streamlit.
"""

from __future__ import annotations
import streamlit as st

from config.settings import THEME
from data.data_manager import PriceData
from analytics.metrics import pct_change_over


def _format_item(name: str, ticker: str, pdat: PriceData) -> str:
    close = pdat.df["Close"]
    if close.empty:
        return ""
    last_price = float(close.iloc[-1])
    chg = pct_change_over(close, 1)
    chg_pct = f"{chg:+.2%}" if chg is not None else "—"

    if chg is None:
        color = THEME["text_muted"]
        arrow = ""
    elif chg > 0:
        color = THEME["positive"]
        arrow = "▲"
    elif chg < 0:
        color = THEME["negative"]
        arrow = "▼"
    else:
        color = THEME["text_muted"]
        arrow = "▪"

    synth_flag = " <span class='tt-synth'>●</span>" if pdat.is_synthetic else ""

    return f"""
    <span class="tt-item">
        <span class="tt-name">{name}</span>
        <span class="tt-price">{last_price:,.2f}</span>
        <span class="tt-chg" style="color:{color};">{arrow} {chg_pct}</span>{synth_flag}
    </span>
    """


def render_ticker_tape(price_data: dict[str, PriceData], assets: list) -> None:
    """Renderiza a fita rolante. `assets` é uma lista de objetos Asset
    (com .ticker e .name); `price_data` é o dict ticker -> PriceData já
    carregado (evita duplicar chamadas de rede/cache)."""
    items_html = "".join(
        _format_item(a.name, a.ticker, price_data[a.ticker])
        for a in assets if a.ticker in price_data and not price_data[a.ticker].df.empty
    )
    if not items_html:
        return

    track_html = items_html + items_html

    st.markdown(f"""
    <style>
        .tt-wrap {{
            background-color: {THEME['ticker_bg']};
            border-bottom: 1px solid {THEME['border']};
            overflow: hidden;
            white-space: nowrap;
            padding: 5px 0;
            margin: 0 0 0.65rem 0;
            width: 100%;
            border-radius: 6px;
            position: relative;
            z-index: 1;
        }}
        .tt-track {{
            display: inline-block;
            animation: tt-scroll 45s linear infinite;
            will-change: transform;
        }}
        .tt-wrap:hover .tt-track {{
            animation-play-state: paused;
        }}
        @keyframes tt-scroll {{
            0%   {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        .tt-item {{
            display: inline-flex;
            align-items: baseline;
            gap: 6px;
            padding: 0 16px;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 0.75rem;
            border-right: 1px solid {THEME['border']};
        }}
        .tt-name {{
            color: {THEME['text_muted']};
            font-weight: 500;
            letter-spacing: 0.02em;
        }}
        .tt-price {{
            color: {THEME['text']};
            font-weight: 600;
        }}
        .tt-chg {{
            font-weight: 600;
        }}
        .tt-synth {{
            color: {THEME['warning']};
            font-size: 0.55rem;
            vertical-align: super;
        }}
    </style>
    <div class="tt-wrap">
        <div class="tt-track">{track_html}</div>
    </div>
    """, unsafe_allow_html=True)