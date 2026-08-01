"""
Analytics — Sinais Leves para Monitoramento Multi-Ativo
============================================================
O Risk Analytics (aba Regimes) roda um HMM completo por ativo — preciso,
mas caro demais para rodar em ~20 ativos simultaneamente numa dashboard.
Este módulo dá um proxy rápido e barato do mesmo tipo de informação
(regime de volatilidade, extremos de retorno), adequado para telas de
visão geral. Para análise aprofundada de um ativo específico, a
recomendação (linkada na UI) é ir para a aba Regimes do Risk Analytics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from analytics.metrics import daily_returns


def volatility_percentile(close: pd.Series, short_window: int = 21, lookback: int = 252) -> float | None:
    """Percentil da volatilidade realizada recente (short_window) em
    relação à distribuição histórica de volatilidade rolante do próprio
    ativo (lookback). 0 = vol atual é a mais baixa do período; 1 = a mais
    alta. Proxy rápido de "em que regime de vol o ativo está agora"."""
    rets = daily_returns(close)
    if len(rets) < short_window + 20:
        return None
    rolling_vol = rets.rolling(short_window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna().tail(lookback)
    if len(rolling_vol) < 20:
        return None
    current_vol = rolling_vol.iloc[-1]
    percentile = float((rolling_vol < current_vol).mean())
    return percentile


def vol_regime_label(percentile: float | None) -> str:
    if percentile is None:
        return "N/D"
    if percentile >= 0.80:
        return "🔴 Vol. Elevada"
    if percentile <= 0.20:
        return "🟢 Vol. Baixa"
    return "⚪ Vol. Normal"


def return_zscore(close: pd.Series, window: int = 63) -> float | None:
    """Z-score do último retorno diário vs. a distribuição recente de
    retornos — sinaliza movimentos atípicos (extremos) do dia."""
    rets = daily_returns(close)
    if len(rets) < window + 5:
        return None
    recent = rets.tail(window)
    std = recent.std(ddof=1)
    if std == 0 or np.isnan(std):
        return None
    return float((rets.iloc[-1] - recent.mean()) / std)


def momentum_label(momentum: float, threshold: float = 0.05) -> str:
    if momentum > threshold:
        return "📈 Momentum Forte (alta)"
    if momentum < -threshold:
        return "📉 Momentum Forte (baixa)"
    return "➖ Momentum Neutro"


def build_signal_row(close: pd.Series, momentum: float) -> dict:
    """Monta a linha de sinais para um ativo — usada na tabela do painel
    consolidado do Dashboard Global."""
    vol_pct = volatility_percentile(close)
    z = return_zscore(close)
    return {
        "vol_percentile": vol_pct,
        "vol_regime": vol_regime_label(vol_pct),
        "return_zscore": z,
        "extreme_move": bool(z is not None and abs(z) > 2),
        "momentum_label": momentum_label(momentum),
    }