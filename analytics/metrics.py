"""
Analytics — Métricas de Série de Preço (v4.4.1)
==================================================
Funções puras (sem estado, sem I/O) sobre pd.Series de preços de fechamento.

CHANGELOG v4.4.1:
- Corrigido FutureWarning em daily_returns: adicionado fill_method=None
  para evitar depreciação futura.
- Proteção extra em max_drawdown para séries vazias.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.Series) -> pd.Series:
    # Correção: fill_method=None elimina o FutureWarning
    return close.pct_change(fill_method=None).dropna()


def pct_change_over(close: pd.Series, days: int) -> float | None:
    """Variação percentual nos últimos `days` pregões."""
    if len(close) < days + 1:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def ytd_return(close: pd.Series) -> float | None:
    this_year = close[close.index.year == close.index[-1].year]
    if this_year.empty:
        return None
    return float(close.iloc[-1] / this_year.iloc[0] - 1)


def annualized_volatility(close: pd.Series, window: int | None = None) -> float:
    rets = daily_returns(close)
    if window:
        rets = rets.tail(window)
    return float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(close: pd.Series, risk_free_annual: float = 0.045, window: int | None = None) -> float:
    rets = daily_returns(close)
    if window:
        rets = rets.tail(window)
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily
    denom = excess.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(excess.mean() / denom * np.sqrt(TRADING_DAYS))


def sortino_ratio(close: pd.Series, risk_free_annual: float = 0.045, window: int | None = None) -> float:
    rets = daily_returns(close)
    if window:
        rets = rets.tail(window)
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily
    downside = excess[excess < 0]
    denom = downside.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(excess.mean() / denom * np.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    cum_max = close.cummax()
    drawdown = close / cum_max - 1
    return float(drawdown.min())


def calmar_ratio(close: pd.Series, window: int | None = None) -> float:
    rets = daily_returns(close)
    if window:
        rets = rets.tail(window)
        px = close.tail(window + 1)
    else:
        px = close
    ann_return = (1 + rets.mean()) ** TRADING_DAYS - 1
    mdd = abs(max_drawdown(px))
    if mdd == 0:
        return 0.0
    return float(ann_return / mdd)


def beta(asset_close: pd.Series, benchmark_close: pd.Series) -> float:
    a = daily_returns(asset_close)
    b = daily_returns(benchmark_close)
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return float("nan")
    cov = joined.cov().iloc[0, 1]
    var_b = joined.iloc[:, 1].var()
    if var_b == 0:
        return float("nan")
    return float(cov / var_b)


def momentum_score(close: pd.Series) -> float:
    """Combinação normalizada de retornos 1M/3M/6M/12M — sinal de tendência."""
    windows = [21, 63, 126, 252]
    scores = []
    for w in windows:
        r = pct_change_over(close, w)
        if r is not None:
            scores.append(r)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def trend_label(close: pd.Series, short: int = 20, long: int = 100) -> str:
    if len(close) < long:
        return "indefinida"
    sma_short = close.tail(short).mean()
    sma_long = close.tail(long).mean()
    if sma_short > sma_long * 1.01:
        return "alta"
    if sma_short < sma_long * 0.99:
        return "baixa"
    return "lateral"


def cumulative_return_series(close: pd.Series) -> pd.Series:
    rets = daily_returns(close)
    return (1 + rets).cumprod() - 1


def summary_row(close: pd.Series, benchmark_close: pd.Series | None = None,
                 risk_free_annual: float = 0.045, window: int | None = None) -> dict:
    """Monta a linha completa de métricas para o Dashboard Global.
    
    CHANGELOG v4.4.0: adicionado parâmetro `window` para filtrar dados
    antes do cálculo das métricas (usado pelo Risk Analytics).
    """
    # Aplica janela se especificada
    if window is not None and window > 0:
        close_windowed = close.tail(window)
    else:
        close_windowed = close
    
    row = {
        "last_price": float(close_windowed.iloc[-1]) if len(close_windowed) else None,
        "chg_1d": pct_change_over(close_windowed, 1),
        "chg_1w": pct_change_over(close_windowed, 5),
        "chg_1m": pct_change_over(close_windowed, 21),
        "chg_ytd": ytd_return(close_windowed),
        "vol_annual": annualized_volatility(close_windowed, window=63),
        "sharpe": sharpe_ratio(close_windowed, risk_free_annual, window=252),
        "sortino": sortino_ratio(close_windowed, risk_free_annual, window=252),
        "max_drawdown": max_drawdown(close_windowed.tail(252)),
        "calmar": calmar_ratio(close_windowed, window=252),
        "momentum": momentum_score(close_windowed),
        "trend": trend_label(close_windowed),
    }
    if benchmark_close is not None:
        row["beta"] = beta(close_windowed, benchmark_close)
    return row