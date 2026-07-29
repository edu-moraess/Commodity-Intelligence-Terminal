"""
Analytics — Risk Management
==============================
VaR (histórico e paramétrico), CVaR/Expected Shortfall, stress testing por
choque percentual, tracking error e information ratio.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as _stats  # scipy é dependência transitiva de sklearn; fallback abaixo se ausente

from analytics.metrics import daily_returns


def historical_var(close: pd.Series, confidence: float = 0.95, window: int = 252) -> float:
    rets = daily_returns(close).tail(window)
    if rets.empty:
        return float("nan")
    return float(-np.percentile(rets, (1 - confidence) * 100))


def historical_cvar(close: pd.Series, confidence: float = 0.95, window: int = 252) -> float:
    rets = daily_returns(close).tail(window)
    if rets.empty:
        return float("nan")
    var_threshold = np.percentile(rets, (1 - confidence) * 100)
    tail = rets[rets <= var_threshold]
    if tail.empty:
        return float(-var_threshold)
    return float(-tail.mean())


def parametric_var(close: pd.Series, confidence: float = 0.95, window: int = 252) -> float:
    rets = daily_returns(close).tail(window)
    if rets.empty:
        return float("nan")
    mu, sigma = rets.mean(), rets.std(ddof=1)
    try:
        z = _stats.norm.ppf(1 - confidence)
    except Exception:  # noqa: BLE001 — fallback sem scipy
        z = -1.645 if confidence == 0.95 else -2.326
    return float(-(mu + z * sigma))


def stress_test(close: pd.Series, shocks_pct: list[float] | None = None) -> pd.DataFrame:
    """Aplica choques percentuais instantâneos ao último preço e reporta o
    novo nível e a perda/ganho absoluto."""
    if shocks_pct is None:
        shocks_pct = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
    last = float(close.iloc[-1])
    rows = []
    for shock in shocks_pct:
        new_price = last * (1 + shock)
        rows.append({
            "choque": f"{shock:+.0%}",
            "preco_atual": round(last, 2),
            "preco_stress": round(new_price, 2),
            "variacao_absoluta": round(new_price - last, 2),
        })
    return pd.DataFrame(rows)


def tracking_error(asset_close: pd.Series, benchmark_close: pd.Series, window: int = 252) -> float:
    a = daily_returns(asset_close).tail(window)
    b = daily_returns(benchmark_close).tail(window)
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if joined.empty:
        return float("nan")
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    return float(diff.std(ddof=1) * np.sqrt(252))


def information_ratio(asset_close: pd.Series, benchmark_close: pd.Series, window: int = 252) -> float:
    a = daily_returns(asset_close).tail(window)
    b = daily_returns(benchmark_close).tail(window)
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if joined.empty:
        return float("nan")
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    te = diff.std(ddof=1)
    if te == 0:
        return 0.0
    return float(diff.mean() / te * np.sqrt(252))


def risk_summary(close: pd.Series, confidence: float = 0.95, window: int = 252) -> dict:
    return {
        "var_historico": historical_var(close, confidence, window),
        "var_parametrico": parametric_var(close, confidence, window),
        "cvar": historical_cvar(close, confidence, window),
        "confianca": confidence,
        "janela_dias": window,
    }