"""
Analytics — Risk Management Institucional
=========================================
VaR (histórico, paramétrico, Monte Carlo, Filtered Historical Simulation,
Cornish-Fisher), Expected Shortfall / CVaR, stress testing, tracking error.

Compatível com a API v1 + novos métodos.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as _stats

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
    except Exception:
        z = -1.645 if confidence == 0.95 else -2.326
    return float(-(mu + z * sigma))


def parametric_cvar(close: pd.Series, confidence: float = 0.95, window: int = 252) -> float:
    """Expected Shortfall sob normalidade."""
    rets = daily_returns(close).tail(window)
    if rets.empty:
        return float("nan")
    mu, sigma = rets.mean(), rets.std(ddof=1)
    z = _stats.norm.ppf(1 - confidence)
    es = -(mu + sigma * _stats.norm.pdf(z) / (1 - confidence))
    return float(es)


def cornish_fisher_var(close: pd.Series, confidence: float = 0.95, window: int = 252) -> float:
    """VaR com correção de Cornish-Fisher (skew + kurtosis)."""
    rets = daily_returns(close).tail(window).dropna()
    if len(rets) < 20:
        return parametric_