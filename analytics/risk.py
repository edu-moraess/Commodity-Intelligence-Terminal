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
        return parametric_var(close, confidence, window)
    mu, sigma = rets.mean(), rets.std(ddof=1)
    s = float(_stats.skew(rets))
    k = float(_stats.kurtosis(rets))
    z = _stats.norm.ppf(1 - confidence)
    z_cf = (z + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36)
    return float(-(mu + z_cf * sigma))


def monte_carlo_var(
    close: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
    n_sims: int = 10000,
    horizon_days: int = 1,
    seed: int = 42,
) -> float:
    """VaR via simulação Monte Carlo (GBM 1-day)."""
    rets = daily_returns(close).tail(window).dropna()
    if rets.empty:
        return float("nan")
    mu, sigma = float(rets.mean()), float(rets.std(ddof=1))
    rng = np.random.default_rng(seed)
    sim_rets = rng.normal(mu * horizon_days, sigma * np.sqrt(horizon_days), size=n_sims)
    return float(-np.percentile(sim_rets, (1 - confidence) * 100))


def filtered_historical_simulation_var(
    close: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
    lam: float = 0.94,
) -> float:
    """Filtered Historical Simulation (Barone-Adesi)."""
    rets = daily_returns(close).tail(window + 5).dropna()
    if len(rets) < 30:
        return historical_var(close, confidence, window)
    var_ewma = (rets ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    vol_ewma = np.sqrt(var_ewma)
    std_resids = (rets / vol_ewma).dropna()
    current_vol = float(vol_ewma.iloc[-1])
    filtered_rets = std_resids * current_vol
    return float(-np.percentile(filtered_rets.tail(window), (1 - confidence) * 100))


def stress_test(close: pd.Series, shocks_pct: list[float] | None = None) -> pd.DataFrame:
    """Aplica choques percentuais instantâneos ao último preço."""
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
    """Resumo completo de risco (histórico + paramétrico + CF + FHS + MC)."""
    return {
        "var_historico": historical_var(close, confidence, window),
        "var_parametrico": parametric_var(close, confidence, window),
        "var_cornish_fisher": cornish_fisher_var(close, confidence, window),
        "var_fhs": filtered_historical_simulation_var(close, confidence, window),
        "var_monte_carlo": monte_carlo_var(close, confidence, window),
        "cvar": historical_cvar(close, confidence, window),
        "cvar_parametrico": parametric_cvar(close, confidence, window),
        "confianca": confidence,
        "janela_dias": window,
    }