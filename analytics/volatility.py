"""
Analytics — Modelagem de Volatilidade GARCH(1,1)
====================================================
Implementação própria via máxima verossimilhança (scipy.optimize), sem
depender do pacote `arch` — reduz superfície de dependências externas
mantendo o rigor estatístico do modelo. Serve como base extensível para
EGARCH/DCC-GARCH em iterações futuras (ver README, seção Roadmap).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from analytics.metrics import daily_returns
import pandas as pd


def _garch11_neg_log_likelihood(params, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10

    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    sigma2 = np.maximum(sigma2, 1e-12)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
    return -ll


def fit_garch11(close: pd.Series, lookback: int = 500) -> dict:
    """Ajusta GARCH(1,1) por MLE nos últimos `lookback` retornos.

    Retorna parâmetros (omega, alpha, beta), a série de variância
    condicional in-sample e a previsão de volatilidade 1 dia à frente.
    """
    rets = (daily_returns(close).tail(lookback) * 100).values  # escala %  melhora condicionamento numérico
    rets = rets - rets.mean()

    x0 = [0.05, 0.08, 0.88]
    bounds = [(1e-6, None), (0, 1), (0, 1)]
    result = minimize(_garch11_neg_log_likelihood, x0, args=(rets,), method="L-BFGS-B", bounds=bounds)

    omega, alpha, beta = result.x
    n = len(rets)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(rets)
    for t in range(1, n):
        sigma2[t] = omega + alpha * rets[t - 1] ** 2 + beta * sigma2[t - 1]

    forecast_1d = omega + alpha * rets[-1] ** 2 + beta * sigma2[-1]
    persistence = alpha + beta

    idx = daily_returns(close).tail(lookback).index
    cond_vol_annualized = pd.Series(np.sqrt(sigma2) / 100 * np.sqrt(252), index=idx)

    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "persistence": float(persistence),
        "conditional_vol_annualized": cond_vol_annualized,
        "forecast_1d_vol_annualized": float(np.sqrt(forecast_1d) / 100 * np.sqrt(252)),
        "converged": bool(result.success),
        "log_likelihood": float(-result.fun),
    }


def ewma_volatility(close: pd.Series, lam: float = 0.94, window: int = 500) -> pd.Series:
    """Volatilidade EWMA (RiskMetrics) — mais leve que GARCH, útil como
    comparação rápida no dashboard de risco."""
    rets = daily_returns(close).tail(window)
    var = rets.copy() ** 2
    ewma_var = var.ewm(alpha=1 - lam, adjust=False).mean()
    return (np.sqrt(ewma_var) * np.sqrt(252))