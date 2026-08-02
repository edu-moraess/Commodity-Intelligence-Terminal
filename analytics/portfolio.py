"""
Analytics — Portfolio Optimization (Institucional)
===================================================
Markowitz (min variance / max Sharpe), Risk Parity, Max Diversification,
CVaR optimization (aproximação via cenários históricos).

Referências:
  - Markowitz (1952)
  - Maillard, Roncalli, Teiletche (2010) Risk Parity
  - Rockafellar & Uryasev (2000) CVaR optimization
"""

from __future__ import annotations
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.logger import get_logger

logger = get_logger("portfolio")

Method = Literal[
    "min_variance", "max_sharpe", "risk_parity",
    "max_diversification", "min_cvar", "equal_weight",
]


def _returns_matrix(price_panel: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    rets = price_panel.pct_change().dropna(how="all")
    rets = rets.dropna(how="all", axis=1)
    if window:
        rets = rets.tail(window)
    return rets.dropna()


def _cov_mean(rets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = rets.mean().values * 252
    cov = rets.cov().values * 252
    cov = cov + np.eye(cov.shape[0]) * 1e-8
    return mu, cov


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def min_variance_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    n = cov.shape[0]
    x0 = equal_weight(n)

    def obj(w):
        return float(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        logger.warning("min_variance_weights: SLSQP não convergiu: %s", res.message)
        return x0
    w = np.maximum(res.x, 0) if long_only else res.x
    return w / w.sum()


def max_sharpe_weights(mu, cov, risk_free: float = 0.045, long_only: bool = True) -> np.ndarray:
    n = len(mu)
    x0 = equal_weight(n)

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        if vol < 1e-12:
            return 1e6
        return -(ret - risk_free) / vol

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        logger.warning("max_sharpe_weights: SLSQP não convergiu: %s", res.message)
        return x0
    w = np.maximum(res.x, 0) if long_only else res.x
    return w / w.sum()


def risk_parity_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    n = cov.shape[0]
    x0 = equal_weight(n)

    def obj(w):
        w = np.asarray(w)
        port_var = float(w @ cov @ w)
        if port_var < 1e-16:
            return 1e6
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 800, "ftol": 1e-14})
    if not res.success:
        logger.warning("risk_parity_weights: SLSQP não convergiu (%s) — fallback 1/vol", res.message)
        vols = np.sqrt(np.diag(cov))
        w = 1.0 / np.maximum(vols, 1e-8)
        return w / w.sum()
    w = np.maximum(res.x, 0)
    return w / w.sum()


def max_diversification_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    n = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    x0 = equal_weight(n)

    def neg_dr(w):
        w = np.asarray(w)
        num = float(w @ vols)
        den = float(np.sqrt(w @ cov @ w))
        if den < 1e-12:
            return 1e6
        return -num / den

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    res = minimize(neg_dr, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500})
    if not res.success:
        logger.warning("max_diversification_weights: SLSQP não convergiu: %s", res.message)
        return x0
    w = np.maximum(res.x, 0)
    return w / w.sum()


def min_cvar_weights(rets: pd.DataFrame, alpha: float = 0.95, long_only: bool = True) -> np.ndarray:
    R = rets.values
    T, n = R.shape
    if T < 30 or n < 2:
        return equal_weight(n)
    x0 = equal_weight(n)

    def cvar_obj(w):
        port_rets = R @ w
        var = np.percentile(port_rets, (1 - alpha) * 100)
        tail = port_rets[port_rets <= var]
        if len(tail) == 0:
            return -var
        return float(-tail.mean())

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(cvar_obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-10})
    if not res.success:
        logger.warning("min_cvar_weights: SLSQP não convergiu: %s", res.message)
        return x0
    w = np.maximum(res.x, 0)
    return w / w.sum()


def portfolio_stats(weights, mu, cov, risk_free: float = 0.045) -> dict:
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free) / vol if vol > 1e-12 else 0.0
    return {"expected_return": ret, "volatility": vol, "sharpe": sharpe}


def risk_contributions(weights, cov) -> np.ndarray:
    port_var = float(weights @ cov @ weights)
    if port_var < 1e-16:
        return np.zeros_like(weights)
    mrc = cov @ weights
    rc = weights * mrc
    return rc / port_var


def efficient_frontier(mu, cov, n_points: int = 30, long_only: bool = True) -> pd.DataFrame:
    n = len(mu)
    targets = np.linspace(float(mu.min()), float(mu.max()), n_points)
    rows = []
    for target in targets:
        def obj(w):
            return float(w @ cov @ w)
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
        ]
        bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
        res = minimize(obj, equal_weight(n), method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300})
        if res.success:
            w = res.x
            rows.append({"return": float(w @ mu), "volatility": float(np.sqrt(w @ cov @ w))})
    return pd.DataFrame(rows)


def optimize_portfolio(
    price_panel: pd.DataFrame,
    method: Method = "max_sharpe",
    window: int = 252,
    risk_free: float = 0.045,
    cvar_alpha: float = 0.95,
    long_only: bool = True,
) -> dict:
    rets = _returns_matrix(price_panel, window=window)
    if rets.shape[1] < 2 or len(rets) < 30:
        raise ValueError("Histórico insuficiente para otimização de portfólio.")

    names = list(rets.columns)
    mu, cov = _cov_mean(rets)
    n = len(names)

    if method == "equal_weight":
        w = equal_weight(n)
    elif method == "min_variance":
        w = min_variance_weights(cov, long_only=long_only)
    elif method == "max_sharpe":
        w = max_sharpe_weights(mu, cov, risk_free=risk_free, long_only=long_only)
    elif method == "risk_parity":
        w = risk_parity_weights(cov, long_only=long_only)
    elif method == "max_diversification":
        w = max_diversification_weights(cov, long_only=long_only)
    elif method == "min_cvar":
        w = min_cvar_weights(rets, alpha=cvar_alpha, long_only=long_only)
    else:
        raise ValueError(f"Método desconhecido: {method}")

    stats = portfolio_stats(w, mu, cov, risk_free=risk_free)
    rc = risk_contributions(w, cov)
    port_rets = (rets * w).sum(axis=1)
    equity = (1 + port_rets).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")

    logger.info(
        "optimize_portfolio OK: method=%s n=%d sharpe=%.3f vol=%.4f",
        method, n, stats["sharpe"], stats["volatility"],
    )
    return {
        "weights": pd.Series(w, index=names, name="weight"),
        "risk_contributions": pd.Series(rc, index=names, name="risk_contribution"),
        "stats": stats,
        "method": method,
        "mu": pd.Series(mu, index=names),
        "vol_asset": pd.Series(np.sqrt(np.diag(cov)), index=names),
        "cov": pd.DataFrame(cov, index=names, columns=names),
        "equity_curve": equity,
        "max_drawdown": max_dd,
        "n_assets": n,
        "window": window,
    }


def compare_methods(price_panel, window: int = 252, risk_free: float = 0.045) -> pd.DataFrame:
    methods = [
        "equal_weight", "min_variance", "max_sharpe",
        "risk_parity", "max_diversification", "min_cvar",
    ]
    rows = []
    for m in methods:
        try:
            res = optimize_portfolio(price_panel, method=m, window=window, risk_free=risk_free)
            rows.append({
                "Método": m,
                "Retorno Anual.": res["stats"]["expected_return"],
                "Vol Anual.": res["stats"]["volatility"],
                "Sharpe": res["stats"]["sharpe"],
                "Max DD (in-sample)": res["max_drawdown"],
                "N ativos > 1%": int((res["weights"] > 0.01).sum()),
            })
        except Exception as exc:
            logger.warning("compare_methods: método '%s' falhou: %s", m, exc)
            rows.append({"Método": m, "Erro": str(exc)[:50]})
    return pd.DataFrame(rows)