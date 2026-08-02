"""
Cache wrappers para funções quant caras.
Streamlit re-executa o script a cada widget; sem cache, GARCH/HMM/SLSQP
recalculam do zero.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from config.settings import CACHE_TTL_COMPUTE


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_regime_summary(
    close: pd.Series,
    n_states: int = 2,
    n_iter: int = 100,
    auto_select: bool = False,
) -> dict:
    from analytics.regimes import regime_summary
    return regime_summary(
        close, n_states=n_states, n_iter=n_iter, auto_select=auto_select
    )


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_fit_garch11(close: pd.Series, lookback: int = 500) -> dict:
    from analytics.volatility import fit_garch11
    return fit_garch11(close, lookback=lookback)


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_fit_volatility_model(
    close: pd.Series,
    model: str = "GARCH",
    lookback: int = 500,
) -> dict:
    from analytics.volatility import fit_volatility_model
    return fit_volatility_model(close, model=model, lookback=lookback)


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_select_best_volatility_model(
    close: pd.Series,
    lookback: int = 500,
    criterion: str = "aic",
) -> dict:
    from analytics.volatility import select_best_volatility_model
    return select_best_volatility_model(
        close, lookback=lookback, criterion=criterion
    )


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_compare_volatility_models(
    close: pd.Series,
    lookback: int = 500,
) -> pd.DataFrame:
    from analytics.volatility import compare_volatility_models
    return compare_volatility_models(close, lookback=lookback)


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_optimize_portfolio(
    price_panel: pd.DataFrame,
    method: str = "max_sharpe",
    window: int = 252,
    risk_free: float = 0.045,
    cvar_alpha: float = 0.95,
    long_only: bool = True,
) -> dict:
    from analytics.portfolio import optimize_portfolio
    return optimize_portfolio(
        price_panel,
        method=method,
        window=window,
        risk_free=risk_free,
        cvar_alpha=cvar_alpha,
        long_only=long_only,
    )


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_walk_forward_backtest(
    panel: pd.DataFrame,
    method: str = "max_sharpe",
    window: int = 252,
    risk_free: float = 0.045,
    long_only: bool = True,
    rebalance_freq: int = 21,
) -> dict:
    from analytics.portfolio_advanced import walk_forward_backtest
    return walk_forward_backtest(
        panel,
        method=method,
        window=window,
        risk_free=risk_free,
        long_only=long_only,
        rebalance_freq=rebalance_freq,
    )


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_compare_methods_advanced(
    panel: pd.DataFrame,
    window: int = 252,
    risk_free: float = 0.045,
    long_only: bool = True,
) -> pd.DataFrame:
    from analytics.portfolio_advanced import compare_methods_advanced
    return compare_methods_advanced(
        panel, window=window, risk_free=risk_free, long_only=long_only
    )


# --------------------------------------------------------------------------
# BACKTESTING DE VaR (analytics/backtesting.py) — faltavam no arquivo
# original; Risk Analytics é uma das páginas mais pesadas (rolling_var_
# forecast roda uma regressão/percentil por dia da janela) e ainda estava
# sem cache nenhum.
# --------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_joint_backtest(
    close: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
    method: str = "historical",
) -> dict:
    from analytics.backtesting import joint_backtest
    return joint_backtest(close, confidence=confidence, window=window, method=method)


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def cached_full_backtest_report(
    close: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
    method: str = "historical",
) -> dict:
    from analytics.backtesting import full_backtest_report
    return full_backtest_report(close, confidence=confidence, window=window, method=method)