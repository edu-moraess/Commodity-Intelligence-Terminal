"""
Forecasting — Scenario Cache (Produção)
=======================================
Cache de `scenario_summary` com fingerprint de parâmetros + TTL Streamlit.

Evita re-simular 2000 paths a cada interação de UI quando os inputs
(método, seed, horizonte, último preço, n_sims) não mudaram.
"""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import CACHE_TTL_COMPUTE, APP_VERSION
from forecasting.mc_engine import scenario_summary as _scenario_summary_raw


def _fingerprint(
    ticker_or_key: str,
    last_price: float,
    last_date: str,
    horizon_days: int,
    n_sims: int,
    method: str,
    seed: int,
    block_size: int | None,
) -> str:
    raw = (
        f"{ticker_or_key}|{last_price:.6f}|{last_date}|{horizon_days}|"
        f"{n_sims}|{method}|{seed}|{block_size}|{APP_VERSION}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@st.cache_data(ttl=CACHE_TTL_COMPUTE, show_spinner=False)
def _cached_scenario(
    fp: str,
    close_json: str,
    horizon_days: int,
    n_sims: int,
    method: str,
    seed: int,
    block_size: int | None,
) -> dict[str, Any]:
    """Cache interno — chave = fingerprint; close serializado como JSON."""
    close = pd.read_json(close_json, typ="series")
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    result = _scenario_summary_raw(
        close,
        horizon_days=horizon_days,
        n_sims=n_sims,
        method=method,
        seed=seed,
        block_size=block_size,
    )
    # Metadados de produção (não afetam cálculos)
    result["cache_fingerprint"] = fp
    result["app_version"] = APP_VERSION
    # fan_chart e final_prices_dist não serializam bem no cache interno do st
    # — st.cache_data pickleia o dict; DataFrame/ndarray ok
    return result


def scenario_summary_cached(
    close: pd.Series,
    horizon_days: int,
    n_sims: int = 2000,
    method: str = "block_bootstrap",
    seed: int = 42,
    block_size: int | None = None,
    cache_key: str = "default",
) -> dict[str, Any]:
    """Wrapper de produção com cache.

    Use na UI no lugar de scenario_summary quando quiser evitar recompute.
    """
    close = close.dropna()
    last_price = float(close.iloc[-1])
    last_date = str(close.index[-1].date()) if hasattr(close.index[-1], "date") else str(close.index[-1])
    fp = _fingerprint(
        cache_key, last_price, last_date, horizon_days, n_sims, method, seed, block_size
    )
    close_json = close.to_json(date_format="iso")
    return _cached_scenario(
        fp, close_json, horizon_days, n_sims, method, seed, block_size
    )
