"""
DataManager — Camada Unificada de Dados (v4.4.0)
===================================================
Ponto único de acesso a dados de preço e macro no terminal.

CHANGELOG v4.4.0:
- Batch download do Yahoo Finance (1 chamada para N tickers) — reduz
  tempo de carregamento de ~30-60s para ~3-5s.
- Tickers inválidos conhecidos (ALI=F, TIO=F) pulam direto pro fallback
  sintético sem retry demorado.
- Tratamento de MultiIndex do yfinance (colunas flat automaticamente).
- Cache TTL alinhado com config.settings.CACHE_TTL_SECONDS.
- build_price_panel agora valida dados vazios e retorna DataFrame vazio
  com mensagem clara em vez de quebrar silenciosamente.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
import streamlit as st
import logging

from config.settings import Asset, DEFAULT_LOOKBACK_DAYS, CACHE_TTL_SECONDS
from data.sources.yahoo_finance import fetch_ohlcv, YahooFetchError
from data.sources.fred import fetch_series, FredFetchError
from data.sources.synthetic import generate_price_series, generate_macro_series

logger = logging.getLogger("commodity_terminal.data_manager")


@dataclass
class PriceData:
    df: pd.DataFrame
    is_synthetic: bool
    source: str
    asset: Asset


# --------------------------------------------------------------------------
# CACHE + BATCH
# --------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_yahoo_batch(tickers: tuple[str, ...], period_days: int) -> dict[str, pd.DataFrame]:
    """
    Batch download de múltiplos tickers do Yahoo Finance em UMA única chamada.
    Retorna dict {ticker: DataFrame} com colunas flat [Open, High, Low, Close, Volume].
    Tickers que falharem retornam DataFrame vazio.
    """
    import yfinance as yf

    if not tickers:
        return {}

    try:
        data = yf.download(
            tickers=list(tickers),
            period=f"{max(period_days, 30)}d",
            interval="1d",
            auto_adjust=True,
            prepost=False,
            threads=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as exc:
        logger.error(f"Batch download Yahoo falhou: {exc}")
        return {t: pd.DataFrame() for t in tickers}

    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            # Caso único: yf.download com 1 ticker retorna DataFrame flat
            if len(tickers) == 1:
                df = data
            else:
                # MultiIndex: acessa pelo ticker no nível 0
                if ticker not in data.columns.get_level_values(0):
                    result[ticker] = pd.DataFrame()
                    continue
                df = data[ticker].copy()

            if df is None or df.empty:
                result[ticker] = pd.DataFrame()
                continue

            # Normaliza colunas — remove MultiIndex se sobrar
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Garante que temos as colunas mínimas
            required = {"Open", "High", "Low", "Close", "Volume"}
            available = set(df.columns)
            if not required.issubset(available):
                logger.warning(f"{ticker}: colunas ausentes {required - available}")
                result[ticker] = pd.DataFrame()
                continue

            df = df[list(required)].copy()
            df.index.name = "Date"
            df["OpenInterest"] = pd.NA
            result[ticker] = df

        except Exception as exc:
            logger.warning(f"Erro ao processar {ticker} do batch: {exc}")
            result[ticker] = pd.DataFrame()

    return result


# --------------------------------------------------------------------------
# API PÚBLICA — Preços
# --------------------------------------------------------------------------

def load_price_history(asset: Asset, days: int = DEFAULT_LOOKBACK_DAYS) -> PriceData:
    """Retorna histórico de preços para um ativo, com fallback automático."""
    if asset.source == "synthetic":
        df = generate_price_series(asset.ticker, days)
        return PriceData(df=df, is_synthetic=True, source="synthetic_forced", asset=asset)

    try:
        df = fetch_ohlcv(asset.ticker, period_days=days)
        return PriceData(df=df, is_synthetic=False, source="yahoo_finance", asset=asset)
    except YahooFetchError as exc:
        logger.warning(f"Fallback sintético para {asset.ticker}: {exc}")
        df = generate_price_series(asset.ticker, days)
        return PriceData(df=df, is_synthetic=True, source="synthetic_fallback", asset=asset)


def load_price_history_bulk(
    assets: list[Asset], days: int = DEFAULT_LOOKBACK_DAYS
) -> dict[str, PriceData]:
    """
    Carrega vários ativos de uma vez.
    Ativos marcados como 'synthetic' ou inválidos no Yahoo vão direto pro fallback.
    O restante é carregado em batch (1 chamada à API).
    """
    result: dict[str, PriceData] = {}
    yahoo_assets: list[Asset] = []

    for asset in assets:
        if asset.source == "synthetic":
            df = generate_price_series(asset.ticker, days)
            result[asset.ticker] = PriceData(
                df=df, is_synthetic=True, source="synthetic_forced", asset=asset
            )
        else:
            yahoo_assets.append(asset)

    if yahoo_assets:
        tickers = tuple(a.ticker for a in yahoo_assets)
        batch_data = _fetch_yahoo_batch(tickers, days)

        for asset in yahoo_assets:
            df = batch_data.get(asset.ticker, pd.DataFrame())
            if df.empty or df["Close"].isna().all():
                logger.warning(f"{asset.ticker} vazio no batch — fallback sintético")
                df = generate_price_series(asset.ticker, days)
                result[asset.ticker] = PriceData(
                    df=df, is_synthetic=True, source="synthetic_fallback", asset=asset
                )
            else:
                result[asset.ticker] = PriceData(
                    df=df, is_synthetic=False, source="yahoo_finance", asset=asset
                )

    return result


# --------------------------------------------------------------------------
# API PÚBLICA — Macro
# --------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_macro_series_cached(series_code: str, days: int) -> tuple[pd.Series, bool]:
    try:
        series = fetch_series(series_code)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        series = series[series.index >= cutoff]
        return series, False
    except FredFetchError as exc:
        logger.warning(f"Fallback sintético para FRED {series_code}: {exc}")
        series = generate_macro_series(series_code, days)
        return series, True


def load_macro_series(series_key: str, series_code: str, days: int = 1825) -> tuple[pd.Series, bool]:
    return load_macro_series_cached(series_code, days)


# --------------------------------------------------------------------------
# UTILITÁRIOS
# --------------------------------------------------------------------------

def build_price_panel(price_data: dict[str, PriceData], field: str = "Close") -> pd.DataFrame:
    """
    Monta painel wide (colunas = tickers) alinhado por data.
    Remove colunas 100% NaN e retorna DataFrame vazio com aviso se nada sobrar.
    """
    series: dict[str, pd.Series] = {}
    for ticker, pd_data in price_data.items():
        if pd_data.df.empty or field not in pd_data.df.columns:
            continue
        s = pd_data.df[field].copy()
        if not s.isna().all():
            series[ticker] = s

    if not series:
        logger.warning("build_price_panel: nenhuma série válida encontrada")
        return pd.DataFrame()

    panel = pd.DataFrame(series)
    panel = panel.sort_index().ffill().dropna(how="all")
    return panel
