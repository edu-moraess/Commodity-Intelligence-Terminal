"""
DataManager — Camada Unificada de Dados
==========================================
Ponto único de acesso a dados de preço e macro no terminal. Responsável por:
  1. Tentar a fonte real (Yahoo Finance / FRED)
  2. Cachear em memória (via st.cache_data, injetado no runtime Streamlit)
  3. Fazer fallback automático e transparente para dados sintéticos,
     marcando a série com `is_synthetic=True`
  4. Logar todas as falhas de forma estruturada (utils/logger.py)

Nenhuma página deve chamar `yfinance` ou `requests` diretamente — todas
passam por aqui, o que mantém a troca de provedor (ex: para Refinitiv/
Bloomberg no futuro) como uma mudança isolada neste arquivo.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from data.sources import yahoo_finance, fred, synthetic
from config.settings import Asset, DEFAULT_LOOKBACK_DAYS
from utils.logger import get_logger

logger = get_logger("data_manager")


@dataclass
class PriceData:
    df: pd.DataFrame          # OHLCV + OpenInterest
    is_synthetic: bool
    source: str
    asset: Asset


def load_price_history(asset: Asset, days: int = DEFAULT_LOOKBACK_DAYS) -> PriceData:
    """Retorna histórico de preços para um ativo, com fallback automático."""
    if asset.source == "synthetic":
        df = synthetic.generate_price_series(asset.ticker, days=days)
        return PriceData(df=df, is_synthetic=True, source="synthetic", asset=asset)

    try:
        df = yahoo_finance.fetch_ohlcv(asset.ticker, period_days=days)
        return PriceData(df=df, is_synthetic=False, source="yahoo_finance", asset=asset)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fallback sintético ativado para %s: %s", asset.ticker, exc)
        df = synthetic.generate_price_series(asset.ticker, days=days)
        return PriceData(df=df, is_synthetic=True, source="synthetic_fallback", asset=asset)


def load_price_history_bulk(assets: list[Asset], days: int = DEFAULT_LOOKBACK_DAYS) -> dict[str, PriceData]:
    """Carrega vários ativos de uma vez, mantendo o mapeamento ticker -> PriceData."""
    result: dict[str, PriceData] = {}
    for asset in assets:
        result[asset.ticker] = load_price_history(asset, days=days)
    return result


def load_macro_series(series_key: str, series_code: str, days: int = 1825) -> tuple[pd.Series, bool]:
    """Retorna (série, is_synthetic) para uma série macro do FRED."""
    try:
        s = fred.fetch_series(series_code)
        return s, False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fallback sintético ativado para série macro %s: %s", series_code, exc)
        return synthetic.generate_macro_series(series_code, days=days), True


def build_price_panel(price_data: dict[str, PriceData], field: str = "Close") -> pd.DataFrame:
    """Monta um painel wide (colunas = tickers) alinhado por data para
    análises de correlação, PCA, etc."""
    series = {}
    for ticker, pd_data in price_data.items():
        series[ticker] = pd_data.df[field]
    panel = pd.DataFrame(series)
    return panel.sort_index().ffill().dropna(how="all")