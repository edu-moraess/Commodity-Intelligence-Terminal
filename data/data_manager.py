"""
DataManager — Camada Unificada de Dados
==========================================
Ponto único de acesso a dados de preço e macro no terminal. Responsável por:
  1. Tentar a fonte real (Yahoo Finance / FRED API) com retry mechanism.
  2. Cachear em memória (via st.cache_data) para performance.
  3. Fazer fallback automático e transparente para dados sintéticos,
     marcando a série com `is_synthetic=True`.
  4. Logar todas as falhas de forma estruturada.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
import time
import yfinance as yf
import requests
import streamlit as st
import os

from config.settings import Asset, DEFAULT_LOOKBACK_DAYS
from utils.logger import get_logger

logger = get_logger("data_manager")

# -----------------------------------------------------------------------------
# Tenta importar pandas_datareader (opcional, para fallback futuro)
# -----------------------------------------------------------------------------
try:
    from pandas_datareader import data as pdr
    _HAS_PDR = True
except (ImportError, ModuleNotFoundError):
    _HAS_PDR = False
    logger.info("pandas_datareader não disponível. Usando API REST do FRED.")


@dataclass
class PriceData:
    df: pd.DataFrame          # OHLCV
    is_synthetic: bool
    source: str
    asset: Asset


# -----------------------------------------------------------------------------
# FUNÇÕES DE BUSCA COM RETRY E VALIDAÇÃO
# -----------------------------------------------------------------------------

def _fetch_yahoo_with_retry(ticker: str, period_days: int, max_retries: int = 3) -> pd.DataFrame:
    """
    Tenta baixar dados do Yahoo Finance com até 3 tentativas.
    Se todas falharem, levanta exceção.
    """
    period = f"{period_days}d"
    last_error = None

    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                prepost=False,
                threads=True,
                progress=False,
            )
            
            if data.empty:
                raise ValueError("DataFrame vazio")
            if data["Close"].isna().all():
                raise ValueError("Todos os preços de fechamento são NaN")
            
            logger.info(f"Dados reais obtidos para {ticker} na tentativa {attempt+1}")
            return data

        except Exception as e:
            last_error = e
            logger.warning(f"Tentativa {attempt+1}/{max_retries} falhou para {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            continue

    raise RuntimeError(f"Falha ao obter dados para {ticker} após {max_retries} tentativas. Último erro: {last_error}")


def _fetch_fred_via_api(series_code: str, max_retries: int = 3) -> pd.Series:
    """
    Busca série do FRED usando a API REST diretamente.
    Requer FRED_API_KEY configurada nas secrets do Streamlit ou variável de ambiente.
    """
    # Tenta obter a chave
    api_key = None
    try:
        api_key = st.secrets.get("FRED_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("FRED_API_KEY")
    
    if not api_key:
        raise RuntimeError("FRED_API_KEY não configurada. Use fallback sintético.")
    
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_code,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "limit": 100000,
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            observations = data.get("observations", [])
            if not observations:
                raise ValueError(f"Nenhuma observação para {series_code}")
            
            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            df = df.set_index("date")["value"]
            
            df = df.sort_index()
            df = df.asfreq('D').ffill()
            
            logger.info(f"Série FRED {series_code} obtida via API REST com sucesso")
            return df
            
        except Exception as e:
            last_error = e
            logger.warning(f"Tentativa {attempt+1}/{max_retries} falhou para FRED {series_code}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            continue
    
    raise RuntimeError(f"Falha ao obter série FRED {series_code} após {max_retries} tentativas. Último erro: {last_error}")


# -----------------------------------------------------------------------------
# FUNÇÕES PÚBLICAS COM CACHE
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300)  # 5 minutos
def load_price_history_cached(ticker: str, days: int) -> tuple[pd.DataFrame, bool, str]:
    """
    Versão cacheada do carregamento de preços.
    Retorna (df, is_synthetic, source).
    """
    try:
        df = _fetch_yahoo_with_retry(ticker, days)
        return df, False, "yahoo_finance"
    except Exception as e:
        logger.warning(f"Fallback sintético ativado para {ticker}: {e}")
        df = _generate_synthetic_price_series(ticker, days)
        return df, True, "synthetic_fallback"


def load_price_history(asset: Asset, days: int = DEFAULT_LOOKBACK_DAYS) -> PriceData:
    """Retorna histórico de preços para um ativo, com fallback automático."""
    if asset.source == "synthetic":
        df = _generate_synthetic_price_series(asset.ticker, days)
        return PriceData(df=df, is_synthetic=True, source="synthetic_forced", asset=asset)

    df, is_synth, source = load_price_history_cached(asset.ticker, days)
    return PriceData(df=df, is_synthetic=is_synth, source=source, asset=asset)


def load_price_history_bulk(assets: list[Asset], days: int = DEFAULT_LOOKBACK_DAYS) -> dict[str, PriceData]:
    """Carrega vários ativos de uma vez."""
    result: dict[str, PriceData] = {}
    for asset in assets:
        result[asset.ticker] = load_price_history(asset, days=days)
    return result


@st.cache_data(ttl=600)  # 10 minutos para dados macro
def load_macro_series_cached(series_code: str, days: int) -> tuple[pd.Series, bool]:
    """
    Versão cacheada do carregamento de séries macro.
    Retorna (série, is_synthetic).
    """
    try:
        s = _fetch_fred_via_api(series_code)
        return s, False
    except Exception as e:
        logger.warning(f"Fallback sintético ativado para série FRED {series_code}: {e}")
        s = _generate_synthetic_macro_series(series_code, days)
        return s, True


def load_macro_series(series_key: str, series_code: str, days: int = 1825) -> tuple[pd.Series, bool]:
    """
    Retorna (série, is_synthetic) para uma série macro do FRED.
    """
    return load_macro_series_cached(series_code, days)


def build_price_panel(price_data: dict[str, PriceData], field: str = "Close") -> pd.DataFrame:
    """Monta um painel wide (colunas = tickers) alinhado por data."""
    series = {}
    for ticker, pd_data in price_data.items():
        if not pd_data.df.empty:
            series[ticker] = pd_data.df[field]
    panel = pd.DataFrame(series)
    return panel.sort_index().ffill().dropna(how="all")


# -----------------------------------------------------------------------------
# GERADORES SINTÉTICOS (FALLBACK) – CORRIGIDOS
# -----------------------------------------------------------------------------

def _generate_synthetic_price_series(ticker: str, days: int) -> pd.DataFrame:
    """Gera dados sintéticos realistas para fallback."""
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    # CORREÇÃO: periods=days para ter exatamente 'days' elementos
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    start_prices = {
        "BZ=F": 85.0, "CL=F": 80.0, "NG=F": 3.0, "RB=F": 3.2, "HO=F": 4.0,
        "BTU": 25.0, "URA": 30.0, "GC=F": 2500.0, "HG=F": 5.0, "SI=F": 30.0,
        "ZS=F": 1200.0, "ZC=F": 480.0, "ZW=F": 600.0, "KC=F": 300.0, "SB=F": 14.0,
    }
    start = start_prices.get(ticker, 100.0)
    
    drift = 0.08 / 252
    vol = 0.25 / np.sqrt(252)
    returns = np.random.normal(drift, vol, days)
    prices = start * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        "High": prices * (1 + np.random.uniform(0, 0.02, days)),
        "Low": prices * (1 - np.random.uniform(0, 0.02, days)),
        "Close": prices,
        "Volume": np.random.randint(1000, 100000, days)
    }, index=dates)
    return df


def _generate_synthetic_macro_series(series_code: str, days: int) -> pd.Series:
    """Gera série macro sintética para fallback."""
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    # CORREÇÃO: periods=days para ter exatamente 'days' elementos
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    base_values = {
        "DTWEXBGS": 120.0,   # DXY
        "DGS10": 4.5,        # Treasury 10Y
        "DFF": 5.0,          # Fed Funds
        "CPIAUCSL": 310.0,   # CPI
        "PPIACO": 240.0,     # PPI
        "INDPRO": 105.0,     # Industrial Production
        "CHNCPIALLMINMEI": 110.0,  # China CPI
    }
    base = base_values.get(series_code, 100.0)
    
    trend = np.linspace(0, 0.05 * days/252, days)
    noise = np.random.normal(0, 0.02, days)
    values = base * (1 + trend + noise)
    return pd.Series(values, index=dates)