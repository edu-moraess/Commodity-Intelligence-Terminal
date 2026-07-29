"""
DataManager — Camada Unificada de Dados
==========================================
Ponto único de acesso a dados de preço e macro no terminal. Responsável por:
  1. Tentar a fonte real (Yahoo Finance / FRED) com retry mechanism.
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
import streamlit as st

from config.settings import Asset, DEFAULT_LOOKBACK_DAYS
from utils.logger import get_logger

logger = get_logger("data_manager")

# -----------------------------------------------------------------------------
# Tenta importar pandas_datareader apenas se disponível
# Caso contrário, usa fallback sintético para todas as séries macro
# -----------------------------------------------------------------------------
try:
    from pandas_datareader import data as pdr
    _HAS_PDR = True
except (ImportError, ModuleNotFoundError):
    _HAS_PDR = False
    logger.warning("pandas_datareader não disponível. Séries FRED usarão fallback sintético.")

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
            # Configuração para evitar bloqueios
            data = yf.download(
                tickers=ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                prepost=False,
                threads=True,
                progress=False,
            )
            
            # Valida se veio dados minimamente aceitáveis
            if data.empty:
                raise ValueError("DataFrame vazio")
            if data["Close"].isna().all():
                raise ValueError("Todos os preços de fechamento são NaN")
            
            # Se chegou aqui, sucesso!
            logger.info(f"Dados reais obtidos para {ticker} na tentativa {attempt+1}")
            return data

        except Exception as e:
            last_error = e
            logger.warning(f"Tentativa {attempt+1}/{max_retries} falhou para {ticker}: {e}")
            if attempt < max_retries - 1:
                # Espera exponencial: 2s, 4s, 8s
                sleep_time = 2 ** (attempt + 1)
                time.sleep(sleep_time)
            continue

    # Se chegou aqui, todas as tentativas falharam
    raise RuntimeError(f"Falha ao obter dados para {ticker} após {max_retries} tentativas. Último erro: {last_error}")


def _fetch_fred_with_retry(series_code: str, max_retries: int = 3) -> pd.Series:
    """
    Tenta buscar série do FRED usando pandas_datareader (se disponível).
    Se não estiver disponível, levanta exceção para trigger do fallback.
    """
    if not _HAS_PDR:
        raise RuntimeError("pandas_datareader não instalado. Use fallback sintético.")
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # Usa a API do FRED (necessita da chave configurada nas secrets)
            series = pdr.DataReader(series_code, "fred")
            if series.empty:
                raise ValueError(f"Série {series_code} vazia")
            # Pega a coluna (geralmente é a única)
            series = series.iloc[:, 0]
            logger.info(f"Série FRED {series_code} obtida com sucesso")
            return series
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

@st.cache_data(ttl=300)  # 5 minutos de cache
def load_price_history_cached(ticker: str, days: int) -> tuple[pd.DataFrame, bool, str]:
    """
    Versão cacheada do carregamento de preços.
    Retorna (df, is_synthetic, source) para permitir cache.
    """
    try:
        df = _fetch_yahoo_with_retry(ticker, days)
        return df, False, "yahoo_finance"
    except Exception as e:
        logger.warning(f"Fallback sintético ativado para {ticker}: {e}")
        # Gera dados sintéticos realistas
        df = _generate_synthetic_price_series(ticker, days)
        return df, True, "synthetic_fallback"


def load_price_history(asset: Asset, days: int = DEFAULT_LOOKBACK_DAYS) -> PriceData:
    """Retorna histórico de preços para um ativo, com fallback automático."""
    if asset.source == "synthetic":
        # Força sintético se o ativo estiver configurado assim
        df = _generate_synthetic_price_series(asset.ticker, days)
        return PriceData(df=df, is_synthetic=True, source="synthetic_forced", asset=asset)

    df, is_synth, source = load_price_history_cached(asset.ticker, days)
    return PriceData(df=df, is_synthetic=is_synth, source=source, asset=asset)


def load_price_history_bulk(assets: list[Asset], days: int = DEFAULT_LOOKBACK_DAYS) -> dict[str, PriceData]:
    """Carrega vários ativos de uma vez, mantendo o mapeamento ticker -> PriceData."""
    result: dict[str, PriceData] = {}
    for asset in assets:
        result[asset.ticker] = load_price_history(asset, days=days)
    return result


@st.cache_data(ttl=600)  # 10 minutos para dados macro (menos voláteis)
def load_macro_series_cached(series_code: str, days: int) -> tuple[pd.Series, bool]:
    """
    Versão cacheada do carregamento de séries macro.
    Retorna (série, is_synthetic).
    """
    try:
        s = _fetch_fred_with_retry(series_code)
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
# GERADORES SINTÉTICOS (FALLBACK)
# -----------------------------------------------------------------------------

def _generate_synthetic_price_series(ticker: str, days: int) -> pd.DataFrame:
    """Gera dados sintéticos realistas para fallback."""
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Preços iniciais realistas por ticker
    start_prices = {
        "BZ=F": 85.0, "CL=F": 80.0, "NG=F": 3.0, "RB=F": 3.2, "HO=F": 4.0,
        "BTU": 25.0, "URA": 30.0, "GC=F": 2500.0, "HG=F": 5.0, "SI=F": 30.0,
        "ZS=F": 1200.0, "ZC=F": 480.0, "ZW=F": 600.0, "KC=F": 300.0, "SB=F": 14.0,
    }
    start = start_prices.get(ticker, 100.0)
    
    # Passeio aleatório com drift e volatilidade
    drift = 0.08 / 252  # 8% anual
    vol = 0.25 / np.sqrt(252)  # 25% anual
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
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Valor inicial por tipo de série (aproximado)
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
    
    # Tendência suave com ruído
    trend = np.linspace(0, 0.05 * days/252, days)  # 5% de crescimento anual
    noise = np.random.normal(0, 0.02, days)  # ruído pequeno
    values = base * (1 + trend + noise)
    return pd.Series(values, index=dates)