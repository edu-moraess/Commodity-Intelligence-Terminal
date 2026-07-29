"""
Fonte de Dados — Yahoo Finance (v4.4.0)
=========================================
Wrapper fino sobre `yfinance` com retry exponencial e normalização de
schema (OHLCV + OpenInterest quando disponível).

CHANGELOG v4.4.0:
- Timeout reduzido de 15s para 8s (evita travamento em tickers inexistentes).
- max_retries reduzido de 3 para 2 (evita retry excessivo em tickers inválidos).
- sleep máximo limitado a 4s (antes era 8s).
- Validação explícita de DataFrame vazio antes de retornar.
"""

from __future__ import annotations
import time
import logging
import pandas as pd

logger = logging.getLogger("commodity_terminal.yahoo")


class YahooFetchError(Exception):
    """Erro ao buscar dados no Yahoo Finance após todas as tentativas."""


def fetch_ohlcv(ticker: str, period_days: int = 730, max_retries: int = 2) -> pd.DataFrame:
    """Busca OHLCV diário para `ticker` nos últimos `period_days`.

    Retorna DataFrame indexado por Date com colunas
    [Open, High, Low, Close, Volume].
    """
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.Ticker(ticker).history(
                period=f"{max(period_days, 30)}d", interval="1d", auto_adjust=False, timeout=8
            )
            if df is None or df.empty:
                raise YahooFetchError(f"Retorno vazio para {ticker}")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Date"
            df["OpenInterest"] = pd.NA
            return df
        except Exception as exc:
            last_err = exc
            logger.warning("Tentativa %s/%s falhou para %s: %s", attempt, max_retries, ticker, exc)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 4))

    raise YahooFetchError(f"Falha ao buscar {ticker} após {max_retries} tentativas: {last_err}")
