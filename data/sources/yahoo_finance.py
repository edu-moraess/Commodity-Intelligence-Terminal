"""
Fonte de Dados — Yahoo Finance
=================================
Wrapper fino sobre `yfinance` com retry exponencial e normalização de
schema (OHLCV + OpenInterest quando disponível). Levanta exceções tipadas
para que o DataManager decida sobre fallback — este módulo não silencia
erros.
"""

from __future__ import annotations
import time
import logging
import pandas as pd

logger = logging.getLogger("commodity_terminal.yahoo")


class YahooFetchError(Exception):
    """Erro ao buscar dados no Yahoo Finance após todas as tentativas."""


def fetch_ohlcv(ticker: str, period_days: int = 730, max_retries: int = 3) -> pd.DataFrame:
    """Busca OHLCV diário para `ticker` nos últimos `period_days`.

    Retorna DataFrame indexado por Date com colunas
    [Open, High, Low, Close, Volume]. `OpenInterest` não é fornecido pelo
    Yahoo para futuros — preenchido como NaN e complementado por fonte
    dedicada (CME/ICE) quando configurada.
    """
    import yfinance as yf  # import local: mantém o app operável mesmo se
                             # a dependência opcional não estiver instalada
                             # em um ambiente de teste isolado.

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.Ticker(ticker).history(
                period=f"{max(period_days, 30)}d", interval="1d", auto_adjust=False
            )
            if df is None or df.empty:
                raise YahooFetchError(f"Retorno vazio para {ticker}")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Date"
            df["OpenInterest"] = pd.NA
            return df
        except Exception as exc:  # noqa: BLE001 — normalizamos qualquer falha de rede/parse
            last_err = exc
            logger.warning("Tentativa %s/%s falhou para %s: %s", attempt, max_retries, ticker, exc)
            time.sleep(min(2 ** attempt, 8))

    raise YahooFetchError(f"Falha ao buscar {ticker} após {max_retries} tentativas: {last_err}")