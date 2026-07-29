"""
Fonte de Dados — FRED (Federal Reserve Economic Data)
========================================================
Usa a API REST pública do FRED diretamente via `requests` (evita
dependência pesada de SDK). Requer `FRED_API_KEY` no ambiente — chave
gratuita em https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations
import os
import logging
import requests
import pandas as pd

logger = logging.getLogger("commodity_terminal.fred")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredFetchError(Exception):
    """Erro ao buscar série no FRED (chave ausente, série inválida, rede)."""


def fetch_series(series_code: str, start_date: str = "2015-01-01") -> pd.Series:
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        raise FredFetchError("FRED_API_KEY não configurada no ambiente (.env)")

    params = {
        "series_id": series_code,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    try:
        resp = requests.get(FRED_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        obs = payload.get("observations", [])
        if not obs:
            raise FredFetchError(f"Sem observações retornadas para {series_code}")
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        series = df.set_index("date")["value"].dropna()
        series.name = series_code
        return series
    except requests.RequestException as exc:
        raise FredFetchError(f"Erro de rede ao buscar {series_code}: {exc}") from exc