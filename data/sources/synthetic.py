"""
Gerador de Dados Sintéticos — Fallback de Continuidade
=========================================================
Quando uma fonte externa (Yahoo Finance, FRED) falha — rate limit, símbolo
indisponível, sem conexão — o terminal NÃO deve quebrar. Este módulo gera
uma série de preços plausível (GBM com regime de volatilidade) ancorada em
um preço-base realista por ativo, para que toda a interface continue
funcional e demonstrável até a fonte real voltar.

Toda série sintética é marcada com `is_synthetic=True` no DataManager e a
interface exibe um badge "DADOS SIMULADOS" — nunca se apresenta como dado
real silenciosamente.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# Preços-base aproximados (ordem de grandeza real, jul/2026) usados apenas
# como âncora do processo estocástico sintético — não são cotações reais.
_BASE_PRICE = {
    "BZ=F": 78.0, "CL=F": 74.0, "NG=F": 3.1, "RB=F": 2.3, "HO=F": 2.5,
    "BTU": 22.0, "URA": 28.0,
    "GC=F": 2450.0, "SI=F": 29.0, "HG=F": 4.4, "ALI=F": 2450.0,
    "LIT": 45.0, "TIO=F": 108.0, "PICK": 38.0,
    "ZS=F": 1050.0, "ZC=F": 445.0, "ZW=F": 585.0, "KC=F": 220.0,
    "SB=F": 19.5, "CT=F": 78.0, "CC=F": 8200.0, "OJ=F": 340.0,
}

_BASE_VOL = {  # volatilidade diária anualizada aproximada por classe
    "BZ=F": 0.32, "CL=F": 0.34, "NG=F": 0.55, "RB=F": 0.36, "HO=F": 0.34,
    "BTU": 0.40, "URA": 0.38,
    "GC=F": 0.15, "SI=F": 0.26, "HG=F": 0.24, "ALI=F": 0.20,
    "LIT": 0.45, "TIO=F": 0.30, "PICK": 0.28,
    "ZS=F": 0.22, "ZC=F": 0.24, "ZW=F": 0.28, "KC=F": 0.34,
    "SB=F": 0.30, "CT=F": 0.26, "CC=F": 0.45, "OJ=F": 0.38,
}


def generate_price_series(ticker: str, days: int = 730, seed: int | None = None) -> pd.DataFrame:
    """Gera uma série OHLCV sintética via GBM com choques de regime.

    Determinístico por ticker (seed derivado do hash do ticker) a menos que
    um `seed` explícito seja passado — evita que o gráfico "pisque" outro
    padrão a cada rerun do Streamlit.

    NOTA (bugfix crítico): `pd.bdate_range(end=X, periods=N)` retorna
    apenas N-1 datas quando X cai num sábado/domingo (comportamento do
    pandas ao excluir o próprio dia não-útil do "hoje" antes de contar
    os N períodos). Isso fazia o app inteiro quebrar (ValueError ao
    montar o DataFrame) sempre que o fallback sintético fosse acionado
    num fim de semana — ou seja, de forma intermitente e dependente do
    dia da semana em que o Yahoo Finance falhasse. Corrigido gerando
    todos os arrays com base em `len(dates)` (o tamanho real do índice
    retornado), nunca no parâmetro `days` diretamente.
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % (2**32))
    s0 = _BASE_PRICE.get(ticker, 100.0)
    ann_vol = _BASE_VOL.get(ticker, 0.25)
    daily_vol = ann_vol / np.sqrt(252)
    mu = 0.0  # drift neutro — não fazemos "previsão" implícita no fallback

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    n = len(dates)  # nunca assumir que n == days (ver nota acima)
    shocks = rng.normal(mu, daily_vol, size=n)

    # injeta 2-4 regimes de volatilidade elevada (choques macro/geopolíticos)
    n_regimes = rng.integers(2, 5)
    for _ in range(n_regimes):
        start = rng.integers(0, max(n - 20, 1))
        length = rng.integers(5, 20)
        end = min(start + length, n)
        shocks[start:end] *= rng.uniform(1.8, 3.0)

    log_returns = shocks
    close = s0 * np.exp(np.cumsum(log_returns))
    close = np.maximum(close, s0 * 0.15)  # piso de sanidade

    daily_range = close * (daily_vol * rng.uniform(0.5, 1.5, size=n))
    high = close + daily_range * rng.uniform(0.2, 0.6, size=n)
    low = close - daily_range * rng.uniform(0.2, 0.6, size=n)
    open_ = low + (high - low) * rng.uniform(0.3, 0.7, size=n)
    volume = rng.integers(5_000, 250_000, size=n)
    open_interest = rng.integers(20_000, 900_000, size=n)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": volume, "OpenInterest": open_interest,
    }, index=dates)
    df.index.name = "Date"
    return df


def generate_macro_series(code: str, days: int = 1825) -> pd.Series:
    """Série macro sintética suave (random walk de baixa volatilidade).

    Mesmo bugfix de `generate_price_series`: usa `len(dates)` em vez de
    `days` para gerar os arrays.
    """
    rng = np.random.default_rng(abs(hash(code)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    n = len(dates)
    base = {"DTWEXBGS": 105.0, "DGS10": 4.2, "FEDFUNDS": 4.5,
            "CPIAUCSL": 315.0, "PPIACO": 260.0, "INDPRO": 103.0,
            "CHNCPIALLMINMEI": 102.0}.get(code, 100.0)
    steps = rng.normal(0, base * 0.0015, size=n)
    series = base + np.cumsum(steps)
    return pd.Series(series, index=dates, name=code)