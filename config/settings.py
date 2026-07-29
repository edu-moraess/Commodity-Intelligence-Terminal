"""
Commodity Intelligence Terminal — Configuração Central
=========================================================
Define o universo de ativos, mapeamento de tickers (Yahoo Finance / FRED),
parâmetros de cache e constantes globais da aplicação.

NOTA DE ENGENHARIA:
- Nem toda commodity possui um futuro líquido e diretamente disponível no
  Yahoo Finance (ex: Carvão, Níquel, Zinco, Minério de Ferro). Nesses casos
  usamos o melhor proxy público disponível (ETF ou contrato correlato) e
  isso fica documentado no campo `note` de cada ativo. Trocar por uma fonte
  paga (Refinitiv, Bloomberg, CME DataMine, S&P Platts) é uma troca de uma
  linha em `data/sources/`, sem refatoração estrutural.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Asset:
    ticker: str          # símbolo na fonte de dados (Yahoo Finance por padrão)
    name: str             # nome de exibição
    category: str         # Energia | Metais | Agricultura
    unit: str             # unidade de cotação
    source: str = "yahoo"  # yahoo | fred | synthetic
    note: Optional[str] = None


# --------------------------------------------------------------------------
# UNIVERSO DE ATIVOS
# --------------------------------------------------------------------------

ENERGY_ASSETS: list[Asset] = [
    Asset("BZ=F", "Brent Crude", "Energia", "USD/bbl"),
    Asset("CL=F", "WTI Crude", "Energia", "USD/bbl"),
    Asset("NG=F", "Natural Gas (Henry Hub)", "Energia", "USD/MMBtu"),
    Asset("RB=F", "Gasolina (RBOB)", "Energia", "USD/gal"),
    Asset("HO=F", "Diesel / Heating Oil", "Energia", "USD/gal"),
    Asset("BTU", "Carvão (Peabody Energy — proxy)", "Energia", "USD",
          note="Sem futuro líquido no Yahoo; proxy de equity do setor até integração de fonte paga (S&P Platts)."),
    Asset("URA", "Urânio (Global X Uranium ETF — proxy)", "Energia", "USD",
          note="Proxy via ETF de mineradoras de urânio; substituir por UxC/Numerco quando disponível."),
]

METALS_ASSETS: list[Asset] = [
    Asset("GC=F", "Ouro", "Metais", "USD/oz t"),
    Asset("SI=F", "Prata", "Metais", "USD/oz t"),
    Asset("HG=F", "Cobre", "Metais", "USD/lb"),
    Asset("ALI=F", "Alumínio", "Metais", "USD/t"),
    Asset("LIT", "Lítio (Global X Lithium ETF — proxy)", "Metais", "USD",
          note="Proxy via ETF; integrar Fastmarkets/Benchmark Mineral Intelligence para preço spot direto."),
    Asset("TIO=F", "Minério de Ferro (SGX)", "Metais", "USD/t"),
    Asset("PICK", "Níquel & Zinco (iShares Metals & Mining — proxy)", "Metais", "USD",
          note="Sem contrato LME direto no Yahoo; proxy setorial até integração de fonte paga (LME Select)."),
]

AGRI_ASSETS: list[Asset] = [
    Asset("ZS=F", "Soja", "Agricultura", "USd/bu"),
    Asset("ZC=F", "Milho", "Agricultura", "USd/bu"),
    Asset("ZW=F", "Trigo", "Agricultura", "USd/bu"),
    Asset("KC=F", "Café Arábica", "Agricultura", "USd/lb"),
    Asset("SB=F", "Açúcar #11", "Agricultura", "USd/lb"),
    Asset("CT=F", "Algodão", "Agricultura", "USd/lb"),
    Asset("CC=F", "Cacau", "Agricultura", "USD/t"),
    Asset("OJ=F", "Suco de Laranja", "Agricultura", "USd/lb"),
]

ALL_ASSETS: list[Asset] = ENERGY_ASSETS + METALS_ASSETS + AGRI_ASSETS

BRAZIL_ASSETS: list[Asset] = [
    Asset("BZ=F", "Petróleo (Brasil exporta Brent-like)", "Brasil", "USD/bbl"),
    Asset("TIO=F", "Minério de Ferro", "Brasil", "USD/t"),
    Asset("ZS=F", "Soja", "Brasil", "USd/bu"),
    Asset("ZC=F", "Milho", "Brasil", "USd/bu"),
    Asset("KC=F", "Café", "Brasil", "USd/lb"),
    Asset("SB=F", "Açúcar", "Brasil", "USd/lb"),
]

# --------------------------------------------------------------------------
# SÉRIES MACRO (FRED)
# --------------------------------------------------------------------------

MACRO_SERIES: dict[str, dict] = {
    "DXY":        {"code": "DTWEXBGS", "name": "US Dollar Index (Trade-Weighted)"},
    "UST10Y":     {"code": "DGS10", "name": "US 10Y Treasury Yield"},
    "FEDFUNDS":   {"code": "FEDFUNDS", "name": "Fed Funds Rate"},
    "CPI":        {"code": "CPIAUCSL", "name": "US CPI (All Urban Consumers)"},
    "PPI":        {"code": "PPIACO", "name": "US PPI (All Commodities)"},
    "INDPRO":     {"code": "INDPRO", "name": "US Industrial Production"},
    "CHINA_CPI":  {"code": "CHNCPIALLMINMEI", "name": "China CPI"},
}

# --------------------------------------------------------------------------
# PARÂMETROS GERAIS
# --------------------------------------------------------------------------

APP_NAME = "Commodity Intelligence Terminal"
APP_ICON = "🛢️"
DEFAULT_LOOKBACK_DAYS = 730           # 2 anos de histórico padrão
CACHE_TTL_SECONDS = 15 * 60           # 15 minutos
RISK_FREE_RATE_ANNUAL = 0.045         # usado em Sharpe/Sortino (ajustável via .env)

FORECAST_HORIZONS = {
    "7 dias": 7,
    "30 dias": 30,
    "90 dias": 90,
    "180 dias": 180,
    "365 dias": 365,
}

THEME = {
    "background": "#0b0e14",
    "surface": "#141822",
    "border": "#232838",
    "text": "#e6e8ee",
    "text_muted": "#8b93a7",
    "accent": "#3fb1ce",
    "positive": "#3ecf8e",
    "negative": "#e5484d",
    "warning": "#f5a623",
}