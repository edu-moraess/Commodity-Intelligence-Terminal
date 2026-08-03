"""
Commodity Intelligence Terminal — Configuração Central (v4.4.0)
================================================================
Define o universo de ativos, mapeamento de tickers (Yahoo Finance / FRED),
parâmetros de cache e constantes globais da aplicação.

CHANGELOG v4.4.0:
- Tickers sem liquidez no Yahoo (ALI=F, TIO=F) forçados para source="synthetic"
  evitando retry demorado e timeout na inicialização.
- TTLs diferenciados: PRICE=300s, MACRO=3600s, COMPUTE=900s (alias CACHE_TTL_SECONDS).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    category: str
    unit: str
    source: str = "yahoo"
    note: Optional[str] = None


# --------------------------------------------------------------------------
# UNIVERSO DE ATIVOS
# --------------------------------------------------------------------------

ENERGY_ASSETS: List[Asset] = [
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

METALS_ASSETS: List[Asset] = [
    Asset("GC=F", "Ouro", "Metais", "USD/oz t"),
    Asset("SI=F", "Prata", "Metais", "USD/oz t"),
    Asset("HG=F", "Cobre", "Metais", "USD/lb"),
    Asset("ALI=F", "Alumínio", "Metais", "USD/t", source="synthetic",
          note="Futuro de alumínio não disponível no Yahoo Finance — dados sintéticos."),
    Asset("LIT", "Lítio (Global X Lithium ETF — proxy)", "Metais", "USD",
          note="Proxy via ETF; integrar Fastmarkets/Benchmark Mineral Intelligence para preço spot direto."),
    Asset("TIO=F", "Minério de Ferro (SGX)", "Metais", "USD/t", source="synthetic",
          note="SGX Iron Ore não disponível no Yahoo Finance — dados sintéticos."),
    Asset("PICK", "Níquel & Zinco (iShares Metals & Mining — proxy)", "Metais", "USD",
          note="Sem contrato LME direto no Yahoo; proxy setorial até integração de fonte paga (LME Select)."),
]

AGRI_ASSETS: List[Asset] = [
    Asset("ZS=F", "Soja", "Agricultura", "USd/bu"),
    Asset("ZC=F", "Milho", "Agricultura", "USd/bu"),
    Asset("ZW=F", "Trigo", "Agricultura", "USd/bu"),
    Asset("KC=F", "Café Arábica", "Agricultura", "USd/lb"),
    Asset("SB=F", "Açúcar #11", "Agricultura", "USd/lb"),
    Asset("CT=F", "Algodão", "Agricultura", "USd/lb"),
    Asset("CC=F", "Cacau", "Agricultura", "USD/t"),
    Asset("OJ=F", "Suco de Laranja", "Agricultura", "USd/lb"),
]

ALL_ASSETS: List[Asset] = ENERGY_ASSETS + METALS_ASSETS + AGRI_ASSETS

BRAZIL_ASSETS: List[Asset] = [
    Asset("BZ=F", "Petróleo (Brasil exporta Brent-like)", "Brasil", "USD/bbl"),
    Asset("TIO=F", "Minério de Ferro", "Brasil", "USD/t", source="synthetic"),
    Asset("ZS=F", "Soja", "Brasil", "USd/bu"),
    Asset("ZC=F", "Milho", "Brasil", "USd/bu"),
    Asset("KC=F", "Café", "Brasil", "USd/lb"),
    Asset("SB=F", "Açúcar", "Brasil", "USd/lb"),
]

# --------------------------------------------------------------------------
# SÉRIES MACRO (FRED)
# --------------------------------------------------------------------------

MACRO_SERIES: Dict[str, dict] = {
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
DEFAULT_LOOKBACK_DAYS = 730

CACHE_TTL_PRICE = 300
CACHE_TTL_MACRO = 3600
CACHE_TTL_COMPUTE = 900
CACHE_TTL_SECONDS = CACHE_TTL_COMPUTE

RISK_FREE_RATE_ANNUAL = 0.045

FORECAST_HORIZONS = {
    "7 dias": 7,
    "30 dias": 30,
    "90 dias": 90,
    "180 dias": 180,
    "365 dias": 365,
}

THEME = {
    # Quant desk / algo terminal
    "background": "#080a0e",
    "surface": "#0f1319",
    "surface_alt": "#151b24",
    "border": "#1e2733",
    "text": "#e8eef6",
    "text_muted": "#8b97a8",

    "accent": "#4c9aff",
    "accent_amber": "#4c9aff",

    "positive": "#00c853",
    "negative": "#ff4d5a",
    "warning": "#f0a030",

    "ticker_bg": "#050608",

    "chart_extra_1": "#7c9cbf",
    "chart_extra_2": "#c4a35a",
}