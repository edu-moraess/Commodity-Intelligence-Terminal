"""
Commodity Intelligence Terminal — Configuração Central (v4.4.0)
================================================================
Define o universo de ativos, mapeamento de tickers (Yahoo Finance / FRED),
parâmetros de cache e constantes globais da aplicação.

CHANGELOG v4.4.0:
- Tickers sem liquidez no Yahoo (ALI=F, TIO=F) forçados para source="synthetic"
  evitando retry demorado e timeout na inicialização.
- CACHE_TTL_SECONDS aumentado para 900s (15 min) para reduzir chamadas à API.
"""

from dataclasses import dataclass
from typing import Optional


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
    # FIX v4.4.0: ALI=F não existe no Yahoo Finance — força sintético
    Asset("ALI=F", "Alumínio", "Metais", "USD/t", source="synthetic",
          note="Futuro de alumínio não disponível no Yahoo Finance — dados sintéticos."),
    Asset("LIT", "Lítio (Global X Lithium ETF — proxy)", "Metais", "USD",
          note="Proxy via ETF; integrar Fastmarkets/Benchmark Mineral Intelligence para preço spot direto."),
    # FIX v4.4.0: TIO=F (SGX Iron Ore) não existe no Yahoo Finance — força sintético
    Asset("TIO=F", "Minério de Ferro (SGX)", "Metais", "USD/t", source="synthetic",
          note="SGX Iron Ore não disponível no Yahoo Finance — dados sintéticos."),
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
    # FIX v4.4.0: TIO=F forçado para synthetic
    Asset("TIO=F", "Minério de Ferro", "Brasil", "USD/t", source="synthetic"),
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
DEFAULT_LOOKBACK_DAYS = 730
CACHE_TTL_SECONDS = 900          # 15 minutos (aumentado de 300)
RISK_FREE_RATE_ANNUAL = 0.045

FORECAST_HORIZONS = {
    "7 dias": 7,
    "30 dias": 30,
    "90 dias": 90,
    "180 dias": 180,
    "365 dias": 365,
}

THEME = {
    # Fundo quase-preto neutro (sem tingimento roxo/azul, mais "terminal" que "SaaS")
    "background": "#0a0a0c",
    "surface": "#131317",
    "surface_alt": "#18181d",
    "border": "#26262c",
    "text": "#e8e8ea",
    "text_muted": "#88888f",

    # Accent principal: dourado/âmbar (identidade Bloomberg — troca o teal genérico
    # que lê como template SaaS padrão). Usado em: tabs ativas, nav ativo, hover
    # de cards/botões, bordas de destaque.
    "accent": "#c9a227",
    "accent_amber": "#c9a227",   # mantido por compatibilidade com código existente

    # Verde/vermelho mais sóbrios (paleta Investing.com), menos saturados que
    # o "candy green/red" genérico de dashboard.
    "positive": "#1fb37a",
    "negative": "#e6484c",

    # Warning agora distinto do accent dourado (antes colidiam — os dois eram
    # praticamente a mesma cor, gerando ambiguidade visual entre "destaque de UI"
    # e "estado de alerta").
    "warning": "#e0793c",

    "ticker_bg": "#000000",

    # Cores extras para gráficos multi-série (line_chart com >4 séries) e
    # setores sem cor própria (ex: "Brasil" no scatter risco-retorno).
    # Antes eram roxo/amarelo hardcoded (#9b8afb / #f7c948) — destoavam
    # completamente da identidade dourado/preto. Agora ficam na mesma
    # família tonal (steel-blue e bronze, ambos "frios/neutros" o
    # suficiente pra não competir com o dourado do accent principal).
    "chart_extra_1": "#5b8fa8",
    "chart_extra_2": "#9c8552",
}