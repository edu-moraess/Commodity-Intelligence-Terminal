import streamlit as st
import pandas as pd

from config.settings import BRAZIL_ASSETS, APP_NAME
from utils.sector_view import render_sector_page

# NOTA v4.5.0: st.set_page_config() removido daqui — agora é chamado
# uma única vez em app.py, antes de st.navigation(...).run().

render_sector_page(
    BRAZIL_ASSETS,
    sector_name="🇧🇷 Commodities Brasileiras",
    sector_note=(
        "Preços internacionais de referência para as principais commodities de exportação "
        "brasileira. Produção, exportações por porto/ferrovia e principais compradores usam "
        "dados de referência estática abaixo — a versão com dados vivos requer integração com "
        "ComexStat/Secex (MDIC) e ANTAQ — ver README."
    ),
)

st.divider()
st.subheader("📦 Referência — Principais Mercados de Exportação (dados estáticos, sujeitos a defasagem)")

ref_data = pd.DataFrame([
    {"Commodity": "Petróleo", "Principais Compradores": "China, Índia, Países Baixos",
     "Principais Portos": "Açu (RJ), Ilhabela/DTSE (SP)"},
    {"Commodity": "Minério de Ferro", "Principais Compradores": "China, Japão, Coreia do Sul",
     "Principais Portos": "Ponta da Madeira (MA), Tubarão (ES)"},
    {"Commodity": "Soja", "Principais Compradores": "China, União Europeia, Tailândia",
     "Principais Portos": "Santos (SP), Paranaguá (PR), Rondonópolis (via ferrovia)"},
    {"Commodity": "Milho", "Principais Compradores": "China, Irã, Japão",
     "Principais Portos": "Santos (SP), Rio Grande (RS)"},
    {"Commodity": "Café", "Principais Compradores": "EUA, Alemanha, Itália",
     "Principais Portos": "Santos (SP)"},
    {"Commodity": "Açúcar", "Principais Compradores": "China, Argélia, Bangladesh",
     "Principais Portos": "Santos (SP)"},
]).set_index("Commodity")

st.dataframe(ref_data, use_container_width=True)
st.caption(
    "⚠️ Tabela de referência estática incluída apenas para contexto qualitativo — "
    "não deve ser usada como fonte de série temporal ou para decisões operacionais. "
    "Para dados oficiais e atualizados: comexstat.mdic.gov.br"
)