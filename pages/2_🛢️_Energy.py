import streamlit as st
from config.settings import ENERGY_ASSETS, APP_NAME
from utils.sector_view import render_sector_page

# NOTA v4.5.0: st.set_page_config() removido daqui — agora é chamado
# uma única vez em app.py, antes de st.navigation(...).run().

render_sector_page(
    ENERGY_ASSETS,
    sector_name="🛢️ Energy Analytics",
    sector_note=(
        "Brent, WTI, Gás Natural, Gasolina, Diesel, Carvão e Urânio. "
        "Módulos de produção OPEP/EUA, EIA Weekly Inventory, Rig Count e "
        "curva de futuros (contango/backwardation) requerem integração "
        "com EIA API / CME — ver README para roadmap de expansão."
    ),
)