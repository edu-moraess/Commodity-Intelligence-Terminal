import streamlit as st
from config.settings import METALS_ASSETS, APP_NAME
from utils.sector_view import render_sector_page

st.set_page_config(page_title=f"Metais — {APP_NAME}", page_icon="⚙️", layout="wide")

render_sector_page(
    METALS_ASSETS,
    sector_name="⚙️ Metals Analytics",
    sector_note=(
        "Ouro, Prata, Cobre, Alumínio, Lítio, Minério de Ferro, Níquel e "
        "Zinco (proxies onde não há futuro líquido no Yahoo Finance). "
        "Produção por país, reservas e custos de produção requerem "
        "integração USGS/Fastmarkets — ver README."
    ),
)