import streamlit as st
from config.settings import AGRI_ASSETS, APP_NAME
from utils.sector_view import render_sector_page

render_sector_page(
    AGRI_ASSETS,
    sector_name="🌾 Agriculture Analytics",
    sector_note=(
        "Soja, Milho, Trigo, Café, Açúcar, Algodão, Cacau e Suco de Laranja. "
        "Dados USDA (produção, safras, estoques), NDVI e El Niño/La Niña "
        "requerem integração com USDA NASS/QuickStats API — ver README."
    ),
)