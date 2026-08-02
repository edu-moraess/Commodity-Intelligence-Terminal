"""
Sector View — Renderização Compartilhada por Setor
======================================================
Energia, Metais e Agricultura compartilham a mesma anatomia de página
(cards de preço, tabela de métricas, candlestick por ativo, correlação
intra-setor). Este módulo centraliza essa renderização para as três
páginas de setor, evitando triplicar \~200 linhas de layout Streamlit.

CHANGELOG Parte 3:
- Skeleton loading (KPI + tabela) no lugar de spinner genérico isolado.
- Export CSV/Excel da tabela de métricas.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

from config.settings import Asset, RISK_FREE_RATE_ANNUAL
from data.data_manager import load_price_history_bulk, build_price_panel
from analytics import metrics, correlation
from charts import plotly_charts as charts
from utils.export import download_dataframe
from utils.skeleton import skeleton_kpi_row, skeleton_table_block


def render_sector_page(
    assets: list[Asset],
    sector_name: str,
    sector_note: str = "",
    methodology_text: str = "",
) -> None:
    st.title(f"{sector_name}")
    if sector_note:
        st.caption(sector_note)

    if methodology_text:
        with st.expander("📘 Metodologia Geral do Setor", expanded=False):
            st.markdown(methodology_text)
        st.divider()

    # -------- Skeleton enquanto carrega --------
    ph_kpi = st.empty()
    ph_table = st.empty()
    with ph_kpi.container():
        skeleton_kpi_row(
            n=min(4, len(assets)) or 1,
            labels=[a.name for a in assets[:4]],
        )
    with ph_table.container():
        skeleton_table_block(title="Métricas Quantitativas Completas")

    price_data = load_price_history_bulk(assets)

    # limpa skeletons
    ph_kpi.empty()
    ph_table.empty()

    any_synthetic = any(pdata.is_synthetic for pdata in price_data.values())
    if any_synthetic:
        st.warning(
            "⚠️ Um ou mais ativos estão exibindo **dados simulados** — a fonte "
            "ao vivo (Yahoo Finance) não respondeu neste ambiente. Em produção "
            "com acesso à internet, os dados reais são usados automaticamente.",
            icon="⚠️",
        )

    # -------- Cards de resumo por ativo --------
    cols = st.columns(min(4, len(assets)) or 1)
    for i, asset in enumerate(assets):
        pdat = price_data[asset.ticker]
        close = pdat.df["Close"]
        row = metrics.summary_row(close, risk_free_annual=RISK_FREE_RATE_ANNUAL)

        last_price = row["last_price"]
        delta = row["chg_1d"]

        if pd.isna(last_price) or last_price is None:
            display_value = "N/D"
            display_delta = None
        else:
            display_value = f"{last_price:.2f}"
            display_delta = f"{delta:+.2%}" if delta is not None and not pd.isna(delta) else None

        with cols[i % len(cols)]:
            st.metric(
                label=f"{asset.name} ({asset.unit})",
                value=display_value,
                delta=display_delta,
            )
            if pdat.is_synthetic:
                st.caption("🔸 simulado")

    st.divider()

    # -------- Tabela completa de métricas --------
    st.subheader("Métricas Quantitativas Completas")
    table_rows = []
    for asset in assets:
        pdat = price_data[asset.ticker]
        close = pdat.df["Close"]
        row = metrics.summary_row(close, risk_free_annual=RISK_FREE_RATE_ANNUAL)
        table_rows.append({
            "Ativo": asset.name,
            "Último": row["last_price"],
            "1D": row["chg_1d"], "1S": row["chg_1w"], "1M": row["chg_1m"], "YTD": row["chg_ytd"],
            "Vol. Anual.": row["vol_annual"], "Sharpe": row["sharpe"], "Sortino": row["sortino"],
            "Max DD": row["max_drawdown"], "Calmar": row["calmar"],
            "Momentum": row["momentum"], "Tendência": row["trend"],
        })
    df_table = pd.DataFrame(table_rows).set_index("Ativo")

    pct_cols = ["1D", "1S", "1M", "YTD", "Vol. Anual.", "Max DD", "Momentum"]
    float_cols = ["Último", "Sharpe", "Sortino", "Calmar"]

    style_dict = {col: "{:.2%}" for col in pct_cols}
    style_dict.update({col: "{:.2f}" for col in float_cols})

    st.dataframe(
        df_table.style.format(style_dict, na_rep="-"),
        width="stretch",
    )
    download_dataframe(
        df_table,
        filename_stem=f"metricas_{sector_name.lower().replace(' ', '_')}",
        key=f"export_sector_{sector_name}",
    )

    with st.expander("📐 Como as métricas são calculadas? (Fórmulas)"):
        st.markdown(r"""
        **Retornos (1D, 1S, 1M, YTD):** Variação percentual simples entre o preço atual e o preço de \(n\) dias atrás.

        **Volatilidade Anualizada:**
        \[ \sigma_{\text{anual}} = \sigma_{\text{diário}} \times \sqrt{252} \]

        **Sharpe Ratio (Sharpe, 1966):**
        \[ \text{Sharpe} = \frac{\bar{R} - R_f}{\sigma} \]
        onde \(\bar{R}\) é o retorno médio diário (anualizado), \(R_f\) é a taxa livre de risco (4.5% a.a.), e \(\sigma\) é o desvio padrão anualizado.

        **Sortino Ratio (Sortino & Price, 1994):**
        \[ \text{Sortino} = \frac{\bar{R} - R_f}{\sigma_{\text{down}}} \]
        onde \(\sigma_{\text{down}}\) é o desvio padrão **apenas** dos retornos negativos (downside deviation).

        **Máximo Drawdown (MDD):**
        \[ \text{MDD} = \max_{t} \left( \frac{\max_{s \leq t} P_s - P_t}{\max_{s \leq t} P_s} \right) \]
        mede a maior queda acumulada do preço de um pico a um vale (janela de 252 pregões).

        **Calmar Ratio:**
        \[ \text{Calmar} = \frac{\text{Retorno Acumulado (252d)}}{|\text{MDD}|} \]
        Quanto maior, melhor o desempenho ajustado à pior perda histórica.

        **Momentum Composto:** Média simples (não ponderada) dos retornos de 1, 3, 6 e 12 meses.

        **Tendência:** Compara a média móvel simples curta (20 pregões) com a longa (100 pregões):
        - **alta**: SMA(20) > SMA(100) × 1,01
        - **baixa**: SMA(20) < SMA(100) × 0,99
        - **lateral**: caso contrário (as duas médias estão próximas)
        """)

    st.divider()

    # -------- Candlestick individual --------
    st.subheader("Análise Individual")
    asset_names = {a.name: a for a in assets}
    selected_name = st.selectbox("Selecionar ativo", list(asset_names.keys()), key=f"sel_{sector_name}")
    selected_asset = asset_names[selected_name]
    selected_df = price_data[selected_asset.ticker].df

    if selected_df.empty or len(selected_df) < 2:
        st.info(f"Dados insuficientes para exibir o gráfico de {selected_asset.name}.")
    else:
        st.plotly_chart(
            charts.candlestick_chart(selected_df.tail(180), title=f"{selected_asset.name} — 180 pregões"),
            width="stretch",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            cum_ret = metrics.cumulative_return_series(selected_df["Close"])
            st.plotly_chart(
                charts.line_chart({selected_asset.name: cum_ret}, title="Retorno Acumulado", y_title="%"),
                width="stretch",
            )
        with col_b:
            try:
                avg_volume = float(selected_df["Volume"].tail(20).mean())
                if pd.isna(avg_volume):
                    avg_volume = 0.0
            except (KeyError, TypeError):
                avg_volume = 0.0
            st.plotly_chart(
                charts.bar_chart(
                    ["Volume médio (20d)"],
                    [avg_volume],
                    title="Volume Médio Recente",
                    positive_negative=False,
                ),
                width="stretch",
            )

    st.divider()

    # -------- Correlação intra-setor --------
    if len(assets) > 2:
        st.subheader(f"Correlação Intra-Setor — {sector_name}")
        panel = build_price_panel(price_data)
        panel.columns = [a.name for a in assets if a.ticker in panel.columns]

        if panel.empty or panel.shape[1] < 2:
            st.info("Dados insuficientes para calcular a matriz de correlação.")
        else:
            corr = correlation.correlation_matrix(panel, window=126)
            st.plotly_chart(
                charts.correlation_heatmap(corr, title="Correlação (126 pregões)"),
                width="stretch",
            )
            download_dataframe(
                corr,
                filename_stem=f"correlacao_{sector_name.lower().replace(' ', '_')}",
                key=f"export_corr_{sector_name}",
            )

    if any(a.note for a in assets):
        with st.expander("📌 Notas metodológicas de fonte de dados"):
            for a in assets:
                if a.note:
                    st.markdown(f"**{a.name}** — {a.note}") 