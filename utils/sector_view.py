"""
Sector View — Renderização Compartilhada por Setor
======================================================
Energia, Metais e Agricultura compartilham a mesma anatomia de página
(cards de preço, tabela de métricas, candlestick por ativo, correlação
intra-setor). Este módulo centraliza essa renderização para as três
páginas de setor, evitando triplicar ~200 linhas de layout Streamlit.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd

from config.settings import Asset, RISK_FREE_RATE_ANNUAL
from data.data_manager import load_price_history_bulk, build_price_panel
from analytics import metrics, correlation
from charts import plotly_charts as charts


def render_sector_page(assets: list[Asset], sector_name: str, sector_note: str = "") -> None:
    st.title(f"{sector_name}")
    if sector_note:
        st.caption(sector_note)

    with st.spinner(f"Carregando dados de {sector_name}..."):
        price_data = load_price_history_bulk(assets)

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
        with cols[i % len(cols)]:
            delta = row["chg_1d"]
            st.metric(
                label=f"{asset.name} ({asset.unit})",
                value=f"{row['last_price']:.2f}" if row["last_price"] else "—",
                delta=f"{delta:+.2%}" if delta is not None else None,
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
    st.dataframe(
        df_table.style.format({c: "{:.2%}" for c in pct_cols} | {
            "Último": "{:.2f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}",
        }),
        use_container_width=True,
    )
    
    # -------- Metodologia das Métricas --------
    with st.expander("📖 Metodologia — Como interpretar esta tabela"):
        st.markdown("""
        | Métrica | O que significa | Como interpretar |
        |---------|---------------|------------------|
        | **1D / 1S / 1M / YTD** | Retorno acumulado no período (1 dia, 1 semana, 1 mês, ano corrente) | Positivo = alta; Negativo = queda |
        | **Vol. Anual.** | Desvio-padrão dos retornos diários × √252 | Quanto maior, mais arriscado o ativo |
        | **Sharpe** | (Retorno excedente) / (Volatilidade excedente) anualizado | > 1 é bom; < 0 é ruim (não compensa o risco) |
        | **Sortino** | Similar ao Sharpe, mas penaliza só a volatilidade negativa (downside) | Mais robusto que Sharpe para distribuições assimétricas |
        | **Max DD** | Maior queda do pico até o vale, nos últimos 252 pregões | Quanto mais próximo de 0, menor o risco de perda severa |
        | **Calmar** | Retorno anualizado / \|Max DD\| | Relação retorno/risco de drawdown; > 2 é excelente |
        | **Momentum** | Média dos retornos 1M/3M/6M/12M | Positivo = tendência de alta; Negativo = tendência de baixa |
        | **Tendência** | Cruzamento de médias móveis (20d vs 100d) | "Alta" = MM20 > MM100; "Baixa" = MM20 < MM100 |
        
        **Taxa livre de risco:** 4,5% a.a. (ajustável via `.env`).  
        **Janela padrão:** Sharpe/Sortino/Calmar usam 252 pregões (~1 ano); Volatilidade usa 63 pregões (~3 meses).
        """)

    st.divider()

    # -------- Candlestick individual --------
    st.subheader("Análise Individual")
    asset_names = {a.name: a for a in assets}
    selected_name = st.selectbox("Selecionar ativo", list(asset_names.keys()), key=f"sel_{sector_name}")
    selected_asset = asset_names[selected_name]
    selected_df = price_data[selected_asset.ticker].df
    st.plotly_chart(
        charts.candlestick_chart(selected_df.tail(180), title=f"{selected_asset.name} — 180 pregões"),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        cum_ret = metrics.cumulative_return_series(selected_df["Close"])
        st.plotly_chart(
            charts.line_chart({selected_asset.name: cum_ret}, title="Retorno Acumulado", y_title="%"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            charts.bar_chart(["Volume médio (20d)"], [float(selected_df["Volume"].tail(20).mean())],
                              title="Volume Médio Recente", positive_negative=False),
            use_container_width=True,
        )

    st.divider()

    # -------- Correlação intra-setor --------
    if len(assets) > 2:
        st.subheader(f"Correlação Intra-Setor — {sector_name}")
        panel = build_price_panel(price_data)
        panel.columns = [a.name for a in assets if a.ticker in panel.columns]
        corr = correlation.correlation_matrix(panel, window=126)
        st.plotly_chart(charts.correlation_heatmap(corr, title="Correlação (126 pregões)"), use_container_width=True)
        
        with st.expander("📖 Metodologia — Correlação Intra-Setor"):
            st.markdown("""
            A matriz exibe o coeficiente de correlação de Pearson entre os **retornos diários** dos ativos do setor, calculado sobre uma janela de 126 pregões (~6 meses).
            
            - **ρ ≈ +1** → ativos se movem juntos (diversificação baixa)
            - **ρ ≈ 0** → movimentos independentes
            - **ρ ≈ –1** → se movem em direções opostas (raro em commodities do mesmo setor)
            
            Em commodities de energia, por exemplo, Brent e WTI costumam ter ρ > 0.9. Já gás natural pode ter correlação mais baixa devido a dinâmicas regionais de oferta/demanda.
            """)

    if any(a.note for a in assets):
        with st.expander("📌 Notas metodológicas de fonte de dados"):
            for a in assets:
                if a.note:
                    st.markdown(f"**{a.name}** — {a.note}")
