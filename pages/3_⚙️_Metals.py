import streamlit as st
from config.settings import METALS_ASSETS, APP_NAME
from utils.sector_view import render_sector_page

# NOTA v4.5.0: st.set_page_config() removido daqui — agora é chamado
# uma única vez em app.py, antes de st.navigation(...).run().

# Metodologia específica para o setor de Metais
METALS_METHODOLOGY = """
**Objetivo:** Analisar o desempenho dos principais metais (preciosos, industriais e de transição energética), com métricas de retorno, risco e correlação intra-setor.

**Principais ativos monitorados:**
- **Ouro, Prata e Paládio:** Metais preciosos, tradicionalmente vistos como reserva de valor e hedge contra inflação.
- **Cobre e Alumínio:** Metais industriais, sensíveis ao ciclo econômico global (construção, infraestrutura, eletrônicos).
- **Lítio, Níquel e Zinco:** Metais de transição energética, com demanda impulsionada por baterias e veículos elétricos.
- **Minério de Ferro:** Matéria-prima siderúrgica, fortemente correlacionada com a China e a indústria da construção.

**Métricas específicas para metais:**
- **Correlação com o Dólar (DXY):** Metais preciosos (ouro) têm correlação negativa com o dólar – quando o dólar sobe, o ouro tende a cair.
- **Prêmio de risco:** O spread entre o preço spot e o futuro pode indicar expectativas de escassez ou excesso de oferta.
- **Estoques LME (London Metal Exchange):** Níveis de estoque impactam diretamente o preço dos metais industriais (disponível via integração futura).

**Interpretação dos indicadores:**
- **Sharpe > 1:** Retorno ajustado ao risco excelente.
- **Sortino > Sharpe:** Indica que o risco de baixa (downside) é menor que o risco total – comum em ativos com assimetria positiva.
- **Drawdown elevado (>20%):** Metais são voláteis; drawdowns profundos são esperados em ciclos de baixa.
- **Momentum positivo:** Indica tendência de alta no curto/médio prazo; negativo, tendência de baixa.
"""

render_sector_page(
    METALS_ASSETS,
    sector_name="⚙️ Metals Analytics",
    sector_note=(
        "Ouro, Prata, Cobre, Alumínio, Lítio, Minério de Ferro, Níquel e "
        "Zinco (proxies onde não há futuro líquido no Yahoo Finance). "
        "Produção por país, reservas e custos de produção requerem "
        "integração USGS/Fastmarkets — ver README."
    ),
    methodology_text=METALS_METHODOLOGY,  # NOVO: passa a metodologia para o renderizador
)