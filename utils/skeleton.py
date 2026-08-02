"""
Skeleton / placeholder loading — UX de carregamento estruturado.
================================================================
Em vez de um spinner genérico central, mostra a *forma* do conteúdo
(KPI cards + bloco de tabela) enquanto os dados chegam.
"""

from __future__ import annotations
import streamlit as st


def skeleton_kpi_row(n: int = 4, labels: list[str] | None = None) -> None:
    """Placeholder de cards KPI (métricas) — mesma geometria dos st.metric finais."""
    cols = st.columns(n)
    for i, col in enumerate(cols):
        label = (labels[i] if labels and i < len(labels) else "—")
        with col:
            st.metric(label, "…", delta=None)
            st.caption("carregando")


def skeleton_table_block(title: str = "Carregando tabela…", rows_hint: int = 6) -> None:
    """Placeholder de área tabular / painel principal."""
    st.subheader(title)
    import pandas as pd
    placeholder = pd.DataFrame(
        {f"col_{j}": ["…"] * rows_hint for j in range(4)}
    )
    st.dataframe(placeholder, width="stretch", hide_index=True)
    st.caption("Aguardando dados…")


def skeleton_chart_block(title: str = "Carregando gráfico…") -> None:
    st.subheader(title)
    st.info("Preparando visualização…", icon="⏳")