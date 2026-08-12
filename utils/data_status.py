"""Componentes compartilhados para transparência de procedência e atualização.

Os dados de mercado exibidos pelo terminal são séries diárias de fechamento ou
séries simuladas quando o provedor não está disponível. Este módulo torna essa
distinção explícita na interface e evita alegações de cotação em tempo real.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import streamlit as st


def last_observation_timestamp(frame: pd.DataFrame | pd.Series | None) -> pd.Timestamp | None:
    """Retorna o timestamp mais recente de uma série, ou ``None`` se ela for inválida."""
    if frame is None or frame.empty:
        return None
    try:
        timestamps = pd.to_datetime(frame.index, errors="coerce")
        valid = timestamps[~pd.isna(timestamps)]
        if len(valid) == 0:
            return None
        return pd.Timestamp(valid.max())
    except (TypeError, ValueError):
        return None


def format_observation_date(frame: pd.DataFrame | pd.Series | None) -> str:
    """Formata a última observação de uma série para apresentação ao usuário."""
    timestamp = last_observation_timestamp(frame)
    return timestamp.strftime("%d/%m/%Y") if timestamp is not None else "N/D"


def price_source_label(price_data: Any) -> str:
    """Converte a origem interna de preços em uma descrição adequada à interface."""
    if getattr(price_data, "is_synthetic", False):
        return "Série simulada — não é cotação de mercado"

    source = getattr(price_data, "source", "")
    if source == "yahoo_finance":
        return "Yahoo Finance — fechamento diário"
    return "Origem não identificada"


def build_price_status_table(price_data: Mapping[str, Any]) -> pd.DataFrame:
    """Monta uma tabela de origem e data para um conjunto de séries de preço."""
    rows: list[dict[str, str]] = []
    for item in price_data.values():
        asset = getattr(item, "asset", None)
        frame = getattr(item, "df", None)
        is_synthetic = bool(getattr(item, "is_synthetic", False))
        rows.append(
            {
                "Ativo": getattr(asset, "name", "Ativo não identificado"),
                "Referência": format_observation_date(frame),
                "Origem": price_source_label(item),
                "Situação": "Simulado" if is_synthetic else "Última observação publicada",
            }
        )
    return pd.DataFrame(rows)


def render_price_data_disclosure(price_data: Mapping[str, Any], *, title: str = "Fonte e data das séries") -> None:
    """Renderiza uma divulgação consistente sobre os preços usados na página."""
    if not price_data:
        st.info("Não há séries de preço disponíveis para informar a procedência.")
        return

    total = len(price_data)
    synthetic_count = sum(bool(getattr(item, "is_synthetic", False)) for item in price_data.values())

    if synthetic_count == total:
        st.warning(
            "⚠️ Esta tela está usando apenas **séries simuladas**. Os valores não são "
            "cotações de mercado e não devem ser usados para acompanhamento de preço, "
            "sinal, risco ou decisão operacional.",
            icon="⚠️",
        )
    elif synthetic_count:
        st.warning(
            f"⚠️ {synthetic_count}/{total} séries desta tela são **simuladas**. Consulte "
            "a tabela de procedência antes de interpretar as métricas.",
            icon="⚠️",
        )

    st.caption(
        "Preços do Yahoo Finance são séries diárias de fechamento; não são cotações "
        "intradiárias nem uma garantia de atualização em tempo real. A referência abaixo "
        "é a última observação presente em cada série."
    )
    with st.expander(title, expanded=False):
        st.dataframe(build_price_status_table(price_data), width="stretch", hide_index=True)


def build_macro_status_table(
    macro_data: Mapping[str, pd.Series],
    synthetic_flags: Mapping[str, bool],
    macro_metadata: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    """Monta uma tabela de frequência, fonte e referência das séries macro."""
    metadata_by_name = {meta.get("name", key): meta for key, meta in macro_metadata.items()}
    rows: list[dict[str, str]] = []
    for name, series in macro_data.items():
        meta = metadata_by_name.get(name, {})
        is_synthetic = bool(synthetic_flags.get(name, False))
        frequency = meta.get("frequency", "Frequência não especificada")
        rows.append(
            {
                "Série": name,
                "Frequência": frequency,
                "Referência": format_observation_date(series),
                "Origem": "Série simulada — não é dado macro publicado" if is_synthetic else "FRED",
                "Situação": "Simulado" if is_synthetic else "Última observação publicada",
            }
        )
    return pd.DataFrame(rows)


def render_macro_data_disclosure(
    macro_data: Mapping[str, pd.Series],
    synthetic_flags: Mapping[str, bool],
    macro_metadata: Mapping[str, Mapping[str, str]],
) -> None:
    """Explica frequência e referência das séries macro antes das análises."""
    synthetic_count = sum(bool(flag) for flag in synthetic_flags.values())
    if synthetic_count:
        st.warning(
            f"⚠️ {synthetic_count} série(s) macro desta tela são **simuladas** e não "
            "representam divulgações econômicas publicadas.",
            icon="⚠️",
        )

    st.caption(
        "As séries macro têm calendários próprios de publicação. DXY e Treasury Yield "
        "são diários; Fed Funds, CPI, PPI, produção industrial e China CPI são mensais. "
        "Uma referência em julho pode ser a observação mais recente publicada, e não uma "
        "cotação intradiária atrasada."
    )
    with st.expander("Fonte, frequência e data das séries macro", expanded=False):
        st.dataframe(
            build_macro_status_table(macro_data, synthetic_flags, macro_metadata),
            width="stretch",
            hide_index=True,
        )
