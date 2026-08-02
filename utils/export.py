"""
Export helpers — CSV / Excel para tabelas do terminal.
======================================================
Uso:
    from utils.export import download_dataframe
    download_dataframe(df, filename_stem="metricas_energy")
"""

from __future__ import annotations
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st


def _stem(name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return f"{safe}_{ts}"


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


def _to_excel_bytes(df: pd.DataFrame) -> bytes | None:
    """Retorna bytes XLSX se openpyxl estiver instalado; senão None."""
    try:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="dados")
        return buf.getvalue()
    except ImportError:
        return None
    except Exception:
        return None


def download_dataframe(
    df: pd.DataFrame,
    filename_stem: str = "export",
    label_prefix: str = "📥 Exportar",
    key: str | None = None,
) -> None:
    """
    Renderiza botões de download CSV (sempre) e Excel (se openpyxl disponível).
    Não altera o DataFrame original.
    """
    if df is None or df.empty:
        return

    stem = _stem(filename_stem)
    csv_bytes = _to_csv_bytes(df)
    xlsx_bytes = _to_excel_bytes(df)

    cols = st.columns(2 if xlsx_bytes is not None else 1)
    with cols[0]:
        st.download_button(
            label=f"{label_prefix} CSV",
            data=csv_bytes,
            file_name=f"{stem}.csv",
            mime="text/csv",
            key=key or f"dl_csv_{stem}",
            use_container_width=True,
        )
    if xlsx_bytes is not None:
        with cols[1]:
            st.download_button(
                label=f"{label_prefix} Excel",
                data=xlsx_bytes,
                file_name=f"{stem}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=(key or f"dl_xlsx_{stem}") + "_xlsx",
                use_container_width=True,
            )