import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import Asset
from data.data_manager import PriceData
from utils.data_status import (
    build_price_status_table,
    format_observation_date,
    last_observation_timestamp,
)


def test_last_observation_timestamp_returns_latest_index_value():
    frame = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-31", "2026-08-12"]),
    )

    assert last_observation_timestamp(frame) == pd.Timestamp("2026-08-12")
    assert format_observation_date(frame) == "12/08/2026"


def test_price_status_table_marks_synthetic_series_without_claiming_quote():
    asset = Asset("TIO=F", "Minério de Ferro", "Metais", "USD/t", source="synthetic")
    frame = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-08-12"]))
    item = PriceData(
        df=frame,
        is_synthetic=True,
        source="synthetic_forced",
        asset=asset,
    )

    table = build_price_status_table({asset.ticker: item})

    assert table.loc[0, "Referência"] == "12/08/2026"
    assert table.loc[0, "Situação"] == "Simulado"
    assert "não é cotação de mercado" in table.loc[0, "Origem"]


def test_price_status_table_describes_yahoo_as_daily_close():
    asset = Asset("CL=F", "WTI Crude", "Energia", "USD/bbl")
    frame = pd.DataFrame({"Close": [70.0]}, index=pd.to_datetime(["2026-08-12"]))
    item = PriceData(
        df=frame,
        is_synthetic=False,
        source="yahoo_finance",
        asset=asset,
    )

    table = build_price_status_table({asset.ticker: item})

    assert table.loc[0, "Situação"] == "Última observação publicada"
    assert table.loc[0, "Origem"] == "Yahoo Finance — fechamento diário"
