from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
import numpy as np

import src.config as config
import src.data_io as data_io
from src.market_data import get_latest_prices


def _parse_money(value: str | float | int | None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value_str = str(value).strip()
    if value_str == "" or value_str == "-":
        return None
    cleaned = (
        value_str.replace("$", "")
        .replace(",", "")
        .replace(" ", "")
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_money(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"${value:,.2f}"


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Fecha",
        "Costo_Value_Invest",
        "Total_Cierre_Value",
        "Diferencia",
        "P/L_Diario",
        "Compra_de_acciones",
        "Ventas",
        "Liquidez",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]


def main() -> int:
    settings = config.SETTINGS
    hist_path = settings.historical_pnl_path

    tz = ZoneInfo("America/Hermosillo")
    now_ts = pd.Timestamp.now(tz=tz)
    if now_ts.weekday() > 4:
        return 0
    market_open = time(7, 30) <= now_ts.time() <= time(15, 0)
    if not market_open:
        return 0

    positions = data_io.load_positions(settings.positions_path)
    tickers = positions["ticker"].tolist()
    latest_prices = get_latest_prices(tickers)
    latest_asof = latest_prices.attrs.get("asof")
    if latest_asof is None:
        return 0

    asof_day = pd.to_datetime(latest_asof).normalize().date()
    total_value = float(
        latest_prices.reindex(tickers).fillna(0.0).mul(positions["quantity"].values).sum()
    )

    if hist_path.exists():
        hist = pd.read_csv(hist_path, sep="\t")
        hist.columns = [c.strip() for c in hist.columns]
    else:
        hist = pd.DataFrame()
    hist = _ensure_columns(hist)

    hist["_date"] = pd.to_datetime(hist["Fecha"], dayfirst=True, errors="coerce")
    hist = hist.sort_values("_date")

    prior_rows = hist[hist["_date"].notna() & (hist["_date"].dt.date < asof_day)]
    last_cost = None
    prev_close = None
    if not prior_rows.empty:
        last_cost = _parse_money(prior_rows.iloc[-1]["Costo_Value_Invest"])
        prev_close = _parse_money(prior_rows.iloc[-1]["Total_Cierre_Value"])

    today_mask = hist["_date"].dt.date == asof_day
    if not today_mask.any():
        new_row = {col: "" for col in hist.columns if col != "_date"}
        new_row["Fecha"] = pd.Timestamp(asof_day).strftime("%d/%m/%y")
        new_row["_date"] = pd.Timestamp(asof_day)
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        today_mask = hist["_date"].dt.date == asof_day

    buy_val = _parse_money(hist.loc[today_mask, "Compra_de_acciones"].iloc[0]) or 0.0
    sell_val = _parse_money(hist.loc[today_mask, "Ventas"].iloc[0]) or 0.0

    if last_cost is not None:
        cost_today = last_cost + buy_val - sell_val
        if _parse_money(hist.loc[today_mask, "Costo_Value_Invest"].iloc[0]) is None:
            hist.loc[today_mask, "Costo_Value_Invest"] = _format_money(cost_today)

    hist.loc[today_mask, "Total_Cierre_Value"] = _format_money(total_value)

    cost_val = _parse_money(hist.loc[today_mask, "Costo_Value_Invest"].iloc[0])
    if cost_val is not None:
        hist.loc[today_mask, "Diferencia"] = _format_money(total_value - cost_val)

    if prev_close is not None:
        hist.loc[today_mask, "P/L_Diario"] = _format_money(total_value - prev_close)

    hist = hist.drop(columns=["_date"])
    hist.to_csv(hist_path, index=False, sep="\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
