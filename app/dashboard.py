import sys
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import zipfile
import plotly.express as px
import plotly.graph_objects as go
import src.config as config
import src.data_io as data_io
import importlib
from src.market_data import get_latest_prices, get_price_history
from src.analytics import (
    compute_holdings_table,
    daily_returns,
    portfolio_returns,
    annualized_return,
    sharpe_ratio,
)

st.sidebar.header("⚙️ Configuración")
latest_snapshot_date = pd.Timestamp.today().date()
hist_path = (Path(__file__).resolve().parents[1] / "data" / "historical_pnl.tsv")
if hist_path.exists():
    try:
        _hist = pd.read_csv(hist_path, sep="\t")
        _hist.columns = [c.strip() for c in _hist.columns]
        if "Fecha" in _hist.columns:
            _dates = pd.to_datetime(_hist["Fecha"], dayfirst=True, errors="coerce")
        elif "date" in _hist.columns:
            _dates = pd.to_datetime(_hist["date"], errors="coerce")
        else:
            _dates = pd.Series(dtype="datetime64[ns]")
        if _dates.notna().any():
            latest_snapshot_date = _dates.max().date()
    except Exception:
        pass
is_cloud = "/mount/src/" in str(Path(__file__).resolve())
if is_cloud:
    access_code = st.sidebar.text_input("Código de acceso", type="password")
    expected_code = st.secrets.get("APP_CODE", os.getenv("APP_CODE"))
    if not expected_code or access_code != expected_code:
        st.sidebar.warning("Acceso restringido.")
        st.stop()
period = st.sidebar.selectbox(
    "Periodo de análisis",
    options=["6mo", "1y", "3y", "5y"],
    index=1
)
if "snapshot_date" not in st.session_state:
    st.session_state["snapshot_date"] = latest_snapshot_date
elif st.session_state["snapshot_date"] < latest_snapshot_date:
    st.session_state["snapshot_date"] = latest_snapshot_date
cutoff_date = st.sidebar.date_input(
    "Market Opening Day",
    key="snapshot_date",
)
snapshot_view_date = pd.to_datetime(cutoff_date).normalize()
start_year = cutoff_date.year
current_year = pd.Timestamp.today().year
year_options = ["Todos"] + [str(y) for y in range(start_year, current_year + 2)]
year_filter = st.sidebar.selectbox(
    "Año de análisis",
    options=year_options,
    index=1 if str(start_year) in year_options else 0,
)

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

SETTINGS = config.SETTINGS
data_io = importlib.reload(data_io)

# --- Cargar datos base ---
positions = data_io.load_positions(SETTINGS.positions_path)
tickers = positions["ticker"].unique().tolist()
all_tickers = tickers + [SETTINGS.benchmark]

# --- Precios actuales ---
latest_prices = get_latest_prices(all_tickers)
latest_asof = latest_prices.attrs.get("asof")
missing_prices = latest_prices[latest_prices.isna()].index.tolist()
available_prices = latest_prices.dropna()
benchmark_price = latest_prices.get(SETTINGS.benchmark)
if pd.notna(benchmark_price) and latest_asof is not None:
    bench_date = pd.to_datetime(latest_asof).normalize()
    if bench_date.weekday() < 5:
        data_io.save_benchmark_close_csv(
            SETTINGS.benchmark_closes_path,
            bench_date,
            float(benchmark_price),
        )
if missing_prices:
    positions = positions[~positions["ticker"].isin(missing_prices)]
    tickers = positions["ticker"].unique().tolist()
    all_tickers = tickers + [SETTINGS.benchmark]
available_prices.attrs["asof"] = latest_asof
holdings = compute_holdings_table(positions, available_prices)

# --- Histórico ---
price_history = get_price_history(
    all_tickers,
    period,
    SETTINGS.history_interval
)
snapshot_date = None
if not price_history.empty:
    snapshot_date = pd.to_datetime(price_history.index.max())
    close_prices = price_history.loc[snapshot_date, tickers]
else:
    snapshot_date = pd.Timestamp.today().normalize()
    close_prices = available_prices.reindex(tickers)
close_prices = close_prices.astype(float)
holdings_snapshot = positions.copy()
holdings_snapshot["close_price"] = holdings_snapshot["ticker"].map(close_prices.to_dict())
holdings_snapshot = holdings_snapshot.dropna(subset=["close_price"])
holdings_snapshot["cost_value"] = holdings_snapshot["quantity"] * holdings_snapshot["avg_price"]
holdings_snapshot["market_value"] = holdings_snapshot["quantity"] * holdings_snapshot["close_price"]
holdings_snapshot["pnl"] = holdings_snapshot["market_value"] - holdings_snapshot["cost_value"]
holdings_snapshot["pnl_pct"] = np.where(
    holdings_snapshot["cost_value"] > 0,
    (holdings_snapshot["pnl"] / holdings_snapshot["cost_value"]) * 100.0,
    np.nan,
)
snapshot_totals = {
    "total_value": float(holdings_snapshot["market_value"].sum()),
    "total_cost": float(holdings_snapshot["cost_value"].sum()),
    "total_pnl": float(holdings_snapshot["pnl"].sum()),
}
snapshot_totals["total_pnl_pct"] = (
    (snapshot_totals["total_pnl"] / snapshot_totals["total_cost"]) * 100.0
    if snapshot_totals["total_cost"]
    else 0.0
)
holdings_snapshot_export = holdings_snapshot[
    [
        "ticker",
        "quantity",
        "avg_price",
        "close_price",
        "cost_value",
        "market_value",
        "pnl",
        "pnl_pct",
    ]
]
last_market_update = None
if latest_asof is not None:
    last_market_update = pd.to_datetime(latest_asof)
elif not price_history.empty:
    last_market_update = pd.to_datetime(price_history.index.max())

now_ts = pd.Timestamp.now(tz=ZoneInfo("America/Hermosillo"))
market_open = False
if last_market_update is not None:
    in_market_day = now_ts.weekday() <= 4
    start_time = now_ts.replace(hour=7, minute=30, second=0, microsecond=0)
    end_time = now_ts.replace(hour=15, minute=0, second=0, microsecond=0)
    market_open = in_market_day and start_time <= now_ts <= end_time
status_color = "#52c41a" if market_open else "#ff4d4f"
status_class = "market-open" if market_open else "market-closed"
st.markdown(
    """
    <style>
    .title-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .title-row h1 {
        margin: 0;
    }
    .market-indicator-inline {
        display: inline-flex;
        align-items: center;
        margin-left: 10px;
    }
    .market-dot {
        width: 0.55em;
        height: 0.55em;
        border-radius: 50%;
        background: var(--dot-color);
        box-shadow: 0 0 8px var(--dot-color);
    }
    .market-open .market-dot {
        animation: pulse 1.2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); opacity: 0.7; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='title-row'>"
    f"<h1>📊 Portfolio Analytics — Live"
    f"<span class='market-indicator-inline {status_class}' style='--dot-color: {status_color};'>"
    f"<span class='market-dot'></span>"
    f"</span>"
    f"</h1>"
    f"</div>",
    unsafe_allow_html=True,
)

# --- Selector de activos ---
selected_tickers = tickers

returns = daily_returns(price_history)

benchmark_ret = returns[SETTINGS.benchmark]
filtered_holdings = holdings[holdings["ticker"].isin(selected_tickers)]
asset_rets = returns[selected_tickers]

weights = filtered_holdings.set_index("ticker")["market_value"]
weights = weights / weights.sum()

portfolio_ret = portfolio_returns(asset_rets, weights)

# --- Métricas clave ---
total_value = holdings["market_value"].sum()
total_pnl = holdings["pnl"].sum()
total_cost = holdings["cost_value"].sum() if "cost_value" in holdings.columns else float("nan")
pnl_total_pct = (total_pnl / total_cost) * 100.0 if total_cost else 0.0
ann_ret = annualized_return(portfolio_ret)
sharpe = sharpe_ratio(portfolio_ret, SETTINGS.risk_free_rate_annual)

def compute_xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    if len(cashflows) < 2:
        return float("nan")
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]
    times = np.array([(cf_date - t0).days / 365.0 for cf_date, _ in cashflows], dtype=float)
    amounts = np.array([cf_amt for _, cf_amt in cashflows], dtype=float)
    guess = 0.1
    for _ in range(100):
        denom = (1.0 + guess) ** times
        f = np.sum(amounts / denom)
        df = np.sum(-times * amounts / denom / (1.0 + guess))
        if df == 0:
            break
        new_guess = guess - f / df
        if abs(new_guess - guess) < 1e-8:
            return float(new_guess)
        guess = new_guess
    return float("nan")

# --- Layout ---
col1, col2, col3, col4, col5, col6 = st.columns([1.6, 1.3, 1.3, 1.3, 1.1, 1.3])

col1.metric("Valor total", f"${total_value:,.2f}")
col2.metric("P/L total", f"${total_pnl:,.2f}")
col3.markdown(
    "<div style='font-size:0.9rem; color:#9e9e9e;'>P/L total (%)</div>"
    f"<div style='font-size:2rem; font-weight:600; color:#52c41a;'>{pnl_total_pct:.2f}%</div>",
    unsafe_allow_html=True,
)
col4.metric("Retorno anual (est.)", f"{ann_ret*100:.2f}%")
sharpe_color = "#9e9e9e"
if pd.notna(sharpe):
    if sharpe < 0.5:
        sharpe_color = "#ff4d4f"
    elif sharpe < 1.0:
        sharpe_color = "#faad14"
    elif sharpe < 1.5:
        sharpe_color = "#52c41a"
    elif sharpe < 2.0:
        sharpe_color = "#1890ff"
    else:
        sharpe_color = "#722ed1"
col5.markdown(
    f"<div style='font-size:0.9rem; color:#9e9e9e;'>Sharpe</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{sharpe_color};'>{sharpe:.2f}</div>",
    unsafe_allow_html=True,
)
transactions_path = ROOT_DIR / "data" / "transactions.csv"
if transactions_path.exists():
    tx_df = pd.read_csv(transactions_path)
    tx_df["date"] = pd.to_datetime(tx_df["date"])
    tx_df["total"] = pd.to_numeric(tx_df["total"], errors="coerce")
    cashflows = [(d, -amt) for d, amt in zip(tx_df["date"], tx_df["total"]) if pd.notna(amt)]
    cashflows.append((pd.Timestamp.today().normalize(), float(total_value)))
    xirr = compute_xirr(cashflows)
    liquidity = 0.0
else:
    tx_df = pd.DataFrame()
    xirr = float("nan")
    liquidity = 0.0

col6.markdown(
    "<div style='font-size:0.9rem; color:#9e9e9e;'>XIRR anual</div>"
    f"<div style='font-size:2rem; font-weight:600; color:#52c41a;'>{xirr*100:.2f}%</div>",
    unsafe_allow_html=True,
)
if last_market_update is not None:
    in_market_day = now_ts.weekday() <= 4
    start_time = now_ts.replace(hour=7, minute=30, second=0, microsecond=0)
    end_time = now_ts.replace(hour=15, minute=0, second=0, microsecond=0)
    in_market_hours = in_market_day and start_time <= now_ts <= end_time
    if in_market_hours:
        merged_ts = last_market_update.replace(
            hour=now_ts.hour,
            minute=now_ts.minute,
            second=now_ts.second,
            microsecond=0,
        )
    else:
        merged_ts = last_market_update
    st.caption(
        f"Última actualización de mercado: {merged_ts.strftime('%Y-%m-%d %H:%M:%S')}"
    )

hist_pnl_df = pd.DataFrame()
hist_pnl_full = pd.DataFrame()
if hasattr(SETTINGS, "historical_pnl_path") and SETTINGS.historical_pnl_path.exists():
    hist_raw = pd.read_csv(SETTINGS.historical_pnl_path, sep="\t")
    hist_raw.columns = [c.strip() for c in hist_raw.columns]
    date_col = "Fecha" if "Fecha" in hist_raw.columns else None
    close_col = "Total Cierre" if "Total Cierre" in hist_raw.columns else None
    if close_col is None and "Total_Cierre_Value" in hist_raw.columns:
        close_col = "Total_Cierre_Value"
    cost_col = "Costo Value Invest" if "Costo Value Invest" in hist_raw.columns else None
    if cost_col is None and "Costo_Value_Invest" in hist_raw.columns:
        cost_col = "Costo_Value_Invest"
    pnl_col = "Ganancia o perdida diaria" if "Ganancia o perdida diaria" in hist_raw.columns else None
    if pnl_col is None and "P/L_Diario" in hist_raw.columns:
        pnl_col = "P/L_Diario"
    diff_col = "Diferencia" if "Diferencia" in hist_raw.columns else None
    buy_col = "Compra_de_acciones" if "Compra_de_acciones" in hist_raw.columns else None
    sell_col = "Ventas" if "Ventas" in hist_raw.columns else None
    if date_col and close_col:
        hist_raw["_date"] = pd.to_datetime(hist_raw[date_col], dayfirst=True, errors="coerce")
        for col in [close_col, cost_col, pnl_col, diff_col, buy_col, sell_col]:
            if col and col in hist_raw.columns:
                hist_raw[col] = (
                    hist_raw[col]
                    .astype(str)
                    .str.replace(r"[^0-9.\-]", "", regex=True)
                    .str.replace(r"^-$", "", regex=True)
                )
                hist_raw[col] = pd.to_numeric(hist_raw[col], errors="coerce")

        hist_pnl_full = hist_raw.dropna(subset=["_date"]).sort_values("_date")
        hist_pnl_full["date"] = hist_pnl_full["_date"]
        if not hist_pnl_full.empty:
            live_day = pd.Timestamp.today().normalize()
            last_row = hist_pnl_full.loc[hist_pnl_full["_date"] < live_day].sort_values("_date")
            last_cost = float(last_row.iloc[-1][cost_col]) if (cost_col and not last_row.empty) else float("nan")
            prev_close = float(last_row.iloc[-1][close_col]) if not last_row.empty else float("nan")

            today_mask = hist_pnl_full["_date"] == live_day
            if not today_mask.any():
                new_row = {date_col: live_day.strftime("%d/%m/%y"), "_date": live_day}
                hist_pnl_full = pd.concat([hist_pnl_full, pd.DataFrame([new_row])], ignore_index=True)
                today_mask = hist_pnl_full["_date"] == live_day

            if cost_col:
                buy_val = float(hist_pnl_full.loc[today_mask, buy_col].iloc[0]) if buy_col else 0.0
                sell_val = float(hist_pnl_full.loc[today_mask, sell_col].iloc[0]) if sell_col else 0.0
                if not np.isnan(last_cost):
                    cost_today = last_cost + buy_val - sell_val
                else:
                    cost_today = last_cost
                if np.isnan(hist_pnl_full.loc[today_mask, cost_col]).all():
                    hist_pnl_full.loc[today_mask, cost_col] = cost_today

            hist_pnl_full.loc[today_mask, close_col] = float(total_value)
            if diff_col and cost_col:
                hist_pnl_full.loc[today_mask, diff_col] = (
                    hist_pnl_full.loc[today_mask, close_col] - hist_pnl_full.loc[today_mask, cost_col]
                )
            if pnl_col and not np.isnan(prev_close):
                hist_pnl_full.loc[today_mask, pnl_col] = float(total_value) - prev_close

            if date_col:
                out_df = hist_pnl_full.copy()
                out_df[date_col] = out_df["_date"].dt.strftime("%d/%m/%y")
                out_df = out_df.drop(columns=["_date", "date"], errors="ignore")
                money_cols = [c for c in [cost_col, close_col, diff_col, pnl_col, buy_col, sell_col, "Liquidez"] if c and c in out_df.columns]
                for col in money_cols:
                    out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
                    out_df[col] = out_df[col].apply(lambda x: "" if pd.isna(x) else f"${x:,.2f}")
                out_df.to_csv(SETTINGS.historical_pnl_path, index=False, sep="\t")

        hist_pnl_full = hist_pnl_full.drop(columns=["_date"])
        hist_pnl_df = hist_pnl_full.copy()

manual_daily_pnl = None
prices_hist = price_history[selected_tickers].copy()
prices_hist.index = pd.to_datetime(prices_hist.index)
if prices_hist.index.tz is not None:
    prices_hist.index = prices_hist.index.tz_localize(None)
today = pd.Timestamp.today().normalize()
prices_hist = prices_hist.loc[prices_hist.index < today]
prices_hist = prices_hist.loc[prices_hist.index.weekday < 5]
prices_hist = prices_hist.ffill()
qty = positions[positions["ticker"].isin(selected_tickers)].set_index("ticker")["quantity"]
daily_value_market_full = prices_hist.mul(qty, axis=1).sum(axis=1)
daily_value_market = daily_value_market_full.loc[daily_value_market_full.index >= pd.Timestamp(cutoff_date)]
if latest_asof is not None:
    latest_asof = pd.to_datetime(latest_asof)
    if latest_asof.tzinfo is not None:
        latest_asof = latest_asof.tz_localize(None)
    if latest_asof not in daily_value_market_full.index:
        latest_total = available_prices.reindex(qty.index).mul(qty).sum()
        daily_value_market_full.loc[latest_asof] = float(latest_total)
        daily_value_market_full = daily_value_market_full.sort_index()
        daily_value_market = daily_value_market_full.loc[daily_value_market_full.index >= pd.Timestamp(cutoff_date)]

active_year = int(year_filter) if year_filter != "Todos" else 2026
close_key = "Total Cierre" if "Total Cierre" in hist_pnl_df.columns else None
if close_key is None and "Total_Cierre_Value" in hist_pnl_df.columns:
    close_key = "Total_Cierre_Value"
cost_key = "Costo Value Invest" if "Costo Value Invest" in hist_pnl_df.columns else None
if cost_key is None and "Costo_Value_Invest" in hist_pnl_df.columns:
    cost_key = "Costo_Value_Invest"

use_broker_year = close_key is not None and cost_key is not None and not hist_pnl_df.empty
if use_broker_year:
    year_rows = hist_pnl_df[hist_pnl_df["date"].dt.year == active_year]
    if not year_rows.empty:
        first_row = year_rows.iloc[0]
        if "P/L_Diario" in year_rows.columns:
            year_pl = float(year_rows["P/L_Diario"].sum())
        else:
            last_row = year_rows.iloc[-1]
            year_pl = float(last_row[close_key] - last_row[cost_key])
        base_close = float(first_row[close_key])
        year_pl_pct = (year_pl / base_close) * 100.0 if base_close else 0.0
    else:
        year_pl = 0.0
        year_pl_pct = 0.0
else:
    year_series = daily_value_market_full.loc[daily_value_market_full.index.year == active_year]
    if len(year_series) >= 1:
        year_first = float(year_series.iloc[0])
        year_last = float(year_series.iloc[-1])
        prev_close_series = daily_value_market_full.loc[daily_value_market_full.index < pd.Timestamp(active_year, 1, 1)]
        if not prev_close_series.empty:
            prev_close = float(prev_close_series.iloc[-1])
            year_pl = year_last - prev_close
            year_pl_pct = (year_pl / prev_close) * 100.0 if prev_close else 0.0
        else:
            year_pl = year_last - year_first
            year_pl_pct = (year_pl / year_first) * 100.0 if year_first else 0.0
    else:
        year_pl = 0.0
        year_pl_pct = 0.0
year_color = "#52c41a" if year_pl > 0 else ("#ff4d4f" if year_pl < 0 else "#9e9e9e")
year_pct_color = "#52c41a" if year_pl_pct > 0 else ("#ff4d4f" if year_pl_pct < 0 else "#9e9e9e")

aport_net = 0.0
aport_pct = 0.0
if not hist_pnl_df.empty and "date" in hist_pnl_df.columns:
    buy_col = "Compra_de_acciones" if "Compra_de_acciones" in hist_pnl_df.columns else None
    sell_col = "Ventas" if "Ventas" in hist_pnl_df.columns else None
    if buy_col and sell_col:
        year_rows = hist_pnl_df[hist_pnl_df["date"].dt.year == active_year]
        aport_net = float(year_rows[buy_col].fillna(0.0).sum()) - float(
            year_rows[sell_col].fillna(0.0).sum()
        )
        if total_value:
            aport_pct = (aport_net / float(total_value)) * 100.0
aport_color = "#52c41a" if aport_net > 0 else ("#ff4d4f" if aport_net < 0 else "#9e9e9e")
aport_pct_color = "#52c41a" if aport_pct > 0 else ("#ff4d4f" if aport_pct < 0 else "#9e9e9e")
aport_pct_bg = (
    "rgba(82, 196, 26, 0.18)"
    if aport_pct > 0
    else ("rgba(255, 77, 79, 0.18)" if aport_pct < 0 else "rgba(158, 158, 158, 0.18)")
)
aport_pct_label = f"↑ {aport_pct:.2f}%" if aport_pct > 0 else (f"↓ {abs(aport_pct):.2f}%" if aport_pct < 0 else f"{aport_pct:.2f}%")

col_y1, col_y2, col_y3, col_y4, col_y5, col_y6 = st.columns([1.6, 1.3, 1.3, 1.3, 1.1, 1.3])
col_y1.markdown(
    "<div style='height:110px; display:flex; flex-direction:column; align-items:flex-start; justify-content:space-between; padding-left:6px;'>"
    f"<div style='font-size:0.9rem; color:#9e9e9e;'>P/L {active_year}</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{year_color};'>${year_pl:,.2f}</div>"
    "<div style='height:24px;'></div>"
    "</div>",
    unsafe_allow_html=True,
)
col_y2.markdown(
    "<div style='height:110px; display:flex; flex-direction:column; align-items:flex-start; justify-content:space-between; padding-left:6px;'>"
    f"<div style='font-size:0.9rem; color:#9e9e9e;'>P/L {active_year} (%)</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{year_pct_color};'>{year_pl_pct:.2f}%</div>"
    "<div style='height:24px;'></div>"
    "</div>",
    unsafe_allow_html=True,
)
liquidity_color = "#52c41a" if liquidity > 0 else "#9e9e9e"
col_y3.markdown(
    "<div style='height:110px; display:flex; flex-direction:column; align-items:flex-start; justify-content:space-between; padding-left:6px;'>"
    "<div style='font-size:0.9rem; color:#9e9e9e;'>Liquidez</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{liquidity_color};'>${liquidity:,.2f}</div>"
    "<div style='height:24px;'></div>"
    "</div>",
    unsafe_allow_html=True,
)
col_y4.markdown(
    "<div style='height:110px; display:flex; flex-direction:column; align-items:flex-start; justify-content:space-between; padding-left:6px;'>"
    f"<div style='font-size:0.9rem; color:#9e9e9e;'>Aportaciones {active_year}</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{aport_color};'>${aport_net:,.2f}</div>"
    f"<div style='margin-top:4px; display:inline-block; padding:2px 8px; border-radius:999px; background:{aport_pct_bg}; color:{aport_pct_color}; font-size:0.9rem; font-weight:600;'>{aport_pct_label}</div>"
    "</div>",
    unsafe_allow_html=True,
)
col_y5.markdown("<div style='height:110px;'></div>", unsafe_allow_html=True)
col_y6.markdown("<div style='height:110px;'></div>", unsafe_allow_html=True)

if not hist_pnl_df.empty:
    if "Total Cierre" in hist_pnl_df.columns:
        close_key = "Total Cierre"
    elif "Total_Cierre_Value" in hist_pnl_df.columns:
        close_key = "Total_Cierre_Value"
    else:
        close_key = None
    daily_value_full = hist_pnl_df.set_index("date")[close_key].copy() if close_key else pd.Series(dtype=float)
    daily_value_full = daily_value_full.loc[daily_value_full.index.weekday < 5]
    daily_value = daily_value_full.copy()
    if "Ganancia o perdida diaria" in hist_pnl_df.columns:
        manual_daily_pnl = hist_pnl_df.set_index("date")["Ganancia o perdida diaria"].copy()
    elif "P/L_Diario" in hist_pnl_df.columns:
        manual_daily_pnl = hist_pnl_df.set_index("date")["P/L_Diario"].copy()
    manual_daily_pnl = manual_daily_pnl.loc[manual_daily_pnl.index.weekday < 5]
    manual_daily_pnl = manual_daily_pnl.copy()
else:
    daily_value_full = daily_value_market_full
    daily_value = daily_value_market

filtered_daily_value = daily_value
if year_filter != "Todos":
    filtered_daily_value = daily_value[daily_value.index.year == int(year_filter)]

filtered_daily_value_market = daily_value_market
if year_filter != "Todos":
    filtered_daily_value_market = daily_value_market[daily_value_market.index.year == int(year_filter)]

if len(daily_value_market_full) >= 1:
    base_value = float(daily_value_market_full.iloc[-1])
    base_date = daily_value_market_full.index[-1]
    if len(daily_value_market_full) >= 2:
        prev_value = float(daily_value_market_full.iloc[-2])
        prev_date = daily_value_market_full.index[-2]
        last_close_pnl = base_value - prev_value
        live_pnl = last_close_pnl
        live_pnl_pct = (live_pnl / prev_value) * 100.0 if prev_value else 0.0
    else:
        prev_value = float("nan")
        prev_date = None
        last_close_pnl = 0.0
        live_pnl = 0.0
        live_pnl_pct = 0.0
else:
    live_pnl = 0.0
    live_pnl_pct = 0.0
    last_close_pnl = 0.0
    base_date = None
    prev_date = None

if snapshot_date is not None and not holdings_snapshot_export.empty:
    data_io.save_daily_holdings_sqlite(
        SETTINGS.snapshots_db,
        snapshot_date,
        holdings_snapshot_export,
    )
    summary_payload = {
        "total_value": snapshot_totals["total_value"],
        "total_cost": snapshot_totals["total_cost"],
        "total_pnl": snapshot_totals["total_pnl"],
        "total_pnl_pct": snapshot_totals["total_pnl_pct"],
        "liquidity": liquidity,
        "live_pnl": live_pnl,
        "live_pnl_pct": live_pnl_pct,
        "last_close_pnl": last_close_pnl,
        "xirr": xirr,
        "sharpe": sharpe if pd.notna(sharpe) else 0.0,
        "ann_return": ann_ret,
        "benchmark": SETTINGS.benchmark,
        "benchmark_close": float(benchmark_price) if pd.notna(benchmark_price) else None,
    }
    data_io.save_daily_summary_sqlite(
        SETTINGS.snapshots_db,
        snapshot_date,
        summary_payload,
    )

live_col1, live_col2, live_col3, live_col4, live_col5, live_col6 = st.columns([1.6, 1.3, 1.3, 1.3, 1.1, 1.3])
live_color = "#52c41a" if live_pnl > 0 else ("#ff4d4f" if live_pnl < 0 else "#9e9e9e")
live_pct_color = "#52c41a" if live_pnl_pct > 0 else ("#ff4d4f" if live_pnl_pct < 0 else "#9e9e9e")
last_close_color = "#52c41a" if last_close_pnl > 0 else ("#ff4d4f" if last_close_pnl < 0 else "#9e9e9e")
live_col1.markdown(
    "<div style='height:110px; display:flex; flex-direction:column; align-items:flex-start; justify-content:space-between; padding-left:6px;'>"
    "<div style='font-size:0.9rem; color:#9e9e9e;'>P/L en vivo</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{live_color};'>${live_pnl:,.2f}</div>"
    "<div style='height:24px;'></div>"
    "</div>",
    unsafe_allow_html=True,
)
live_col2.markdown(
    "<div style='height:110px; display:flex; flex-direction:column; align-items:flex-start; justify-content:space-between; padding-left:6px;'>"
    "<div style='font-size:0.9rem; color:#9e9e9e;'>P/L en vivo (%)</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{live_pct_color};'>{live_pnl_pct:.2f}%</div>"
    "<div style='height:24px;'></div>"
    "</div>",
    unsafe_allow_html=True,
)
live_col3.markdown(
    "<div style='height:110px; display:flex; flex-direction:column; align-items:flex-start; justify-content:space-between; padding-left:6px;'>"
    "<div style='font-size:0.9rem; color:#9e9e9e;'>P/L último cierre</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{last_close_color};'>${last_close_pnl:,.2f}</div>"
    "<div style='height:24px;'></div>"
    "</div>",
    unsafe_allow_html=True,
)
live_col5.markdown("<div style='height:110px;'></div>", unsafe_allow_html=True)
live_col6.markdown("<div style='height:110px;'></div>", unsafe_allow_html=True)
if year_filter != "Todos":
    st.caption(f"Año activo: {year_filter}")
if base_date is not None:
    source_label = "datos de mercado"
    if prev_date is not None:
        st.caption(
            f"Último cierre: {base_date.date().isoformat()} · Base P/L cierre: "
            f"{prev_date.date().isoformat()} ${prev_value:,.2f} → {base_date.date().isoformat()} ${base_value:,.2f} "
            f"({source_label})"
        )
    else:
        prev_full_value = None
        prev_full_date = None
        if daily_value_market_full is not None and len(daily_value_market_full) >= 2:
            prev_full_value = float(daily_value_market_full.iloc[-2])
            prev_full_date = daily_value_market_full.index[-2]
        st.caption(
            f"Último cierre: {base_date.date().isoformat()} · P/L cierre: ${last_close_pnl:,.2f} "
            f"· Base cierre: ${base_value:,.2f}"
            + (
                f" · Cierre anterior: {prev_full_date.date().isoformat()} ${prev_full_value:,.2f}"
                if prev_full_date is not None
                else ""
            )
            + f" ({source_label})"
        )

st.subheader("📋 Holdings")
if not filtered_holdings.empty:
    def _pnl_color(val):
        if pd.isna(val):
            return ""
        return "color: #52c41a;" if val > 0 else ("color: #ff4d4f;" if val < 0 else "")

    display_holdings = filtered_holdings.copy()
    display_holdings.columns = [c.replace("_", " ").title() for c in display_holdings.columns]
    center_styles = [
        {"selector": "table", "props": [("margin", "0 auto")]},
        {"selector": "th, td", "props": [("text-align", "center"), ("font-size", "0.9rem")]},
        {"selector": "thead th", "props": [("text-align", "center")]},
        {"selector": "tbody th", "props": [("text-align", "center")]},
        {"selector": ".row_heading", "props": [("text-align", "center")]},
        {"selector": ".col_heading", "props": [("text-align", "center")]},
        {"selector": ".index_name", "props": [("text-align", "center")]},
        {"selector": ".blank", "props": [("text-align", "center")]},
    ]
    styler = display_holdings.style
    styler = styler.set_properties(**{"text-align": "center"}, subset=pd.IndexSlice[:, :])
    styler = styler.set_table_styles(center_styles)
    format_map = {}
    for col in display_holdings.columns:
        if not pd.api.types.is_numeric_dtype(display_holdings[col]):
            continue
        if "Pct" in col:
            format_map[col] = "{:.2f}%"
        elif any(k in col for k in ["Price", "Value", "Pnl"]):
            format_map[col] = "${:,.2f}"
        else:
            format_map[col] = "{:.2f}"
    if format_map:
        styler = styler.format(format_map)
    if "pnl" in filtered_holdings.columns:
        styler = styler.applymap(_pnl_color, subset=["Pnl"])
    if "pnl_pct" in filtered_holdings.columns:
        styler = styler.applymap(_pnl_color, subset=["Pnl Pct"])
    st.dataframe(styler, use_container_width=True)
    weights_pct = filtered_holdings.set_index("ticker")["weight_pct"]
    hhi = float((weights_pct / 100.0).pow(2).sum())
    st.caption(f"HHI (concentración): {hhi:.3f}")

    st.subheader("📈 Tickets en vivo")
    prev_close = price_history[tickers].ffill().iloc[-2] if len(price_history) >= 2 else pd.Series(dtype=float)
    qty_map = positions.set_index("ticker")["quantity"]
    live_view = pd.DataFrame({
        "ticker": filtered_holdings["ticker"],
    })
    live_view["precio_actual"] = live_view["ticker"].map(available_prices)
    live_view["precio_prev"] = live_view["ticker"].map(prev_close)
    live_view["cantidad"] = live_view["ticker"].map(qty_map)
    live_view["valor_portafolio"] = live_view["precio_actual"] * live_view["cantidad"]
    live_view["ganancia"] = (live_view["precio_actual"] - live_view["precio_prev"]) * live_view["cantidad"]
    live_view["ganancia_pct"] = (
        (live_view["precio_actual"] - live_view["precio_prev"]) / live_view["precio_prev"]
    ) * 100.0
    live_view = live_view.drop(columns=["precio_prev", "cantidad"]).sort_values("ganancia", ascending=False)
    live_view.columns = [c.replace("_", " ").title() for c in live_view.columns]
    live_styler = live_view.style
    live_styler = live_styler.set_properties(**{"text-align": "center"}, subset=pd.IndexSlice[:, :])
    live_styler = live_styler.set_table_styles(center_styles)
    live_format = {}
    for col in live_view.columns:
        if not pd.api.types.is_numeric_dtype(live_view[col]):
            continue
        if "Pct" in col:
            live_format[col] = "{:.2f}%"
        elif any(k in col for k in ["Precio", "Valor", "Ganancia"]):
            live_format[col] = "${:,.2f}"
        else:
            live_format[col] = "{:.2f}"
    if live_format:
        live_styler = live_styler.format(live_format)
    live_styler = live_styler.applymap(_pnl_color, subset=["Ganancia", "Ganancia Pct"])
    st.dataframe(live_styler, use_container_width=True)

    if not tx_df.empty:
        with st.expander("📑 Flujos para XIRR", expanded=False):
            tx_view = tx_df.copy()
            tx_view["total"] = tx_view["total"].astype(float)
            tx_view = tx_view.rename(
                columns={
                    "date": "Fecha",
                    "stock": "Ticker",
                    "quantity": "Cantidad",
                    "price": "Precio",
                    "total": "Total",
                }
            )
            tx_view["Fecha"] = tx_view["Fecha"].dt.strftime("%Y-%m-%d")
            tx_view.columns = [c.replace("_", " ").title() for c in tx_view.columns]
            tx_styler = tx_view.style
            tx_styler = tx_styler.set_table_styles(center_styles)
            tx_format = {}
            for col in tx_view.columns:
                if not pd.api.types.is_numeric_dtype(tx_view[col]):
                    continue
                if col in ["Precio", "Total"]:
                    tx_format[col] = "${:,.2f}"
                else:
                    tx_format[col] = "{:.2f}"
            if tx_format:
                tx_styler = tx_styler.format(tx_format)
            st.dataframe(tx_styler, use_container_width=True)

            cf_df = pd.DataFrame(cashflows, columns=["date", "amount"])
            cf_df = cf_df.sort_values("date")
            cf_df = cf_df[cf_df["amount"] < 0]
            cf_df["cumulative"] = cf_df["amount"].cumsum()
            cf_df = cf_df.groupby("date", as_index=False)["cumulative"].last()
            use_hist_cost = (
                not hist_pnl_df.empty
                and "date" in hist_pnl_df.columns
                and (
                    "Costo_Value_Invest" in hist_pnl_df.columns
                    or "Costo Value Invest" in hist_pnl_df.columns
                )
            )
            if use_hist_cost:
                cost_col = (
                    "Costo_Value_Invest"
                    if "Costo_Value_Invest" in hist_pnl_df.columns
                    else "Costo Value Invest"
                )
                broker_cost = hist_pnl_df[["date", cost_col]].dropna()
                broker_cost = broker_cost[broker_cost["date"] >= pd.Timestamp(cutoff_date)]
                broker_cost = broker_cost.rename(columns={cost_col: "cumulative"})
                broker_cost["cumulative"] = -broker_cost["cumulative"].astype(float)
                broker_cost = broker_cost.groupby("date", as_index=False)["cumulative"].last()
                full_index = pd.Index(cf_df["date"]).union(broker_cost["date"])
                cf_df = cf_df.set_index("date").reindex(full_index)
                cf_df.loc[broker_cost["date"], "cumulative"] = broker_cost["cumulative"].values
                cf_df = cf_df.reset_index().rename(columns={"index": "date"})
            cf_df = cf_df.sort_values("date")
            cf_df["date_str"] = pd.to_datetime(cf_df["date"]).dt.strftime("%Y-%m-%d")
            fig_cf = px.line(
                cf_df,
                x="date_str",
                y="cumulative",
                markers=True,
                labels={"cumulative": "Flujo acumulado", "date_str": "Fecha"},
            )
            fig_cf.update_traces(
                marker=dict(size=6, color="#ff4d4f"),
                hovertemplate="Fecha: %{x}<br>Flujo acumulado: %{y:.2f}<extra></extra>",
            )
            if not cf_df.empty:
                y_min = float(pd.to_numeric(cf_df["cumulative"], errors="coerce").min())
                y_max = float(pd.to_numeric(cf_df["cumulative"], errors="coerce").max())
                y_pad = max((y_max - y_min) * 0.08, 50.0)
                fig_cf.update_yaxes(range=[y_min - y_pad, y_max + y_pad])
                x_min = pd.to_datetime(cf_df["date"]).min()
                x_max = pd.to_datetime(cf_df["date"]).max()
                if pd.notna(x_min) and pd.notna(x_max):
                    x_pad = pd.Timedelta(days=7)
                    fig_cf.update_xaxes(range=[x_min - x_pad, x_max + x_pad])
            st.plotly_chart(fig_cf, use_container_width=True)
    else:
        st.info("No hay transacciones para calcular XIRR.")

    with st.expander("📅 Histórico de cierres diarios", expanded=False):
        if daily_value_full is None or daily_value_full.empty:
            st.info("No hay cierres diarios para mostrar.")
        else:
            close_series_table = daily_value_full.copy()
            close_series_table = close_series_table.loc[close_series_table.index.weekday < 5]
            close_series_table = close_series_table.sort_index()
            year_options_table = sorted(
                [y for y in close_series_table.index.year.unique().tolist() if y >= 2026]
            )
            if not year_options_table:
                year_options_table = [pd.Timestamp.today().year]
            selected_year_table = st.selectbox(
                "Año",
                options=year_options_table,
                index=len(year_options_table) - 1,
                key="close_table_year",
            )
            close_series_year = close_series_table[
                close_series_table.index.year == int(selected_year_table)
            ]
            if close_series_year.empty:
                st.info("No hay cierres diarios para el año seleccionado.")
            else:
                open_series_table = close_series_year.shift(1).fillna(close_series_year)
                if manual_daily_pnl is not None and not manual_daily_pnl.empty:
                    pl_series_table = manual_daily_pnl.reindex(close_series_year.index)
                else:
                    pl_series_table = close_series_year - open_series_table
                daily_df_table = pd.DataFrame(
                    {
                        "date": close_series_year.index,
                        "open": open_series_table.values,
                        "close": close_series_year.values,
                        "pl": pl_series_table.values,
                    }
                ).dropna()
                pl_table = daily_df_table.copy()
                pl_table["Fecha"] = pl_table["date"].dt.strftime("%Y-%m-%d")
                pl_table = pl_table.rename(
                    columns={
                        "open": "Open (prev cierre)",
                        "close": "Close",
                        "pl": "P/L",
                    }
                )
                pl_table = pl_table[["Fecha", "Open (prev cierre)", "Close", "P/L"]]
                pl_center_styles = [
                    {"selector": "th", "props": [("text-align", "center")]},
                    {"selector": "th.row_heading", "props": [("text-align", "center")]},
                    {"selector": "th.col_heading", "props": [("text-align", "center")]},
                ]
                pl_styler = (
                    pl_table.style
                    .set_properties(**{"text-align": "center"})
                    .set_table_styles(pl_center_styles)
                    .hide(axis="index")
                    .format(
                        {
                            "Open (prev cierre)": "${:,.2f}",
                            "Close": "${:,.2f}",
                            "P/L": "${:,.2f}",
                        }
                    )
                )
                def _pnl_color_table(val):
                    if pd.isna(val):
                        return ""
                    return "color: #52c41a;" if val > 0 else ("color: #ff4d4f;" if val < 0 else "")
                pl_styler = pl_styler.applymap(_pnl_color_table, subset=["P/L"])
                st.table(pl_styler)

    with st.expander("📅 Histórico real (transacciones)", expanded=False):
        if tx_df.empty:
            st.info("No hay transacciones para construir el histórico real.")
        else:
            hist_prices = price_history[tickers].ffill().dropna(how="all")
            tx_hist = tx_df.copy()
            tx_hist["stock"] = tx_hist["stock"].astype(str).str.upper().str.strip()
            tx_hist["date"] = pd.to_datetime(tx_hist["date"])
            tx_hist["quantity"] = pd.to_numeric(tx_hist["quantity"], errors="coerce")
            tx_hist["price"] = pd.to_numeric(tx_hist["price"], errors="coerce")
            tx_hist["total"] = pd.to_numeric(tx_hist["total"], errors="coerce")
            tx_hist = tx_hist.dropna(subset=["date", "stock", "quantity", "price", "total"])
            tx_daily = tx_hist.groupby(["date", "stock"], as_index=False)["quantity"].sum()
            qty_matrix = (
                tx_daily.pivot(index="date", columns="stock", values="quantity")
                .reindex(hist_prices.index)
                .fillna(0.0)
                .cumsum()
            )
            qty_matrix = qty_matrix.reindex(columns=hist_prices.columns, fill_value=0.0)
            if not tx_hist.empty:
                first_tx_date = tx_hist["date"].min()
                qty_matrix = qty_matrix.loc[qty_matrix.index >= first_tx_date]
            port_value = (qty_matrix * hist_prices).sum(axis=1)
            port_value = port_value.dropna()
            port_value = port_value[port_value > 0]

            tx_detail = tx_hist.merge(
                latest_prices.rename("price_live"),
                left_on="stock",
                right_index=True,
                how="left",
            )
            tx_detail["pnl_current"] = (tx_detail["price_live"] - tx_detail["price"]) * tx_detail["quantity"]
            tx_detail["pnl_current_pct"] = ((tx_detail["price_live"] / tx_detail["price"]) - 1.0) * 100.0

            tx_table = tx_detail.copy()
            tx_table["Fecha"] = tx_table["date"].dt.strftime("%Y-%m-%d")
            tx_table = tx_table.rename(
                columns={
                    "stock": "Ticker",
                    "quantity": "Cantidad",
                    "price": "Precio Compra",
                    "total": "Total",
                    "price_live": "Precio Actual",
                    "pnl_current": "P/L Actual",
                    "pnl_current_pct": "P/L Actual %",
                }
            )
            tx_table = tx_table[
                [
                    "Fecha",
                    "Ticker",
                    "Cantidad",
                    "Precio Compra",
                    "Total",
                    "Precio Actual",
                    "P/L Actual",
                    "P/L Actual %",
                ]
            ]
            tx_table = tx_table.sort_values(["Fecha", "Ticker"])

            tx_table_styler = tx_table.style
            tx_table_styler = tx_table_styler.set_table_styles(center_styles)
            tx_table_format = {
                "Precio Compra": "${:,.2f}",
                "Total": "${:,.2f}",
                "Precio Actual": "${:,.2f}",
                "P/L Actual": "${:,.2f}",
                "P/L Actual %": "{:.2f}%",
                "Cantidad": "{:.4f}",
            }
            tx_table_styler = tx_table_styler.format(tx_table_format)
            tx_table_styler = tx_table_styler.applymap(_pnl_color, subset=["P/L Actual", "P/L Actual %"])
            st.dataframe(tx_table_styler, use_container_width=True)

            net_cf = tx_hist.groupby("date")["total"].sum()
            net_cf = net_cf.reindex(port_value.index).fillna(0.0)
            invested_cum = net_cf.cumsum()

            equity_df = pd.DataFrame(
                {"Fecha": port_value.index, "Valor cartera": port_value.values}
            )
            equity_df["Fecha"] = equity_df["Fecha"].dt.strftime("%Y-%m-%d")
            fig_equity = px.line(
                equity_df,
                x="Fecha",
                y="Valor cartera",
                markers=True,
                labels={"Valor cartera": "Valor cartera", "Fecha": "Fecha"},
            )
            fig_equity.update_traces(
                marker=dict(size=6, color="#ff4d4f"),
                hovertemplate="Fecha: %{x}<br>Valor cartera: %{y:.2f}<extra></extra>",
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            port_value_filled = port_value.ffill()
            market_change = port_value_filled.diff() - net_cf
            market_change = market_change.dropna()
            if not market_change.empty:
                mc_df = pd.DataFrame(
                    {
                        "Fecha": market_change.index,
                        "Cambio por mercado": market_change.values,
                        "Valor cartera": port_value_filled.reindex(market_change.index).values,
                        "Aportaciones netas": net_cf.reindex(market_change.index).values,
                    }
                )
                use_broker_change = (
                    not hist_pnl_df.empty
                    and "date" in hist_pnl_df.columns
                    and (
                        "P/L_Diario" in hist_pnl_df.columns
                        or "Ganancia o perdida diaria" in hist_pnl_df.columns
                    )
                )
                if use_broker_change:
                    change_col = (
                        "P/L_Diario"
                        if "P/L_Diario" in hist_pnl_df.columns
                        else "Ganancia o perdida diaria"
                    )
                    close_col = (
                        "Total_Cierre_Value"
                        if "Total_Cierre_Value" in hist_pnl_df.columns
                        else "Total Cierre"
                    )
                    compra_col = (
                        "Compra_de_acciones"
                        if "Compra_de_acciones" in hist_pnl_df.columns
                        else None
                    )
                    ventas_col = "Ventas" if "Ventas" in hist_pnl_df.columns else None
                    broker_change = hist_pnl_df.copy()
                    broker_change = broker_change.dropna(subset=["date"])
                    broker_change = broker_change.sort_values("date")
                    broker_change = broker_change.dropna(subset=[change_col, close_col])
                    broker_change = broker_change.set_index("date")
                    aport_net = pd.Series(0.0, index=broker_change.index)
                    if compra_col and ventas_col:
                        aport_net = (
                            broker_change[compra_col].fillna(0.0)
                            - broker_change[ventas_col].fillna(0.0)
                        )
                    full_index = pd.Index(mc_df["Fecha"]).union(broker_change.index)
                    mc_df = mc_df.set_index("Fecha").reindex(full_index)
                    mc_df.loc[broker_change.index, "Cambio por mercado"] = broker_change[change_col].astype(float)
                    mc_df.loc[broker_change.index, "Valor cartera"] = broker_change[close_col].astype(float)
                    mc_df.loc[broker_change.index, "Aportaciones netas"] = aport_net.astype(float)
                    mc_df = mc_df.reset_index().rename(columns={"index": "Fecha"})
                mc_df["Fecha"] = pd.to_datetime(mc_df["Fecha"]).dt.strftime("%Y-%m-%d")
                fig_change = px.line(
                    mc_df,
                    x="Fecha",
                    y="Cambio por mercado",
                    markers=True,
                    labels={"Cambio por mercado": "Cambio por mercado", "Fecha": "Fecha"},
                )
                fig_change.update_traces(
                    marker=dict(size=6, color="#ff4d4f"),
                    customdata=mc_df[["Valor cartera", "Aportaciones netas"]].values,
                    hovertemplate=(
                        "Fecha: %{x}"
                        "<br>Cambio por mercado: %{y:.2f}"
                        "<br>Valor cartera: %{customdata[0]:.2f}"
                        "<br>Aportaciones netas: %{customdata[1]:.2f}"
                        "<extra></extra>"
                    ),
                )
                st.plotly_chart(fig_change, use_container_width=True)

            contrib_df = pd.DataFrame(
                {"Fecha": invested_cum.index, "Aportaciones netas": invested_cum.values}
            )
            use_broker_contrib = (
                not hist_pnl_df.empty
                and "date" in hist_pnl_df.columns
                and (
                    "Costo_Value_Invest" in hist_pnl_df.columns
                    or "Costo Value Invest" in hist_pnl_df.columns
                )
            )
            if use_broker_contrib:
                cost_col = (
                    "Costo_Value_Invest"
                    if "Costo_Value_Invest" in hist_pnl_df.columns
                    else "Costo Value Invest"
                )
                broker_contrib = hist_pnl_df[["date", cost_col]].dropna()
                broker_contrib = broker_contrib.rename(
                    columns={"date": "Fecha", cost_col: "Aportaciones netas"}
                )
                full_index = pd.Index(contrib_df["Fecha"]).union(broker_contrib["Fecha"])
                contrib_df = contrib_df.set_index("Fecha").reindex(full_index)
                contrib_df.loc[broker_contrib["Fecha"], "Aportaciones netas"] = broker_contrib[
                    "Aportaciones netas"
                ].values
                contrib_df = contrib_df.reset_index()
            contrib_df["Fecha"] = pd.to_datetime(contrib_df["Fecha"]).dt.strftime("%Y-%m-%d")
            fig_contrib = px.line(
                contrib_df,
                x="Fecha",
                y="Aportaciones netas",
                markers=True,
                labels={"Aportaciones netas": "Aportaciones netas", "Fecha": "Fecha"},
            )
            fig_contrib.update_traces(
                marker=dict(size=6, color="#ff4d4f"),
                hovertemplate="Fecha: %{x}<br>Aportaciones netas: %{y:.2f}<extra></extra>",
            )
            st.plotly_chart(fig_contrib, use_container_width=True)

            pnl_series = port_value - invested_cum
            live_date = pd.to_datetime(latest_asof) if latest_asof is not None else port_value.index.max()
            live_date = live_date.normalize()
            pnl_base_live = invested_cum.iloc[-1]
            live_pnl = total_value - pnl_base_live
            pnl_series = pnl_series.copy()
            if live_date in pnl_series.index:
                pnl_series.loc[live_date] = live_pnl
            else:
                pnl_series = pd.concat([pnl_series, pd.Series({live_date: live_pnl})]).sort_index()
            port_value_live = port_value.copy()
            if live_date in port_value_live.index:
                port_value_live.loc[live_date] = total_value
            else:
                port_value_live = pd.concat([port_value_live, pd.Series({live_date: total_value})]).sort_index()
            invested_live = invested_cum.copy()
            if live_date not in invested_live.index:
                invested_live = pd.concat([invested_live, pd.Series({live_date: invested_cum.iloc[-1]})]).sort_index()

            pnl_df = pd.DataFrame(
                {
                    "Fecha": pnl_series.index,
                    "P/L acumulado": pnl_series.values,
                    "Valor cartera": port_value_live.reindex(pnl_series.index).values,
                    "Aportaciones": invested_live.reindex(pnl_series.index).values,
                }
            )
            use_broker_pnl = (
                not hist_pnl_df.empty
                and "date" in hist_pnl_df.columns
                and (
                    "Total_Cierre_Value" in hist_pnl_df.columns
                    or "Total Cierre" in hist_pnl_df.columns
                )
                and (
                    "Costo_Value_Invest" in hist_pnl_df.columns
                    or "Costo Value Invest" in hist_pnl_df.columns
                )
            )
            if use_broker_pnl:
                close_col = (
                    "Total_Cierre_Value"
                    if "Total_Cierre_Value" in hist_pnl_df.columns
                    else "Total Cierre"
                )
                cost_col = (
                    "Costo_Value_Invest"
                    if "Costo_Value_Invest" in hist_pnl_df.columns
                    else "Costo Value Invest"
                )
                broker_pnl = hist_pnl_df[["date", close_col, cost_col]].dropna()
                broker_pnl = broker_pnl.rename(
                    columns={
                        "date": "Fecha",
                        close_col: "Valor cartera",
                        cost_col: "Aportaciones",
                    }
                )
                broker_pnl["P/L acumulado"] = (
                    broker_pnl["Valor cartera"].astype(float)
                    - broker_pnl["Aportaciones"].astype(float)
                )
                full_index = pd.Index(pnl_df["Fecha"]).union(broker_pnl["Fecha"])
                pnl_df = pnl_df.set_index("Fecha").reindex(full_index)
                pnl_df.loc[broker_pnl["Fecha"], "Valor cartera"] = broker_pnl["Valor cartera"].values
                pnl_df.loc[broker_pnl["Fecha"], "Aportaciones"] = broker_pnl["Aportaciones"].values
                pnl_df.loc[broker_pnl["Fecha"], "P/L acumulado"] = broker_pnl["P/L acumulado"].values
                pnl_df = pnl_df.reset_index()
            pnl_df["Fecha"] = pnl_df["Fecha"].dt.strftime("%Y-%m-%d")
            fig_pnl = px.line(
                pnl_df,
                x="Fecha",
                y="P/L acumulado",
                markers=True,
                labels={"P/L acumulado": "P/L acumulado", "Fecha": "Fecha"},
            )
            fig_pnl.update_traces(
                marker=dict(size=6, color="#ff4d4f"),
                customdata=pnl_df[["Valor cartera", "Aportaciones"]].values,
                hovertemplate=(
                    "Fecha: %{x}"
                    "<br>Valor cartera: %{customdata[0]:.2f}"
                    "<br>Aportaciones: %{customdata[1]:.2f}"
                    "<br>P/L acumulado: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

            port_value_adj = port_value.copy()
            net_cf_adj = net_cf.copy()
            use_broker_twr = (
                not hist_pnl_df.empty
                and "date" in hist_pnl_df.columns
                and (
                    "Total_Cierre_Value" in hist_pnl_df.columns
                    or "Total Cierre" in hist_pnl_df.columns
                )
            )
            if use_broker_twr:
                close_col = (
                    "Total_Cierre_Value"
                    if "Total_Cierre_Value" in hist_pnl_df.columns
                    else "Total Cierre"
                )
                compra_col = (
                    "Compra_de_acciones"
                    if "Compra_de_acciones" in hist_pnl_df.columns
                    else None
                )
                ventas_col = "Ventas" if "Ventas" in hist_pnl_df.columns else None
                broker_twr = hist_pnl_df[["date", close_col]].dropna()
                broker_twr = broker_twr.set_index("date")
                broker_net_cf = pd.Series(0.0, index=broker_twr.index)
                if compra_col and ventas_col:
                    broker_net_cf = (
                        hist_pnl_df.set_index("date")[compra_col].fillna(0.0)
                        - hist_pnl_df.set_index("date")[ventas_col].fillna(0.0)
                    ).reindex(broker_twr.index)
                full_index = port_value_adj.index.union(broker_twr.index)
                port_value_adj = port_value_adj.reindex(full_index)
                net_cf_adj = net_cf_adj.reindex(full_index).fillna(0.0)
                port_value_adj.loc[broker_twr.index] = broker_twr[close_col].astype(float)
                net_cf_adj.loc[broker_twr.index] = broker_net_cf.astype(float)

            twr_base = port_value_adj.shift(1)
            twr_daily = ((port_value_adj - net_cf_adj) / twr_base) - 1.0
            twr_daily = twr_daily.replace([np.inf, -np.inf], np.nan)
            twr_daily = twr_daily.dropna()
            twr_cum = (1.0 + twr_daily).cumprod() - 1.0
            if not twr_cum.empty:
                twr_df = pd.DataFrame(
                    {"Fecha": twr_cum.index, "TWR acumulado %": twr_cum.values * 100.0}
                )
                twr_df["Fecha"] = twr_df["Fecha"].dt.strftime("%Y-%m-%d")
                fig_twr = px.line(
                    twr_df,
                    x="Fecha",
                    y="TWR acumulado %",
                    markers=True,
                    labels={"TWR acumulado %": "TWR acumulado (%)", "Fecha": "Fecha"},
                )
                fig_twr.update_traces(
                    marker=dict(size=6, color="#ff4d4f"),
                    hovertemplate="Fecha: %{x}<br>TWR acumulado: %{y:.2f}%<extra></extra>",
                )
                st.plotly_chart(fig_twr, use_container_width=True)

else:
    st.dataframe(filtered_holdings, use_container_width=True)

import plotly.express as px

st.subheader("📉 P/L cierres diarios (mercado)")
hist_close_full = pd.Series(dtype=float)
qty_matrix_hist = pd.DataFrame()
hist_prices_hist = pd.DataFrame()
if not tx_df.empty and not price_history.empty:
    tx_hist = tx_df.copy()
    tx_hist["stock"] = tx_hist["stock"].astype(str).str.upper().str.strip()
    tx_hist["date"] = pd.to_datetime(tx_hist["date"])
    tx_hist["quantity"] = pd.to_numeric(tx_hist["quantity"], errors="coerce")
    tx_hist = tx_hist.dropna(subset=["date", "stock", "quantity"])
    hist_prices = price_history[tickers].ffill().dropna(how="all")
    tx_daily = tx_hist.groupby(["date", "stock"], as_index=False)["quantity"].sum()
    qty_matrix = (
        tx_daily.pivot(index="date", columns="stock", values="quantity")
        .reindex(hist_prices.index)
        .fillna(0.0)
        .cumsum()
    )
    qty_matrix = qty_matrix.reindex(columns=hist_prices.columns, fill_value=0.0)
    if not tx_hist.empty:
        first_tx_date = tx_hist["date"].min()
        qty_matrix = qty_matrix.loc[qty_matrix.index >= first_tx_date]
        hist_prices = hist_prices.loc[hist_prices.index >= first_tx_date]
    hist_close_full = (qty_matrix * hist_prices).sum(axis=1).dropna()
    qty_matrix_hist = qty_matrix
    hist_prices_hist = hist_prices

use_broker_history = not hist_pnl_df.empty and "date" in hist_pnl_df.columns
if use_broker_history:
    close_full = daily_value_full.copy()
else:
    close_full = hist_close_full if not hist_close_full.empty else daily_value_market_full
close_full = close_full.loc[close_full.index.weekday < 5]
close_full = close_full.sort_index()
history_start_date = close_full.index.min() if not close_full.empty else pd.Timestamp(cutoff_date)
close_series_all = close_full.loc[close_full.index >= history_start_date]

if len(close_series_all) >= 1:
    pnl_close = close_series_all.diff().dropna()
    pnl_close = pnl_close.loc[pnl_close.index >= pd.Timestamp(cutoff_date)]
    range_label = st.radio(
        "Rango",
        options=["1D", "1M", "1Y", "YTD", "ALL"],
        horizontal=True,
        index=1,
        label_visibility="collapsed",
    )
    last_date = close_series_all.index.max()
    if range_label == "1D":
        start_date = last_date - pd.DateOffset(days=1)
    elif range_label == "1M":
        start_date = last_date - pd.DateOffset(months=1)
    elif range_label == "1Y":
        start_date = last_date - pd.DateOffset(years=1)
    elif range_label == "YTD":
        start_date = pd.Timestamp(last_date.year, 1, 1)
    else:
        start_date = daily_value_market.index.min()

    close_series = close_series_all.loc[close_series_all.index >= start_date]
    open_series = close_series.shift(1)
    if use_broker_history and manual_daily_pnl is not None and not manual_daily_pnl.empty:
        pl_series = manual_daily_pnl.reindex(close_series.index)
    elif not qty_matrix_hist.empty and not hist_prices_hist.empty:
        prices_aligned = hist_prices_hist.reindex(close_series.index).ffill()
        qty_prev = qty_matrix_hist.reindex(close_series.index).shift(1).ffill().fillna(0.0)
        pl_series = (prices_aligned.diff() * qty_prev).sum(axis=1)
    else:
        pl_series = close_series - open_series
    open_series = open_series.fillna(close_series)
    pl_series = pl_series.fillna(0.0)

    daily_df = pd.DataFrame(
        {
            "date": close_series.index,
            "open": open_series.values,
            "close": close_series.values,
            "pl": pl_series.values,
        }
    ).dropna()

    if daily_df.empty:
        st.info("No hay datos de mercado para el rango seleccionado.")
    else:
        daily_df["date_str"] = daily_df["date"].dt.strftime("%d %b %Y")
        base_close = float(daily_df["close"].iloc[0])
        fig_close = go.Figure()
        fig_close.add_trace(
            go.Scatter(
                x=daily_df["date_str"],
                y=daily_df["close"],
                mode="lines+markers",
                name="Cierre diario",
                marker=dict(size=7, color="#ff4d4f"),
                customdata=daily_df[["open", "close", "pl", "date_str"]].values,
                hovertemplate=(
                    "Fecha: %{customdata[3]}<br>"
                    "Open (prev cierre): %{customdata[0]:.2f}<br>"
                    "Close: %{customdata[1]:.2f}<br>"
                    "P/L: %{customdata[2]:.2f}<extra></extra>"
                ),
            )
        )
        fig_close.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Cierre diario",
        )
        if range_label == "1D":
            if len(pl_series) >= 1:
                range_pl = float(pl_series.iloc[-1])
                base_val = float(open_series.iloc[-1])
                range_pl_pct = (range_pl / base_val) if base_val else 0.0
            else:
                range_pl = 0.0
                range_pl_pct = 0.0
        elif range_label == "YTD":
            year_start = pd.Timestamp(last_date.year, 1, 1)
            ytd_pl = pl_series.loc[pl_series.index >= year_start]
            if len(ytd_pl) >= 1:
                range_pl = float(ytd_pl.sum())
                base_val = float(close_full.loc[close_full.index >= year_start].iloc[0])
                range_pl_pct = (range_pl / base_val) if base_val else 0.0
            else:
                range_pl = 0.0
                range_pl_pct = 0.0
        else:
            range_pl = float(pl_series.sum()) if len(pl_series) >= 1 else 0.0
            base_val = float(close_series.iloc[0]) if len(close_series) >= 1 else float("nan")
            range_pl_pct = (range_pl / base_val) if base_val else 0.0
            if range_label == "1M":
                year_start = pd.Timestamp(last_date.year, 1, 1)
                first_trading = close_full.loc[close_full.index >= year_start]
                if not first_trading.empty and close_series.index.min() == first_trading.index.min():
                    prev_close_series = close_full.loc[close_full.index < year_start]
                    if not prev_close_series.empty:
                        prev_close = float(prev_close_series.iloc[-1])
                        range_pl_pct = (range_pl / prev_close) if prev_close else 0.0
            if range_label == "ALL":
                prev_close_series = close_full.loc[close_full.index < pd.Timestamp(cutoff_date)]
                if not prev_close_series.empty:
                    prev_close = float(prev_close_series.iloc[-1])
                    range_pl_pct = (range_pl / prev_close) if prev_close else 0.0
        metric_spacer, metric_col = st.columns([4, 1])
        with metric_col:
            st.metric(f"P/L {range_label}", f"${range_pl:,.2f}", delta=f"{range_pl_pct:.2%}")
        st.markdown("<div style='margin-top:-16px;'></div>", unsafe_allow_html=True)
        last_row = daily_df.iloc[-1]
        fig_close.add_scatter(
            x=[last_row["date_str"]],
            y=[last_row["close"]],
            mode="markers+text",
            text=[f"P/L cierre: {last_row['pl']:.2f}"],
            textposition="bottom right",
            textfont=dict(color="#ff4d4f"),
            marker=dict(size=8, color="#ff4d4f"),
            showlegend=False,
        )
        if not daily_df.empty:
            min_close = float(daily_df["close"].min())
            max_close = float(daily_df["close"].max())
            pad = (max_close - min_close) * 0.1 if max_close != min_close else 1.0
            fig_close.update_yaxes(range=[min_close - pad, max_close + pad])
    fig_close.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_close, use_container_width=True)
else:
    st.info("No hay suficientes cierres para graficar P/L diario.")

st.subheader("🧭 Seguimiento avanzado")

st.caption(f"Benchmark vs Portafolio ({SETTINGS.benchmark})")
perf_range = st.radio(
    "Rango",
    options=["1D", "1M", "1Y", "YTD", "ALL"],
    horizontal=True,
    index=3,
    label_visibility="collapsed",
)
perf_year = snapshot_view_date.year
if not portfolio_ret.empty and not benchmark_ret.empty:
    use_broker_history = "date" in hist_pnl_df.columns and not hist_pnl_df.empty
    port_series = daily_value_full.copy() if use_broker_history else daily_value_market_full.copy()
    bench_series = price_history[SETTINGS.benchmark].copy()
    bench_series.index = pd.to_datetime(bench_series.index)
    if bench_series.index.tz is not None:
        bench_series.index = bench_series.index.tz_localize(None)
    bench_saved = data_io.load_benchmark_closes_csv(SETTINGS.benchmark_closes_path)
    if not bench_saved.empty:
        bench_saved_series = bench_saved.set_index("date")["close"].astype(float)
        bench_series = pd.concat([bench_series, bench_saved_series]).sort_index()
        bench_series = bench_series[~bench_series.index.duplicated(keep="last")]
    aligned = pd.concat([port_series, bench_series], axis=1).dropna()
    aligned.columns = ["portfolio_value", "benchmark_close"]
    aligned = aligned.loc[aligned.index >= history_start_date]
    aligned = aligned[aligned.index.year == int(perf_year)]
    if not aligned.empty:
        if perf_range == "1D":
            start_date = aligned.index.max() - pd.DateOffset(days=1)
        elif perf_range == "1M":
            start_date = aligned.index.max() - pd.DateOffset(months=1)
        elif perf_range == "1Y":
            start_date = aligned.index.max() - pd.DateOffset(years=1)
        elif perf_range == "YTD":
            start_date = pd.Timestamp(int(perf_year), 1, 2)
        else:
            start_date = aligned.index.min()
        aligned = aligned.loc[aligned.index >= start_date]

    if use_broker_history and perf_range == "YTD":
        broker_rows = hist_pnl_df[hist_pnl_df["date"].dt.year == int(perf_year)].copy()
        if not broker_rows.empty and close_key in broker_rows.columns:
            if "P/L_Diario" in broker_rows.columns:
                pl_col = "P/L_Diario"
            elif "Ganancia o perdida diaria" in broker_rows.columns:
                pl_col = "Ganancia o perdida diaria"
            else:
                pl_col = None
            base_close = float(broker_rows.iloc[0][close_key])
            broker_rows = broker_rows.loc[broker_rows["date"] >= start_date]
            if pl_col:
                pl_series_broker = broker_rows[pl_col].fillna(0.0).astype(float)
                cum_port = pl_series_broker.cumsum() / base_close if base_close else pl_series_broker * 0.0
                pl_port = pl_series_broker.cumsum()
                perf_dates = broker_rows["date"]
                bench_series = bench_series.reindex(perf_dates).ffill()
                base_bench = float(bench_series.iloc[0]) if not bench_series.empty else float("nan")
                cum_bench = (bench_series / base_bench) - 1.0 if base_bench else bench_series * 0.0
                perf_df = pd.DataFrame(
                    {
                        "date": perf_dates.values,
                        "Portafolio": cum_port.values,
                        SETTINGS.benchmark: cum_bench.values,
                        "pl_port": pl_port.values,
                        "pl_bench": (bench_series.values - base_bench) if base_bench else np.zeros(len(perf_dates)),
                    }
                )
                base_port = base_close
            else:
                perf_df = pd.DataFrame(columns=["date", "Portafolio", SETTINGS.benchmark, "pl_port", "pl_bench"])
        else:
            perf_df = pd.DataFrame(columns=["date", "Portafolio", SETTINGS.benchmark, "pl_port", "pl_bench"])
    elif len(aligned) >= 1:
        base_port = float(aligned["portfolio_value"].iloc[0])
        base_bench = float(aligned["benchmark_close"].iloc[0])
        cum_port = (aligned["portfolio_value"] / base_port) - 1.0
        cum_bench = (aligned["benchmark_close"] / base_bench) - 1.0
        perf_df = pd.DataFrame(
            {
                "date": aligned.index,
                "Portafolio": cum_port.values,
                SETTINGS.benchmark: cum_bench.values,
                "pl_port": aligned["portfolio_value"].values - base_port,
                "pl_bench": aligned["benchmark_close"].values - base_bench,
            }
        )
    else:
        perf_df = pd.DataFrame(columns=["date", "Portafolio", SETTINGS.benchmark, "pl_port", "pl_bench"])
    perf_df["date_str"] = perf_df["date"].dt.strftime("%d %b %Y")
    if len(aligned) >= 1:
        ytd_port = float(cum_port.iloc[-1])
        ytd_bench = float(cum_bench.iloc[-1])
        delta_perf = ytd_port - ytd_bench
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric(f"{perf_range} Portafolio", f"{ytd_port:.2%}")
        col_b2.metric(f"{perf_range} {SETTINGS.benchmark}", f"{ytd_bench:.2%}")
        col_b3.metric(f"Delta vs {SETTINGS.benchmark}", f"{delta_perf:.2%}")
        st.caption(
            f"Base acumulado: Portafolio ${base_port:,.2f} · {SETTINGS.benchmark} ${base_bench:,.2f}"
        )
    else:
        st.info("No hay suficientes cierres para comparar con benchmark.")
    fig_perf = px.line(
        perf_df,
        x="date_str",
        y=["Portafolio", SETTINGS.benchmark],
        markers=True,
        labels={"value": "Retorno acumulado", "date_str": "Fecha", "variable": "Serie"},
    )
    fig_perf.update_yaxes(tickformat=".1%")
    if not perf_df.empty:
        min_val = min(perf_df["Portafolio"].min(), perf_df[SETTINGS.benchmark].min())
        max_val = max(perf_df["Portafolio"].max(), perf_df[SETTINGS.benchmark].max())
        pad = (max_val - min_val) * 0.1 if max_val != min_val else 0.005
        fig_perf.update_yaxes(range=[min_val - pad, max_val + pad])
    fig_perf.update_traces(
        marker=dict(size=6, color="#ff4d4f"),
    )
    for trace in fig_perf.data:
        if trace.name == "Portafolio":
            trace.customdata = perf_df["pl_port"].values
            trace.hovertemplate = (
                "Fecha: %{x}"
                "<br>P/L: %{customdata:.2f}"
                "<br>Retorno: %{y:.2%}"
                "<extra></extra>"
            )
        else:
            trace.customdata = perf_df["pl_bench"].values
            trace.hovertemplate = (
                "Fecha: %{x}"
                f"<br>P/L {SETTINGS.benchmark}: "
                "%{customdata:.2f}"
                "<br>Retorno: %{y:.2%}"
                "<extra></extra>"
            )
    fig_perf.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_perf, use_container_width=True)
else:
    st.info("No hay suficientes datos para comparar con el benchmark.")



st.caption("Riesgo y alertas")
if not portfolio_ret.empty:
    port_ret_filtered = portfolio_ret.copy()
    if year_filter != "Todos":
        port_ret_filtered = port_ret_filtered[port_ret_filtered.index.year == int(year_filter)]
    if port_ret_filtered.empty:
        st.info("No hay datos para métricas de riesgo en el rango seleccionado.")
    else:
        equity_curve = (1.0 + port_ret_filtered).cumprod()
        peak = equity_curve.cummax()
        drawdown = (equity_curve / peak) - 1.0
        max_dd = float(drawdown.min())
        worst_day = float(port_ret_filtered.min())
        ann_vol = float(port_ret_filtered.std(ddof=1) * np.sqrt(252))
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Drawdown máx.", f"{max_dd:.2%}")
        col_r2.metric("Peor día", f"{worst_day:.2%}")
        col_r3.metric("Volatilidad anual", f"{ann_vol:.2%}")
        if max_dd < -0.2:
            st.warning("Drawdown elevado: supera -20%.")
        if worst_day < -0.05:
            st.warning("Pérdida diaria alta: peor día < -5%.")
else:
    st.info("No hay datos para métricas de riesgo.")

snap_summary_df = data_io.load_daily_summary_sqlite(SETTINGS.snapshots_db)
snap_holdings_df = data_io.load_daily_holdings_sqlite(SETTINGS.snapshots_db)
if not snap_summary_df.empty:
    snap_dates = sorted(snap_summary_df["date"].dt.date.unique().tolist())
    snap_default = snapshot_view_date.date()
    if snap_default not in snap_dates and snap_dates:
        snap_default = snap_dates[-1]
    snap_pick = st.selectbox(
        "Fecha del snapshot",
        options=snap_dates,
        index=snap_dates.index(snap_default) if snap_default in snap_dates else 0,
        key="snapshot_download_date",
    )
    snap_row = snap_summary_df[snap_summary_df["date"] == pd.Timestamp(snap_pick)]
    if not snap_row.empty:
        snap_row = snap_row.iloc[0]
        with st.expander(f"📅 Snapshot del día: {pd.Timestamp(snap_pick).date().isoformat()}", expanded=False):
            snap_summary_out = pd.DataFrame([snap_row]).copy()
            snap_summary_out["date"] = snap_summary_out["date"].dt.strftime("%Y-%m-%d")
            snap_holdings_raw = snap_holdings_df[snap_holdings_df["date"] == pd.Timestamp(snap_pick)].copy()
            snap_holdings = snap_holdings_raw.rename(
                columns={
                    "ticker": "Ticker",
                    "quantity": "Cantidad",
                    "avg_price": "Avg Price",
                    "close_price": "Close Price",
                    "cost_value": "Cost Value",
                    "market_value": "Market Value",
                    "pnl": "P/L",
                    "pnl_pct": "P/L %",
                }
            )
            snap_holdings = snap_holdings[
                [
                    "Ticker",
                    "Cantidad",
                    "Avg Price",
                    "Close Price",
                    "Cost Value",
                    "Market Value",
                    "P/L",
                    "P/L %",
                ]
            ]

            tickets_snap = snap_holdings.copy()
            tickets_snap = tickets_snap.rename(
                columns={
                    "Close Price": "Precio Actual",
                    "Market Value": "Valor Portafolio",
                    "P/L": "Ganancia",
                    "P/L %": "Ganancia Pct",
                }
            )
            tickets_snap = tickets_snap[
                ["Ticker", "Precio Actual", "Valor Portafolio", "Ganancia", "Ganancia Pct"]
            ]

            broker_rows = hist_pnl_df.copy()
            broker_rows = broker_rows[broker_rows["date"] <= pd.Timestamp(snap_pick)]
            if "P/L_Diario" in broker_rows.columns:
                pl_col = "P/L_Diario"
            elif "Ganancia o perdida diaria" in broker_rows.columns:
                pl_col = "Ganancia o perdida diaria"
            else:
                pl_col = None
            close_col = (
                "Total_Cierre_Value"
                if "Total_Cierre_Value" in broker_rows.columns
                else "Total Cierre"
            )
            close_hist = broker_rows.set_index("date")[close_col].astype(float).sort_index()
            pl_hist = broker_rows.set_index("date")[pl_col].astype(float).sort_index() if pl_col else close_hist.diff().fillna(0.0)
            pl_table_snap = pd.DataFrame(
                {
                    "Fecha": close_hist.index.strftime("%Y-%m-%d"),
                    "Open (prev cierre)": close_hist.shift(1).fillna(close_hist).values,
                    "Close": close_hist.values,
                    "P/L": pl_hist.values,
                }
            )

            perf_df_snap = None
            if not perf_df.empty:
                perf_df_snap = perf_df[perf_df["date"] <= pd.Timestamp(snap_pick)].copy()
                perf_df_snap = perf_df_snap.rename(
                    columns={
                        "Portafolio": "Portafolio Return",
                        SETTINGS.benchmark: f"{SETTINGS.benchmark} Return",
                    }
                )
                perf_df_snap["date"] = perf_df_snap["date"].dt.strftime("%Y-%m-%d")

            excel_buffer = io.BytesIO()
            try:
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    snap_summary_out.to_excel(writer, index=False, sheet_name="Summary")
                    snap_holdings.to_excel(writer, index=False, sheet_name="Holdings")
                    tickets_snap.to_excel(writer, index=False, sheet_name="Tickets")
                    pl_table_snap.to_excel(writer, index=False, sheet_name="PL_Cierres")
                    if perf_df_snap is not None:
                        perf_df_snap.to_excel(writer, index=False, sheet_name="Seguimiento")
                excel_buffer.seek(0)
                st.download_button(
                    f"Descargar snapshot {pd.Timestamp(snap_pick).date().isoformat()}",
                    data=excel_buffer,
                    file_name=f"snapshot_{pd.Timestamp(snap_pick).date().isoformat()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                st.info("No se pudo generar el archivo Excel. Falta instalar openpyxl.")
    else:
        st.info("Aún no hay snapshot guardado para esta fecha.")

st.caption("Aportes y retiros")
flows_file = st.file_uploader("Sube CSV de aportes/retiros (date, amount)", type=["csv"])
if flows_file is not None:
    flows_df = pd.read_csv(flows_file)
    if {"date", "amount"} <= set(flows_df.columns):
        flows_df["date"] = pd.to_datetime(flows_df["date"])
        total_flows = flows_df["amount"].sum()
        st.metric("Flujo neto", f"${total_flows:,.2f}")
        st.dataframe(flows_df.sort_values("date"), use_container_width=True)
    else:
        st.warning("El CSV debe tener columnas: date, amount.")
else:
    st.info("Añade aportes/retiros para separar rendimiento real vs. dinero nuevo.")
