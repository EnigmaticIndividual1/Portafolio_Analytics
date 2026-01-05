import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import src.config as config
import src.data_io as data_io
from src.market_data import get_latest_prices, get_price_history
from src.analytics import (
    compute_holdings_table,
    daily_returns,
    portfolio_returns,
    annualized_return,
    sharpe_ratio,
)

st.sidebar.header("⚙️ Configuración")
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
cutoff_date = st.sidebar.date_input(
    "Inicio histórico (manual)",
    value=pd.Timestamp("2026-01-02").date(),
)
start_year = cutoff_date.year
current_year = pd.Timestamp.today().year
year_options = ["Todos"] + [str(y) for y in range(start_year, current_year + 2)]
year_filter = st.sidebar.selectbox(
    "Año de análisis",
    options=year_options,
    index=1 if str(start_year) in year_options else 0,
)

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

st.title("📊 Portfolio Analytics — Live")

SETTINGS = config.SETTINGS

# --- Cargar datos base ---
positions = data_io.load_positions(SETTINGS.positions_path)
tickers = positions["ticker"].unique().tolist()
all_tickers = tickers + [SETTINGS.benchmark]

# --- Precios actuales ---
latest_prices = get_latest_prices(all_tickers)
holdings = compute_holdings_table(positions, latest_prices)

# --- Histórico ---
price_history = get_price_history(
    all_tickers,
    period,
    SETTINGS.history_interval
)
last_market_update = None
if not price_history.empty:
    last_market_update = pd.to_datetime(price_history.index.max())

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
ann_ret = annualized_return(portfolio_ret)
sharpe = sharpe_ratio(portfolio_ret, SETTINGS.risk_free_rate_annual)

# --- Layout ---
col1, col2, col3, col4 = st.columns(4)

col1.metric("Valor total", f"${total_value:,.2f}")
col2.metric("P/L total", f"${total_pnl:,.2f}")
col3.metric("Retorno anual (est.)", f"{ann_ret*100:.2f}%")
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
col4.markdown(
    f"<div style='font-size:0.9rem; color:#9e9e9e;'>Sharpe</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{sharpe_color};'>{sharpe:.2f}</div>",
    unsafe_allow_html=True,
)
if last_market_update is not None:
    st.caption(f"Última actualización de mercado: {last_market_update.strftime('%Y-%m-%d %H:%M:%S')}")

hist_pnl_df = pd.DataFrame()
if hasattr(SETTINGS, "historical_pnl_path") and SETTINGS.historical_pnl_path.exists():
    hist_raw = pd.read_csv(SETTINGS.historical_pnl_path, sep="\t")
    hist_raw.columns = [c.strip() for c in hist_raw.columns]
    if "Fecha" in hist_raw.columns and "Total Cierre" in hist_raw.columns:
        hist_raw["date"] = pd.to_datetime(hist_raw["Fecha"], dayfirst=True, errors="coerce")
        hist_raw["Total Cierre"] = (
            hist_raw["Total Cierre"]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
        if "Ganancia o perdida diaria" in hist_raw.columns:
            hist_raw["Ganancia o perdida diaria"] = (
                hist_raw["Ganancia o perdida diaria"]
                .astype(str)
                .str.replace(r"[^0-9.\-]", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )
        hist_pnl_df = hist_raw.dropna(subset=["date"]).sort_values("date")
        hist_pnl_df = hist_pnl_df[hist_pnl_df["date"] >= pd.Timestamp(cutoff_date)]

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

if not hist_pnl_df.empty:
    daily_value_full = hist_pnl_df.set_index("date")["Total Cierre"].copy()
    daily_value_full = daily_value_full.loc[daily_value_full.index.weekday < 5]
    daily_value = daily_value_full.loc[daily_value_full.index >= pd.Timestamp(cutoff_date)]
    if "Ganancia o perdida diaria" in hist_pnl_df.columns:
        manual_daily_pnl = hist_pnl_df.set_index("date")["Ganancia o perdida diaria"].copy()
        manual_daily_pnl = manual_daily_pnl.loc[manual_daily_pnl.index.weekday < 5]
        manual_daily_pnl = manual_daily_pnl.loc[manual_daily_pnl.index >= pd.Timestamp(cutoff_date)]
else:
    daily_value_full = daily_value_market_full
    daily_value = daily_value_market

filtered_daily_value = daily_value
if year_filter != "Todos":
    filtered_daily_value = daily_value[daily_value.index.year == int(year_filter)]

filtered_daily_value_market = daily_value_market
if year_filter != "Todos":
    filtered_daily_value_market = daily_value_market[daily_value_market.index.year == int(year_filter)]

if len(filtered_daily_value_market) >= 1:
    base_value = float(filtered_daily_value_market.iloc[-1])
    base_date = filtered_daily_value_market.index[-1]
    live_pnl = total_value - base_value
    live_pnl_pct = (live_pnl / base_value) * 100.0 if base_value else 0.0
    if len(filtered_daily_value_market) >= 2:
        prev_value = float(filtered_daily_value_market.iloc[-2])
        prev_date = filtered_daily_value_market.index[-2]
        last_close_pnl = base_value - prev_value
    else:
        prev_value = float("nan")
        prev_date = None
        last_close_pnl = 0.0
else:
    live_pnl = 0.0
    live_pnl_pct = 0.0
    last_close_pnl = 0.0
    base_date = None
    prev_date = None

col5, col6, col7 = st.columns(3)
live_color = "#52c41a" if live_pnl > 0 else ("#ff4d4f" if live_pnl < 0 else "#9e9e9e")
live_pct_color = "#52c41a" if live_pnl_pct > 0 else ("#ff4d4f" if live_pnl_pct < 0 else "#9e9e9e")
last_close_color = "#52c41a" if last_close_pnl > 0 else ("#ff4d4f" if last_close_pnl < 0 else "#9e9e9e")
col5.markdown(
    f"<div style='font-size:0.9rem; color:#9e9e9e;'>P/L en vivo</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{live_color};'>${live_pnl:,.2f}</div>",
    unsafe_allow_html=True,
)
col6.markdown(
    f"<div style='font-size:0.9rem; color:#9e9e9e;'>P/L en vivo (%)</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{live_pct_color};'>{live_pnl_pct:.2f}%</div>",
    unsafe_allow_html=True,
)
col7.markdown(
    f"<div style='font-size:0.9rem; color:#9e9e9e;'>P/L último cierre</div>"
    f"<div style='font-size:2rem; font-weight:600; color:{last_close_color};'>${last_close_pnl:,.2f}</div>",
    unsafe_allow_html=True,
)
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

st.divider()

st.subheader("✍️ Registrar cierre manual")
with st.form("manual_close_form"):
    input_date = st.date_input("Fecha de cierre", value=pd.Timestamp.today().date())
    close_value = st.number_input("Cierre total (USD)", min_value=0.0, step=100.0, format="%.2f")
    auto_pnl = st.checkbox("Calcular P/L automáticamente", value=True)
    pnl_value = st.number_input("P/L del día (USD)", step=10.0, format="%.2f")
    submitted = st.form_submit_button("Guardar cierre")

if submitted:
    hist_path = SETTINGS.historical_pnl_path
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path, sep="\t")
    else:
        hist_df = pd.DataFrame(columns=["Fecha", "Total Cierre", "Ganancia o perdida diaria"])

    def parse_money(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    if auto_pnl:
        prev_close = None
        if "Total Cierre" in hist_df.columns and not hist_df.empty:
            prev_values = parse_money(hist_df["Total Cierre"]).dropna()
            if not prev_values.empty:
                prev_close = float(prev_values.iloc[-1])
        pnl_value = close_value - prev_close if prev_close is not None else 0.0

    date_str = pd.Timestamp(input_date).strftime("%d/%m/%y")
    row = {
        "Fecha": date_str,
        "Total Cierre": f"${close_value:,.2f}",
        "Ganancia o perdida diaria": f"${pnl_value:,.2f}",
    }

    if "Fecha" not in hist_df.columns:
        hist_df["Fecha"] = np.nan
    if "Total Cierre" not in hist_df.columns:
        hist_df["Total Cierre"] = np.nan
    if "Ganancia o perdida diaria" not in hist_df.columns:
        hist_df["Ganancia o perdida diaria"] = np.nan

    if (hist_df["Fecha"] == date_str).any():
        hist_df.loc[hist_df["Fecha"] == date_str, ["Total Cierre", "Ganancia o perdida diaria"]] = [
            row["Total Cierre"],
            row["Ganancia o perdida diaria"],
        ]
    else:
        hist_df = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)

    hist_df.to_csv(hist_path, sep="\t", index=False)
    st.success(f"Cierre guardado para {date_str}.")

st.subheader("📋 Holdings")
st.dataframe(filtered_holdings, use_container_width=True)

import plotly.express as px

st.subheader("📉 P/L cierres diarios (mercado)")
if len(daily_value_market) >= 1:
    pnl_close = daily_value_market.diff().dropna()
    pnl_close = pnl_close.loc[pnl_close.index >= pd.Timestamp(cutoff_date)]
    range_col, avg_col = st.columns([3, 1])
    with range_col:
        range_label = st.radio(
            "Rango",
            options=["1D", "1M", "1Y", "ALL"],
            horizontal=True,
            index=1,
            label_visibility="collapsed",
        )
    last_date = daily_value_market.index.max()
    if range_label == "1D":
        start_date = last_date - pd.DateOffset(days=1)
    elif range_label == "1M":
        start_date = last_date - pd.DateOffset(months=1)
    elif range_label == "1Y":
        start_date = last_date - pd.DateOffset(years=1)
    else:
        start_date = daily_value_market.index.min()

    close_series = daily_value_market.loc[daily_value_market.index >= start_date]
    if year_filter != "Todos":
        year_value = int(year_filter)
        close_series = close_series.loc[close_series.index.year == year_value]
    open_series = daily_value_market_full.shift(1).reindex(close_series.index)
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
        fig_close = px.line(
            daily_df,
            x="date",
            y="close",
            labels={"close": "Cierre diario", "date": "Fecha"},
        )
        avg_pl = float(daily_df["pl"].mean()) if not daily_df.empty else 0.0
        with avg_col:
            st.metric("Promedio P/L", f"${avg_pl:,.2f}")
    fig_close.update_traces(
        hovertemplate=(
            "Fecha: %{x|%Y-%m-%d}<br>"
            "Open (prev cierre): %{customdata[0]:.2f}<br>"
            "Close: %{customdata[1]:.2f}<br>"
            "P/L: %{customdata[2]:.2f}<extra></extra>"
        ),
        customdata=daily_df[["open", "close", "pl"]].values,
    )
    last_row = daily_df.iloc[-1]
    fig_close.add_scatter(
        x=[last_row["date"]],
        y=[last_row["close"]],
        mode="markers+text",
        text=[f"P/L cierre: {last_row['pl']:.2f}"],
        textposition="top right",
        marker=dict(size=8, color="#ff4d4f"),
        name="Ultimo cierre",
        showlegend=False,
    )
    st.plotly_chart(fig_close, use_container_width=True)
else:
    st.info("No hay suficientes cierres para graficar P/L diario.")

st.subheader("🧭 Seguimiento avanzado")

st.caption("Benchmark vs Portafolio")
if not portfolio_ret.empty and not benchmark_ret.empty:
    aligned = pd.concat([portfolio_ret, benchmark_ret], axis=1).dropna()
    aligned = aligned.loc[aligned.index >= pd.Timestamp(cutoff_date)]
    aligned.columns = ["portfolio", "benchmark"]
    cum_port = (1.0 + aligned["portfolio"]).cumprod() - 1.0
    cum_bench = (1.0 + aligned["benchmark"]).cumprod() - 1.0
    perf_df = pd.DataFrame(
        {
            "date": aligned.index,
            "Portafolio": cum_port.values,
            "Benchmark": cum_bench.values,
        }
    )
    if not aligned.empty:
        last_perf = aligned.iloc[-1]
        delta_perf = last_perf["portfolio"] - last_perf["benchmark"]
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Retorno último cierre (Portafolio)", f"{last_perf['portfolio']:.2%}")
        col_b2.metric("Retorno último cierre (Benchmark)", f"{last_perf['benchmark']:.2%}")
        col_b3.metric("Delta vs Benchmark", f"{delta_perf:.2%}")
    fig_perf = px.line(
        perf_df,
        x="date",
        y=["Portafolio", "Benchmark"],
        labels={"value": "Retorno acumulado", "date": "Fecha", "variable": "Serie"},
    )
    fig_perf.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_perf, use_container_width=True)
else:
    st.info("No hay suficientes datos para comparar con el benchmark.")

st.caption("Concentración y diversificación")
if not filtered_holdings.empty:
    if year_filter != "Todos":
        st.caption("Basado en holdings actuales (sin histórico por año).")
    weights_pct = filtered_holdings.set_index("ticker")["weight_pct"]
    top_weights = weights_pct.sort_values(ascending=False).head(5).reset_index()
    top_weights.columns = ["ticker", "weight_pct"]
    hhi = float((weights_pct / 100.0).pow(2).sum())
    col_c1, col_c2 = st.columns(2)
    col_c1.dataframe(top_weights, use_container_width=True)
    col_c2.metric("HHI (concentración)", f"{hhi:.3f}")
    if weights_pct.max() > 30:
        st.warning("Concentración alta: un activo supera 30% del portafolio.")
else:
    st.info("No hay holdings para calcular concentración.")

st.caption("Distribución por clase")
if "asset_class" in filtered_holdings.columns:
    if year_filter != "Todos":
        st.caption("Basado en holdings actuales (sin histórico por año).")
    class_df = (
        filtered_holdings.groupby("asset_class", as_index=False)["market_value"]
        .sum()
        .sort_values("market_value", ascending=False)
    )
    fig_class = px.pie(
        class_df,
        names="asset_class",
        values="market_value",
    )
    st.plotly_chart(fig_class, use_container_width=True)
else:
    st.info("No hay columna asset_class para agrupar por clase.")

st.caption("Riesgo y alertas")
if not portfolio_ret.empty:
    port_ret_filtered = portfolio_ret.loc[portfolio_ret.index >= pd.Timestamp(cutoff_date)]
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
