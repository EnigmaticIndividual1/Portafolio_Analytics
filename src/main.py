from __future__ import annotations
import pandas as pd

from src.config import SETTINGS
from src.data_io import load_positions, save_table
from src.market_data import get_latest_prices, get_price_history
from src.analytics import (
    compute_holdings_table, daily_returns, portfolio_returns,
    annualized_return, annualized_volatility, sharpe_ratio, max_drawdown,
    beta_vs_benchmark, var_cvar, correlation_matrix, risk_contributions
)
from src.charts import (
    save_weights_bar, save_pnl_bar, save_equity_curve, save_correlation_heatmap
)

def run() -> None:
    SETTINGS.reports_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.charts_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.tables_dir.mkdir(parents=True, exist_ok=True)

    positions = load_positions(SETTINGS.positions_path)
    tickers = sorted(positions["ticker"].unique().tolist())
    all_tickers = tickers + [SETTINGS.benchmark]

    latest_prices = get_latest_prices(all_tickers)
    holdings = compute_holdings_table(positions, latest_prices)

    weights = holdings.set_index("ticker")["market_value"]
    weights = weights / weights.sum()

    price_history = get_price_history(all_tickers, SETTINGS.history_period, SETTINGS.history_interval)
    rets = daily_returns(price_history)

    benchmark_ret = rets[SETTINGS.benchmark].copy()
    asset_rets = rets.drop(columns=[SETTINGS.benchmark], errors="ignore")

    portfolio_ret = portfolio_returns(asset_rets, weights)

    port_ann_ret = annualized_return(portfolio_ret)
    port_ann_vol = annualized_volatility(portfolio_ret)
    port_sharpe = sharpe_ratio(portfolio_ret, SETTINGS.risk_free_rate_annual)
    port_mdd = max_drawdown(portfolio_ret)
    port_beta = beta_vs_benchmark(portfolio_ret, benchmark_ret)

    var_95, cvar_95 = var_cvar(portfolio_ret, alpha=0.05)

    corr = correlation_matrix(asset_rets)
    risk_contrib = risk_contributions(asset_rets, weights)

    summary = pd.DataFrame([{
        "total_market_value": float(holdings["market_value"].sum()),
        "total_cost_value": float(holdings["cost_value"].sum()),
        "total_pnl": float(holdings["pnl"].sum()),
        "total_pnl_pct": float((holdings["pnl"].sum() / holdings["cost_value"].sum()) * 100.0),
        "annual_return_est": port_ann_ret,
        "annual_volatility_est": port_ann_vol,
        "sharpe_est": port_sharpe,
        "max_drawdown": port_mdd,
        "beta_vs_benchmark": port_beta,
        "VaR_95_daily": var_95,
        "CVaR_95_daily": cvar_95,
        "benchmark": SETTINGS.benchmark,
        "history_period": SETTINGS.history_period,
    }])

    save_table(holdings, SETTINGS.tables_dir / "holdings.xlsx")
    save_table(summary, SETTINGS.tables_dir / "summary.xlsx")
    save_table(risk_contrib, SETTINGS.tables_dir / "risk_contributions.xlsx")
    save_table(corr.reset_index().rename(columns={"index": "ticker"}), SETTINGS.tables_dir / "correlations.xlsx")

    save_weights_bar(holdings, SETTINGS.charts_dir / "weights.png")
    save_pnl_bar(holdings, SETTINGS.charts_dir / "pnl.png")
    save_equity_curve(portfolio_ret, benchmark_ret, SETTINGS.charts_dir / "equity_curve.png")
    save_correlation_heatmap(corr, SETTINGS.charts_dir / "correlation.png")

    print("OK: Reportes generados en /reports")

if __name__ == "__main__":
    run()