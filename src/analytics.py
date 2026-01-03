from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_holdings_table(positions: pd.DataFrame, latest_prices: pd.Series) -> pd.DataFrame:
    df = positions.copy()
    df["market_price"] = df["ticker"].map(latest_prices)
    if df["market_price"].isna().any():
        missing = df.loc[df["market_price"].isna(), "ticker"].tolist()
        raise ValueError(f"No pude obtener precio para: {missing}")

    df["market_value"] = df["quantity"] * df["market_price"]
    df["cost_value"] = df["quantity"] * df["avg_price"]
    df["pnl"] = df["market_value"] - df["cost_value"]
    df["pnl_pct"] = (df["pnl"] / df["cost_value"]) * 100.0

    total_value = df["market_value"].sum()
    df["weight_pct"] = (df["market_value"] / total_value) * 100.0
    df["weight"] = df["market_value"] / total_value

    cols = [
        "ticker", "quantity", "avg_price", "market_price",
        "cost_value", "market_value", "pnl", "pnl_pct", "weight_pct"
    ]
    keep = [c for c in cols if c in df.columns]
    if "asset_class" in df.columns:
        keep.insert(1, "asset_class")

    df = df[keep].sort_values("weight_pct", ascending=False)
    return df

def daily_returns(price_history: pd.DataFrame) -> pd.DataFrame:
    return price_history.pct_change().dropna(how="all")

def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weights = weights.reindex(returns.columns).fillna(0.0)
    pr = returns.mul(weights, axis=1).sum(axis=1)
    pr.name = "portfolio_return"
    return pr

def annualized_volatility(series: pd.Series) -> float:
    return float(series.std(ddof=1) * np.sqrt(TRADING_DAYS))

def annualized_return(series: pd.Series) -> float:
    return float((1.0 + series).prod() ** (TRADING_DAYS / len(series)) - 1.0)

def sharpe_ratio(series: pd.Series, risk_free_rate_annual: float) -> float:
    rf_daily = (1.0 + risk_free_rate_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = series - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((excess.mean() / vol) * np.sqrt(TRADING_DAYS))

def max_drawdown(series: pd.Series) -> float:
    equity = (1.0 + series).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def beta_vs_benchmark(portfolio_ret: pd.Series, benchmark_ret: pd.Series) -> float:
    aligned = pd.concat([portfolio_ret, benchmark_ret], axis=1).dropna()
    if len(aligned) < 30:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var = np.var(aligned.iloc[:, 1])
    if var == 0:
        return float("nan")
    return float(cov / var)

def var_cvar(series: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    r = series.dropna().values
    if len(r) == 0:
        return float("nan"), float("nan")
    var = np.quantile(r, alpha)
    cvar = r[r <= var].mean() if np.any(r <= var) else float("nan")
    return float(var), float(cvar)

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()

def risk_contributions(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    aligned = returns.dropna()
    w = weights.reindex(aligned.columns).fillna(0.0).values.reshape(-1, 1)

    cov = aligned.cov().values * TRADING_DAYS
    port_var = float((w.T @ cov @ w).squeeze())
    port_vol = np.sqrt(port_var) if port_var > 0 else np.nan

    mrc = (cov @ w).squeeze() / port_vol if port_vol and not np.isnan(port_vol) else np.full(len(w), np.nan)
    rc = (w.squeeze() * mrc)

    out = pd.DataFrame({
        "ticker": aligned.columns,
        "weight": w.squeeze(),
        "risk_contribution": rc,
    })
    total = out["risk_contribution"].sum()
    out["risk_contribution_pct"] = out["risk_contribution"] / total if total != 0 else np.nan
    out = out.sort_values("risk_contribution_pct", ascending=False)
    return out