from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_weights_bar(holdings: pd.DataFrame, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df = holdings.sort_values("weight_pct", ascending=True)
    plt.figure()
    plt.barh(df["ticker"], df["weight_pct"])
    plt.title("Peso por activo (%)")
    plt.xlabel("Peso (%)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def save_pnl_bar(holdings: pd.DataFrame, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df = holdings.sort_values("pnl", ascending=True)
    plt.figure()
    plt.barh(df["ticker"], df["pnl"])
    plt.title("P/L por activo (USD)")
    plt.xlabel("P/L (USD)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def save_equity_curve(portfolio_ret: pd.Series, benchmark_ret: pd.Series | None, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    equity = (1.0 + portfolio_ret).cumprod()
    plt.figure()
    plt.plot(equity.index, equity.values, label="Portfolio")

    if benchmark_ret is not None:
        beq = (1.0 + benchmark_ret).cumprod()
        plt.plot(beq.index, beq.values, label="Benchmark")

    plt.title("Equity curve (normalizada)")
    plt.xlabel("Fecha")
    plt.ylabel("Crecimiento (base 1.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def save_correlation_heatmap(corr: pd.DataFrame, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(corr.values, interpolation="nearest")
    plt.title("Matriz de correlación")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()