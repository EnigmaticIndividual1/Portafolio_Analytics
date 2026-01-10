from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"ticker", "quantity", "avg_price"}

def load_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"positions.csv no tiene columnas requeridas: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise")
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="raise")
    return df

def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

def save_daily_snapshot_csv(path: Path, snapshot_date: pd.Timestamp, total_value: float, weights: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "date": snapshot_date.date().isoformat(),
        "total_value": float(total_value),
        "weights_json": json.dumps(weights, sort_keys=True),
    }
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            if (df["date"] == record["date"]).any():
                df.loc[df["date"] == record["date"], ["total_value", "weights_json"]] = [
                    record["total_value"],
                    record["weights_json"],
                ]
            else:
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])
    else:
        df = pd.DataFrame([record])
    df = df.sort_values("date")
    df.to_csv(path, index=False)

def load_daily_snapshots_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "total_value", "weights_json"])
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def save_daily_snapshot_sqlite(path: Path, snapshot_date: pd.Timestamp, total_value: float, weights: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                date TEXT PRIMARY KEY,
                total_value REAL NOT NULL,
                weights_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_snapshots (date, total_value, weights_json)
            VALUES (?, ?, ?)
            """,
            (
                snapshot_date.date().isoformat(),
                float(total_value),
                json.dumps(weights, sort_keys=True),
            ),
        )
        conn.commit()
    finally:
        conn.close()

def load_daily_snapshots_sqlite(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "total_value", "weights_json"])
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT date, total_value, weights_json FROM daily_snapshots ORDER BY date",
            conn,
        )
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

def save_daily_holdings_csv(path: Path, snapshot_date: pd.Timestamp, holdings_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = holdings_df.copy()
    snapshot["date"] = snapshot_date.date().isoformat()
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df = df[df["date"] != snapshot["date"].iloc[0]]
        df = pd.concat([df, snapshot], ignore_index=True)
    else:
        df = snapshot
    sort_cols = [c for c in ["date", "ticker"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    df.to_csv(path, index=False)

def save_daily_holdings_sqlite(path: Path, snapshot_date: pd.Timestamp, holdings_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_holdings (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                close_price REAL NOT NULL,
                cost_value REAL NOT NULL,
                market_value REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                PRIMARY KEY (date, ticker)
            )
            """
        )
        snapshot = holdings_df.copy()
        snapshot["date"] = snapshot_date.date().isoformat()
        for _, row in snapshot.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_holdings
                (date, ticker, quantity, avg_price, close_price, cost_value, market_value, pnl, pnl_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["date"],
                    row["ticker"],
                    float(row["quantity"]),
                    float(row["avg_price"]),
                    float(row["close_price"]),
                    float(row["cost_value"]),
                    float(row["market_value"]),
                    float(row["pnl"]),
                    float(row["pnl_pct"]),
                ),
            )
        conn.commit()
    finally:
        conn.close()

def save_daily_summary_sqlite(path: Path, snapshot_date: pd.Timestamp, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_value REAL NOT NULL,
                total_cost REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_pnl_pct REAL NOT NULL,
                liquidity REAL NOT NULL,
                live_pnl REAL NOT NULL,
                live_pnl_pct REAL NOT NULL,
                last_close_pnl REAL NOT NULL,
                xirr REAL NOT NULL,
                sharpe REAL NOT NULL,
                ann_return REAL NOT NULL,
                benchmark TEXT NOT NULL,
                benchmark_close REAL
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_summary
            (date, total_value, total_cost, total_pnl, total_pnl_pct, liquidity, live_pnl, live_pnl_pct,
             last_close_pnl, xirr, sharpe, ann_return, benchmark, benchmark_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_date.date().isoformat(),
                float(summary.get("total_value", 0.0)),
                float(summary.get("total_cost", 0.0)),
                float(summary.get("total_pnl", 0.0)),
                float(summary.get("total_pnl_pct", 0.0)),
                float(summary.get("liquidity", 0.0)),
                float(summary.get("live_pnl", 0.0)),
                float(summary.get("live_pnl_pct", 0.0)),
                float(summary.get("last_close_pnl", 0.0)),
                float(summary.get("xirr", 0.0)),
                float(summary.get("sharpe", 0.0)),
                float(summary.get("ann_return", 0.0)),
                str(summary.get("benchmark", "")),
                None if summary.get("benchmark_close") is None else float(summary.get("benchmark_close")),
            ),
        )
        conn.commit()
    finally:
        conn.close()

def load_daily_holdings_sqlite(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "quantity",
                "avg_price",
                "close_price",
                "cost_value",
                "market_value",
                "pnl",
                "pnl_pct",
            ]
        )
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT date, ticker, quantity, avg_price, close_price, cost_value, market_value, pnl, pnl_pct "
            "FROM daily_holdings ORDER BY date, ticker",
            conn,
        )
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_daily_summary_sqlite(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "total_value",
                "total_cost",
                "total_pnl",
                "total_pnl_pct",
                "liquidity",
                "live_pnl",
                "live_pnl_pct",
                "last_close_pnl",
                "xirr",
                "sharpe",
                "ann_return",
                "benchmark",
                "benchmark_close",
            ]
        )
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT date, total_value, total_cost, total_pnl, total_pnl_pct, liquidity, live_pnl, "
            "live_pnl_pct, last_close_pnl, xirr, sharpe, ann_return, benchmark, benchmark_close "
            "FROM daily_summary ORDER BY date",
            conn,
        )
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

def save_benchmark_close_csv(path: Path, snapshot_date: pd.Timestamp, close_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "date": snapshot_date.date().isoformat(),
        "close": float(close_value),
    }
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            if (df["date"] == record["date"]).any():
                df.loc[df["date"] == record["date"], ["close"]] = [record["close"]]
            else:
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])
    else:
        df = pd.DataFrame([record])
    df = df.sort_values("date")
    df.to_csv(path, index=False)

def load_benchmark_closes_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "close"])
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")
