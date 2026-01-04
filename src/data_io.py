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
