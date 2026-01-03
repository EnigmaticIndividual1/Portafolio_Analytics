from __future__ import annotations
import pandas as pd
from pathlib import Path

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