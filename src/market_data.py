from __future__ import annotations
import pandas as pd
import yfinance as yf

def get_latest_prices(tickers: list[str]) -> pd.Series:
    data = yf.download(tickers, period="5d", interval="1d", auto_adjust=False, progress=False)
    close = data["Close"]
    latest = close.dropna(how="all").iloc[-1]
    latest.name = "price"
    return latest

def get_price_history(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False)
    close = data["Close"].dropna(how="all")
    close.index = pd.to_datetime(close.index)
    return close