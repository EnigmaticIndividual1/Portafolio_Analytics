from __future__ import annotations
import pandas as pd
import yfinance as yf

def get_latest_prices(tickers: list[str]) -> pd.Series:
    data = yf.download(tickers, period="5d", interval="1d", auto_adjust=False, progress=False)
    close = data["Close"]
    latest = close.dropna(how="all").iloc[-1]
    latest.attrs["asof"] = close.dropna(how="all").index[-1]
    latest.name = "price"
    return latest

def get_price_history(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False)
    close = data["Close"].dropna(how="all")
    close.index = pd.to_datetime(close.index)
    return close


def get_ticker_events(tickers: list[str]) -> pd.DataFrame:
    events: list[dict[str, object]] = []
    for ticker in tickers:
        try:
            cal = yf.Ticker(ticker).calendar
        except Exception:
            continue
        if not isinstance(cal, pd.DataFrame) or cal.empty:
            continue
        if cal.shape[1] == 1:
            cal_series = cal.iloc[:, 0]
        else:
            cal_series = cal.iloc[0]
        for label in ["Earnings Date", "Ex-Dividend Date", "Dividend Date"]:
            if label not in cal_series:
                continue
            value = cal_series[label]
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            if pd.isna(value):
                continue
            events.append(
                {
                    "ticker": ticker,
                    "event": label,
                    "date": pd.to_datetime(value).date(),
                }
            )
    if not events:
        return pd.DataFrame(columns=["ticker", "event", "date"])
    return pd.DataFrame(events)
