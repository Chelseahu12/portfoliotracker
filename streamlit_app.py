import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List

_DATA_DIR = Path(".portfolio_data")
_DATA_DIR.mkdir(exist_ok=True)
_HOLDINGS_FILE = _DATA_DIR / "holdings.parquet"

def _load_holdings() -> pd.DataFrame:
    if _HOLDINGS_FILE.exists():
        return pd.read_parquet(_HOLDINGS_FILE)
    return pd.DataFrame(columns=["ticker", "shares", "buy_price", "added_on"])

def _save_holdings(df: pd.DataFrame) -> None:
    df.to_parquet(_HOLDINGS_FILE, index=False)

@dataclass
class Portfolio:
    holdings: pd.DataFrame = field(default_factory=_load_holdings)

    def add(self, ticker: str, shares: float) -> None:
        ticker = ticker.upper()
        self._validate_ticker(ticker)
        current_price = yf.Ticker(ticker).history(period="1d")["Close"][0]

        if ticker in self.holdings["ticker"].values:
            existing = self.holdings[self.holdings["ticker"] == ticker]
            total_shares = existing["shares"].values[0] + shares
            new_buy_price = (
                (existing["buy_price"].values[0] * existing["shares"].values[0] + current_price * shares)
                / total_shares
            )
            self.holdings.loc[self.holdings["ticker"] == ticker, ["shares", "buy_price"]] = [total_shares, new_buy_price]
        else:
            self.holdings = pd.concat([
                self.holdings,
                pd.DataFrame({
                    "ticker": [ticker],
                    "shares": [shares],
                    "buy_price": [current_price],
                    "added_on": [date.today()],
                })
            ], ignore_index=True)
        _save_holdings(self.holdings)

    def remove(self, ticker: str) -> None:
        ticker = ticker.upper()
        self.holdings = self.holdings[self.holdings["ticker"] != ticker]
        _save_holdings(self.holdings)

    def _price_history(self, start: Optional[str | datetime | date] = None, end: Optional[str | datetime | date] = None) -> pd.DataFrame:
        if self.holdings.empty:
            raise ValueError("Portfolio is empty – add holdings first.")
        tickers = " ".join(self.holdings["ticker"].unique())
        df = yf.download(tickers=tickers, start=start, end=end, progress=False)["Adj Close"].ffill().dropna(how="all")
        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers.strip())
        return df

    def market_value(self) -> float:
        if self.holdings.empty:
            return 0.0
        prices = self._price_history().iloc[-1]
        shares = self.holdings.set_index("ticker")["shares"].reindex(prices.index)
        return float((prices * shares).sum())

    def daily_rundown(self) -> str:
        today = date.today()
        yesterday = today - timedelta(days=5)
        prices = self._price_history(start=yesterday, end=today + timedelta(days=1))

        if len(prices) < 2:
            return "Not enough data to compute daily change yet."

        latest = prices.iloc[-1]
        prev = prices.iloc[-2]
        pct_change = ((latest - prev) / prev).mul(100).round(2)
        shares = self.holdings.set_index("ticker")["shares"].reindex(latest.index)
        pos_change = ((latest - prev) * shares).sum()

        value = self.market_value()
        total_cost = (self.holdings["shares"] * self.holdings["buy_price"]).sum()
        total_pnl = value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost != 0 else 0

        summary_lines = [
            f"📈 Portfolio value: ${value:,.2f}",
            f"💰 Total P/L: ${total_pnl:,.2f} ({total_pnl_pct:.2f}%)",
            f"🔄 Day-over-day P/L: ${pos_change:,.2f} ({(pos_change / (value - pos_change))*100:,.2f}%)",
            "",
            "Ticker  Δ% (1d)",
            "──────  ────────",
        ]
        summary_lines.extend([f"{t:6}  {pct:+6.2f}%" for t, pct in pct_change.items()])
        return "\n".join(summary_lines)

    def plot_portfolio_pnl(self) -> None:
        import matplotlib.pyplot as plt

        if self.holdings.empty:
            print("Portfolio is empty.")
            return

        start = self.holdings["added_on"].min()
        prices = self._price_history(start=start).ffill().dropna(how="all")
        shares = self.holdings.set_index("ticker")["shares"].reindex(prices.columns)
        buy_prices = self.holdings.set_index("ticker")["buy_price"].reindex(prices.columns)

        daily_value = prices.mul(shares, axis=1).sum(axis=1)
        cost_basis = (shares * buy_prices).sum()
        pnl = daily_value - cost_basis

        plt.figure(figsize=(10, 5))
        plt.plot(pnl.index, pnl.values, label="Cumulative P/L", lw=2)
        plt.axhline(0, color="gray", linestyle="--", lw=1)
        plt.title("Portfolio P/L Over Time")
        plt.xlabel("Date")
        plt.ylabel("Profit / Loss ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

    @staticmethod
    def _validate_ticker(ticker: str) -> None:
        if not ticker.isalpha():
            raise ValueError("Ticker symbols should contain only letters (e.g., 'AAPL').")
