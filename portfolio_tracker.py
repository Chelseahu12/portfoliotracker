import pandas as pd

class Portfolio:
    def __init__(self):
        self.holdings = pd.DataFrame(columns=["ticker", "shares", "buy_price", "added_on"])

    def add(self, ticker, shares):
        buy_price = 100  # Placeholder; you can use live price logic
        self.holdings = pd.concat([
            self.holdings,
            pd.DataFrame([{
                "ticker": ticker.upper(),
                "shares": shares,
                "buy_price": buy_price,
                "added_on": pd.Timestamp.now().date()
            }])
        ], ignore_index=True)

    def remove(self, ticker):
        self.holdings = self.holdings[self.holdings["ticker"] != ticker.upper()]

    def daily_rundown(self):
        return self.holdings.to_dict(orient="records")

    def plot_portfolio_pnl(self):
        import matplotlib.pyplot as plt
        plt.plot([1, 2, 3], [10, 15, 12])  # Dummy plot
