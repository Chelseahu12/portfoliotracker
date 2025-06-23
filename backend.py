from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from streamlit_app import Portfolio
from datetime import datetime
from pathlib import Path
import json
import io
import pandas as pd
import yfinance as yf

app = FastAPI()

# ────────────────────── CORS (allow local front‑ends) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Global objects & paths ─────────────────────────
portfolio = Portfolio()
_DATA_DIR = Path(".portfolio_data")
_DATA_DIR.mkdir(exist_ok=True)
SELL_LOG_FILE = _DATA_DIR / "sell_log.json"

# ─────────────────────────────── Pydantic DTOs ──────────────────────────────
class AddRequest(BaseModel):
    ticker: str
    shares: float

class SellRequest(BaseModel):
    ticker: str
    shares: float
    sell_date: str  # YYYY-MM-DD

# ───────────────────────────── Portfolio routes ─────────────────────────────
@app.post("/add")
def add_stock(req: AddRequest):
    try:
        portfolio.add(req.ticker, req.shares)
        return {"message": f"Added {req.shares} shares of {req.ticker.upper()}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/rundown")
def get_rundown():
    try:
        summary = portfolio.daily_rundown()

        # Validate DataFrame before proceeding
        for ticker in portfolio.holdings["ticker"].unique():
            df = yf.download(ticker, period="1d")
            if df.empty or "Adj Close" not in df.columns:
                raise ValueError(f"No price data found for {ticker}")

        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/pnl")
def get_portfolio_pnl():
    try:
        import matplotlib.pyplot as plt
        import base64

        fig = plt.figure()
        portfolio.plot_portfolio_pnl()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        return {"image_base64": img_str}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ───────────────────────────── Sell + history ───────────────────────────────
@app.post("/sell")
def sell_stock(req: SellRequest):
    try:
        ticker = req.ticker.upper()
        record = portfolio.holdings[portfolio.holdings["ticker"] == ticker]

        if record.empty:
            raise ValueError(f"No holdings found for ticker {ticker}")

        record = record.iloc[0]
        held_shares = float(record["shares"])

        if req.shares > held_shares:
            raise ValueError(f"Cannot sell {req.shares} shares; only {held_shares} available")

        buy_price = float(record["buy_price"])
        buy_date = str(record["added_on"])

        total_cost = buy_price * req.shares
        sell_price = yf.download(ticker, period="1d")["Adj Close"].iloc[0]
        total_proceeds = sell_price * req.shares
        pnl = total_proceeds - total_cost
        roi_multiple = sell_price / buy_price if buy_price else None

        # Adjust holdings
        if req.shares == held_shares:
            portfolio.remove(ticker)
        else:
            portfolio.holdings.loc[portfolio.holdings["ticker"] == ticker, "shares"] -= req.shares

        log_entry = {
            "ticker": ticker,
            "shares": req.shares,
            "buy_date": buy_date,
            "sell_date": req.sell_date,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "pnl": pnl,
            "roi_multiple": roi_multiple,
            "logged_at": datetime.now().isoformat(timespec="seconds")
        }

        history = []
        if SELL_LOG_FILE.exists():
            history = json.loads(SELL_LOG_FILE.read_text())
        history.append(log_entry)
        SELL_LOG_FILE.write_text(json.dumps(history, indent=2))

        return log_entry
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/history")
def get_sale_history():
    try:
        if SELL_LOG_FILE.exists():
            return json.loads(SELL_LOG_FILE.read_text())
        return []
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ─────────────────────────── Export to Excel (.xlsx) ─────────────────────────
@app.get("/export_history")
def export_history():
    """Download sales history as an Excel spreadsheet."""
    try:
        if not SELL_LOG_FILE.exists():
            raise HTTPException(status_code=404, detail="No sales history yet.")
        data = json.loads(SELL_LOG_FILE.read_text())
        if not data:
            raise HTTPException(status_code=404, detail="Sales history is empty.")

        df = pd.DataFrame(data)
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        headers = {
            "Content-Disposition": "attachment; filename=sales_history.xlsx"
        }
        return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
