from fastapi import FastAPI, HTTPException
import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import Optional

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/stock/{symbol}")
async def get_stock_data(symbol: str, period: Optional[str] = "1mo"):
    """
    Get historical stock data for a given symbol
    period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return {
            "symbol": symbol,
            "data": hist.to_dict('records'),
            "info": ticker.info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

