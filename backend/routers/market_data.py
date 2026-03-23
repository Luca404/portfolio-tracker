import json
import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models import ETFPriceCacheModel, StockPriceCacheModel, ExchangeRateCacheModel
from utils import (
    get_db,
    get_supabase,
    verify_token,
    get_risk_free_rate,
    get_market_benchmark_data,
    fetch_us_treasury_rate,
    fetch_ecb_rate,
    fetch_market_benchmark,
)

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/cache-status")
def get_cache_status(user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    """Mostra lo stato della cache prezzi: ultimo dato di mercato e quando è stata scaricata."""

    def last_date(history_json: str) -> str | None:
        try:
            history = json.loads(history_json or "[]")
            return history[-1].get("date") if history else None
        except Exception:
            return None

    etfs = db.query(ETFPriceCacheModel).all()
    stocks = db.query(StockPriceCacheModel).all()
    fx = db.query(ExchangeRateCacheModel).all()

    return {
        "etf": [
            {
                "isin": e.isin,
                "last_price": e.last_price,
                "last_market_date": last_date(e.history_json),
                "cache_updated_at": e.updated_at.isoformat() if e.updated_at else None,
            }
            for e in etfs
        ],
        "stock": [
            {
                "symbol": s.symbol,
                "last_price": s.last_price,
                "last_market_date": last_date(s.history_json),
                "cache_updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in stocks
        ],
        "fx": [
            {
                "pair": f.pair,
                "last_market_date": last_date(f.history_json),
                "cache_updated_at": f.updated_at.isoformat() if f.updated_at else None,
            }
            for f in fx
        ],
    }


@router.get("/{symbol}")
def get_market_data(symbol: str):
    """Get basic market data for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk-free-rate/{currency}")
def get_risk_free_rate_endpoint(currency: str, db: Session = Depends(get_db)):
    currency = currency.upper()
    if currency not in ["USD", "EUR"]:
        raise HTTPException(status_code=400, detail=f"Unsupported currency: {currency}")

    try:
        data = fetch_us_treasury_rate(db) if currency == "USD" else fetch_ecb_rate(db)
        return {"currency": currency, "current_rate": data.get("current_rate", 0.0), "history": data.get("history", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/{currency}")
def get_benchmark_endpoint(currency: str, db: Session = Depends(get_db)):
    currency = currency.upper()
    if currency not in ["USD", "EUR"]:
        raise HTTPException(status_code=400, detail=f"Unsupported currency: {currency}")

    try:
        data = fetch_market_benchmark(currency, db)
        return {"currency": currency, "symbol": data.get("symbol", ""), "last_price": data.get("last_price", 0.0), "history": data.get("history", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio-context/{portfolio_id}")
def get_portfolio_market_context(portfolio_id: int, user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    sb = get_supabase()
    portfolio = sb.table("portfolios").select("reference_currency").eq("id", portfolio_id).eq("user_id", user_id).execute()
    if not portfolio.data:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    currency = (portfolio.data[0].get("reference_currency") or "EUR").upper()

    try:
        rf_rate = get_risk_free_rate(currency, db)
        benchmark_data = get_market_benchmark_data(currency, db)

        rf_source_map = {"USD": "US Treasury 10Y", "EUR": "ECB Main Refinancing Operations"}
        benchmark_name_map = {"USD": "S&P 500", "EUR": "Vanguard FTSE All-World"}

        return {
            "portfolio_id": portfolio_id,
            "currency": currency,
            "risk_free_rate": {"current_rate": rf_rate, "source": rf_source_map.get(currency, "Unknown")},
            "benchmark": {
                "symbol": benchmark_data.get("symbol", ""),
                "name": benchmark_name_map.get(currency, "Unknown"),
                "last_price": benchmark_data.get("last_price", 0.0),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
