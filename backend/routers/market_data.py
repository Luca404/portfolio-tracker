import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import UserModel, PortfolioModel
from utils import (
    get_db,
    verify_token,
    get_risk_free_rate,
    get_market_benchmark_data,
    fetch_us_treasury_rate,
    fetch_ecb_rate,
    fetch_market_benchmark,
)

router = APIRouter(prefix="/market-data", tags=["market-data"])


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
    """
    Get risk-free rate for a currency.
    - USD: US Treasury 10Y
    - EUR: ECB Main Refinancing Operations rate

    Returns:
        {
            "currency": "USD",
            "current_rate": 4.5,
            "history": [{"date": "DD-MM-YYYY", "rate": 4.5}, ...]
        }
    """
    currency = currency.upper()
    if currency not in ["USD", "EUR"]:
        raise HTTPException(status_code=400, detail=f"Unsupported currency: {currency}. Supported: USD, EUR")

    try:
        if currency == "USD":
            data = fetch_us_treasury_rate(db)
        else:  # EUR
            data = fetch_ecb_rate(db)

        return {
            "currency": currency,
            "current_rate": data.get("current_rate", 0.0),
            "history": data.get("history", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching risk-free rate: {str(e)}")


@router.get("/benchmark/{currency}")
def get_benchmark_endpoint(currency: str, db: Session = Depends(get_db)):
    """
    Get market benchmark data for a currency.
    - USD: S&P 500 (^GSPC)
    - EUR: VWCE (VWCE.DE - Vanguard FTSE All-World)

    Returns:
        {
            "currency": "USD",
            "symbol": "^GSPC",
            "last_price": 4500.00,
            "history": [{"date": "DD-MM-YYYY", "price": 4500}, ...]
        }
    """
    currency = currency.upper()
    if currency not in ["USD", "EUR"]:
        raise HTTPException(status_code=400, detail=f"Unsupported currency: {currency}. Supported: USD, EUR")

    try:
        data = fetch_market_benchmark(currency, db)
        return {
            "currency": currency,
            "symbol": data.get("symbol", ""),
            "last_price": data.get("last_price", 0.0),
            "history": data.get("history", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching benchmark data: {str(e)}")


@router.get("/portfolio-context/{portfolio_id}")
def get_portfolio_market_context(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    """
    Get appropriate risk-free rate and benchmark for a portfolio
    based on its reference_currency.

    Returns:
        {
            "portfolio_id": 1,
            "currency": "EUR",
            "risk_free_rate": {
                "current_rate": 4.5,
                "source": "ECB Main Refinancing Operations"
            },
            "benchmark": {
                "symbol": "VWCE.DE",
                "name": "Vanguard FTSE All-World",
                "last_price": 100.50
            }
        }
    """
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    currency = (portfolio.reference_currency or "EUR").upper()

    try:
        # Fetch risk-free rate
        rf_rate = get_risk_free_rate(currency, db)

        # Fetch benchmark data
        benchmark_data = get_market_benchmark_data(currency, db)

        # Descriptive names
        rf_source_map = {
            "USD": "US Treasury 10Y",
            "EUR": "ECB Main Refinancing Operations"
        }

        benchmark_name_map = {
            "USD": "S&P 500",
            "EUR": "Vanguard FTSE All-World"
        }

        return {
            "portfolio_id": portfolio_id,
            "currency": currency,
            "risk_free_rate": {
                "current_rate": rf_rate,
                "source": rf_source_map.get(currency, "Unknown")
            },
            "benchmark": {
                "symbol": benchmark_data.get("symbol", ""),
                "name": benchmark_name_map.get(currency, "Unknown"),
                "last_price": benchmark_data.get("last_price", 0.0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market context: {str(e)}")
