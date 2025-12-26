from typing import Optional
from pydantic import BaseModel


class Portfolio(BaseModel):
    name: str
    description: Optional[str] = ""
    initial_capital: float = 10000.0
    reference_currency: Optional[str] = "EUR"
    risk_free_source: Optional[str] = "auto"  # "auto", "USD_TREASURY", "EUR_ECB", or custom value
    market_benchmark: Optional[str] = "auto"  # "auto", "SP500", "VWCE", or custom ticker
