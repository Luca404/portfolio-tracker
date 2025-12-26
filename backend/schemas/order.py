from datetime import date
from typing import List, Optional, Union
from pydantic import BaseModel, field_validator

from utils.dates import parse_date_input


class Order(BaseModel):
    portfolio_id: int
    symbol: str
    name: Optional[str] = ""
    exchange: Optional[str] = ""
    currency: Optional[str] = ""
    isin: Optional[str] = ""
    ter: Optional[Union[str, float]] = ""
    quantity: float
    price: float
    commission: float = 0.0
    instrument_type: str = "stock"
    order_type: str
    date: date

    @field_validator("date", mode="before")
    @classmethod
    def parse_date_field(cls, v):
        parsed = parse_date_input(v)
        if parsed:
            return parsed
        return v


class OptimizationRequest(BaseModel):
    portfolio_id: int
    symbols: List[str]
    optimization_type: str = "max_sharpe"
