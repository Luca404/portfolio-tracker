from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel
from models.transaction import TransactionType


class TransactionBase(BaseModel):
    type: TransactionType
    category: str
    subcategory: Optional[str] = None
    amount: float
    description: Optional[str] = None
    date: datetime
    ticker: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None


class TransactionCreate(TransactionBase):
    pass


class TransactionUpdate(BaseModel):
    type: Optional[TransactionType] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    amount: Optional[float] = None
    description: Optional[str] = None
    date: Optional[datetime] = None
    ticker: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None


class TransactionResponse(TransactionBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TransactionStats(BaseModel):
    total_expenses: float
    total_income: float
    total_investments: float
    balance: float
    expenses_by_category: dict[str, float]
    monthly_trend: list[dict[str, Any]]
