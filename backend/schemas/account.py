from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AccountBase(BaseModel):
    name: str
    icon: str = "💳"
    initial_balance: float = 0.0
    is_favorite: bool = False


class AccountCreate(AccountBase):
    pass


class AccountUpdate(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None
    initial_balance: Optional[float] = None
    current_balance: Optional[float] = None
    is_favorite: Optional[bool] = None


class Account(AccountBase):
    id: int
    user_id: int
    current_balance: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
