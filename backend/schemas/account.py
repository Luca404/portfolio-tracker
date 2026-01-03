from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AccountBase(BaseModel):
    name: str
    icon: str = "💳"
    initial_balance: float = 0.0


class AccountCreate(AccountBase):
    pass


class AccountUpdate(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None
    initial_balance: Optional[float] = None


class Account(AccountBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
