from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from .base import Base


class TransactionType(str, enum.Enum):
    EXPENSE = "expense"
    INCOME = "income"
    INVESTMENT = "investment"
    TRANSFER = "transfer"


class TransactionModel(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(SQLEnum(TransactionType), nullable=False)
    category = Column(String, nullable=False)
    subcategory = Column(String, nullable=True)
    amount = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    date = Column(DateTime, nullable=False)

    # Campi specifici per investimenti
    ticker = Column(String, nullable=True)
    quantity = Column(Float, nullable=True)
    price = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("UserModel", back_populates="transactions")
