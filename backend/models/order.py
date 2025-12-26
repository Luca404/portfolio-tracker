from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, text
from sqlalchemy.orm import relationship
from .base import Base


class OrderModel(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False)
    isin = Column(String, default="")
    ter = Column(String, default="")
    name = Column(String, default="")
    exchange = Column(String, default="")
    currency = Column(String, default="")
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    instrument_type = Column(String, default="stock")
    order_type = Column(String, nullable=False)  # buy / sell
    date = Column(Date, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    portfolio = relationship("PortfolioModel", back_populates="orders")
