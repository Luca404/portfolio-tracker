from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, text
from sqlalchemy.orm import relationship
from .base import Base


class PortfolioModel(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    initial_capital = Column(Float, default=0.0)  # kept for schema compatibility, not used in logic
    reference_currency = Column(String, default="EUR")  # Valuta di riferimento per conversioni
    risk_free_source = Column(String, default="auto")  # "auto", "USD_TREASURY", "EUR_ECB", oppure valore custom
    market_benchmark = Column(String, default="auto")  # "auto", "SP500", "VWCE", oppure ticker custom
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    user = relationship("UserModel", back_populates="portfolios")
    orders = relationship("OrderModel", back_populates="portfolio", cascade="all, delete")
