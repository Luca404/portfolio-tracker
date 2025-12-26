from sqlalchemy import Column, DateTime, Float, String, text
from .base import Base


class ETFPriceCacheModel(Base):
    __tablename__ = "etf_price_cache"
    isin = Column(String, primary_key=True, index=True)
    last_price = Column(Float, default=0.0)
    currency = Column(String, default="")
    history_json = Column(String, default="")
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class StockPriceCacheModel(Base):
    __tablename__ = "stock_price_cache"
    symbol = Column(String, primary_key=True, index=True)
    last_price = Column(Float, default=0.0)
    history_json = Column(String, default="")
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class ExchangeRateCacheModel(Base):
    __tablename__ = "exchange_rate_cache"
    pair = Column(String, primary_key=True, index=True)  # es: "USDEUR=X"
    history_json = Column(String, default="")
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class RiskFreeRateCacheModel(Base):
    __tablename__ = "risk_free_rate_cache"
    currency = Column(String, primary_key=True, index=True)  # "USD" or "EUR"
    current_rate = Column(Float, default=0.0)  # Tasso annuale in percentuale (es: 4.5)
    history_json = Column(String, default="")  # [{"date": "...", "rate": X.XX}]
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class MarketBenchmarkCacheModel(Base):
    __tablename__ = "market_benchmark_cache"
    currency = Column(String, primary_key=True, index=True)  # "USD" (SP500) or "EUR" (VWCE)
    symbol = Column(String, default="")  # "^GSPC" or "VWCE.DE"
    last_price = Column(Float, default=0.0)
    history_json = Column(String, default="")  # [{"date": "...", "price": X.XX}]
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
