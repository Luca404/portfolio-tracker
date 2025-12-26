import os
import json
import time
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Union

import jwt
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import requests
from sqlalchemy import (Column, Date, DateTime, Float, ForeignKey, Integer,
                        String, create_engine, select, text)
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from etf_cache_ucits import ETF_UCITS_CACHE

app = FastAPI(title="Portfolio Tracker API")

# Security
def load_dotenv():
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Database (SQLite for persistence)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./portfolio.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 30})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

FMP_API_KEY = os.environ.get("FMP_API_KEY")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
AV_BASE = "https://www.alphavantage.co/query"
SYMBOL_CACHE = {}
DATE_FMT = "%d-%m-%Y"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from routers import auth_router, portfolios_router, orders_router

# Import models and create tables
from models import (
    Base, UserModel, PortfolioModel, OrderModel,
    ETFPriceCacheModel, StockPriceCacheModel, ExchangeRateCacheModel,
    RiskFreeRateCacheModel, MarketBenchmarkCacheModel
)
from utils.database import engine as db_engine, run_migrations

Base.metadata.create_all(bind=db_engine)
run_migrations()

app.include_router(auth_router)
app.include_router(portfolios_router)
app.include_router(orders_router)


def format_date(d: Optional[date]):
    return d.strftime(DATE_FMT) if d else None


def format_datetime(dt: Optional[datetime]):
    return dt.strftime(f"{DATE_FMT} %H:%M:%S") if dt else None


def parse_date_input(val) -> Optional[date]:
    """
    Converte una data in date (day-first) cercando vari formati comuni.
    """
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        for fmt in [DATE_FMT, "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"]:
            try:
                return datetime.strptime(val, fmt).date()
            except Exception:
                continue
        try:
            return pd.to_datetime(val, dayfirst=True).date()
        except Exception:
            return None
    try:
        return pd.to_datetime(val, dayfirst=True).date()
    except Exception:
        return None


# Database models
# class UserModel(Base):
#     __tablename__ = "users"
#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True, nullable=False)
#     username = Column(String, unique=True, index=True, nullable=False)
#     hashed_password = Column(String, nullable=False)
#     created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
# 
#     portfolios = relationship("PortfolioModel", back_populates="user", cascade="all, delete")
# 
# 
# class PortfolioModel(Base):
#     __tablename__ = "portfolios"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
#     name = Column(String, nullable=False)
#     description = Column(String, default="")
#     initial_capital = Column(Float, default=0.0)  # kept for schema compatibility, not used in logic
#     reference_currency = Column(String, default="EUR")  # Valuta di riferimento per conversioni
#     risk_free_source = Column(String, default="auto")  # "auto", "USD_TREASURY", "EUR_ECB", oppure valore custom
#     market_benchmark = Column(String, default="auto")  # "auto", "SP500", "VWCE", oppure ticker custom
#     created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
# 
#     user = relationship("UserModel", back_populates="portfolios")
#     orders = relationship("OrderModel", back_populates="portfolio", cascade="all, delete")
# 
# 
# class OrderModel(Base):
#     __tablename__ = "orders"
#     id = Column(Integer, primary_key=True, index=True)
#     portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
#     symbol = Column(String, nullable=False)
#     isin = Column(String, default="")
#     ter = Column(String, default="")
#     name = Column(String, default="")
#     exchange = Column(String, default="")
#     currency = Column(String, default="")
#     quantity = Column(Float, nullable=False)
#     price = Column(Float, nullable=False)
#     commission = Column(Float, default=0.0)
#     instrument_type = Column(String, default="stock")
#     order_type = Column(String, nullable=False)  # buy / sell
#     date = Column(Date, nullable=False)
#     created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
# 
#     portfolio = relationship("PortfolioModel", back_populates="orders")
# 
# 
# class ETFPriceCacheModel(Base):
#     __tablename__ = "etf_price_cache"
#     isin = Column(String, primary_key=True, index=True)
#     last_price = Column(Float, default=0.0)
#     currency = Column(String, default="")
#     history_json = Column(String, default="")
#     updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
# 
# 
# class StockPriceCacheModel(Base):
#     __tablename__ = "stock_price_cache"
#     symbol = Column(String, primary_key=True, index=True)
#     last_price = Column(Float, default=0.0)
#     history_json = Column(String, default="")
#     updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
# 
# 
# class ExchangeRateCacheModel(Base):
#     __tablename__ = "exchange_rate_cache"
#     pair = Column(String, primary_key=True, index=True)  # es: "USDEUR=X"
#     history_json = Column(String, default="")
#     updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
# 
# 
# class RiskFreeRateCacheModel(Base):
#     __tablename__ = "risk_free_rate_cache"
#     currency = Column(String, primary_key=True, index=True)  # "USD" or "EUR"
#     current_rate = Column(Float, default=0.0)  # Tasso annuale in percentuale (es: 4.5)
#     history_json = Column(String, default="")  # [{"date": "...", "rate": X.XX}]
#     updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
# 
# 
# class MarketBenchmarkCacheModel(Base):
#     __tablename__ = "market_benchmark_cache"
#     currency = Column(String, primary_key=True, index=True)  # "USD" (SP500) or "EUR" (VWCE)
#     symbol = Column(String, default="")  # "^GSPC" or "VWCE.DE"
#     last_price = Column(Float, default=0.0)
#     history_json = Column(String, default="")  # [{"date": "...", "price": X.XX}]
#     updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
# 
# 
Base.metadata.create_all(bind=engine)


def ensure_orders_table_has_isin():
    """
    Minimal migration for existing SQLite DBs: add `isin` column if missing.
    """
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(orders)")).fetchall()]
    if "isin" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE orders ADD COLUMN isin STRING"))
    if "ter" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE orders ADD COLUMN ter STRING"))


def ensure_portfolios_table_has_reference_currency():
    """
    Minimal migration for existing SQLite DBs: add `reference_currency` column if missing.
    """
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(portfolios)")).fetchall()]
    if "reference_currency" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE portfolios ADD COLUMN reference_currency STRING DEFAULT 'EUR'"))


def ensure_portfolios_table_has_market_settings():
    """
    Minimal migration for existing SQLite DBs: add `risk_free_source` and `market_benchmark` columns if missing.
    """
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(portfolios)")).fetchall()]
    if "risk_free_source" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE portfolios ADD COLUMN risk_free_source STRING DEFAULT 'auto'"))
    if "market_benchmark" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE portfolios ADD COLUMN market_benchmark STRING DEFAULT 'auto'"))


ensure_orders_table_has_isin()
ensure_portfolios_table_has_reference_currency()
ensure_portfolios_table_has_market_settings()


# Cache invalidation: valida solo se il dato più recente è odierno o max 3 giorni fa
# (per gestire weekend/festivi)
MAX_DAYS_STALE = 3  # Accetta dati fino a 3 giorni fa come "freschi"


def commit_with_retry(db: Session, retries: int = 3, base_delay: float = 0.3):
    """
    Commit helper per gestire i transient "database is locked" di SQLite.
    """
    for attempt in range(retries):
        try:
            db.commit()
            return
        except OperationalError as e:
            if "database is locked" in str(e).lower():
                db.rollback()
                time.sleep(base_delay * (attempt + 1))
                continue
            db.rollback()
            raise
    raise HTTPException(status_code=500, detail="Database is locked, please retry")


def is_cache_data_fresh(history: list) -> bool:
    """
    Verifica se i dati storici in cache sono ancora validi.
    Cache è valida se il dato più recente è:
    - Di oggi
    - Di ieri/l'altro ieri (max 3 giorni fa) per gestire weekend/festivi

    Args:
        history: Lista di dict con chiave "date" in formato DATE_FMT

    Returns:
        True se la cache è ancora fresca, False altrimenti
    """
    if not history:
        return False

    try:
        # Prendi la data più recente dalla history
        latest_date_str = history[-1].get("date")
        if not latest_date_str:
            return False

        latest_date = parse_date_input(latest_date_str)
        if not latest_date:
            return False

        # Calcola differenza in giorni
        today = datetime.now(timezone.utc).date()
        days_old = (today - latest_date).days

        # Cache valida se dato più recente ha max MAX_DAYS_STALE giorni
        return days_old <= MAX_DAYS_STALE

    except Exception:
        return False


def merge_historical_data(cached_history: list, new_history: list) -> list:
    """
    Unisce i dati storici esistenti in cache con nuovi dati scaricati,
    evitando duplicati e mantenendo solo i dati più recenti per ogni data.

    Args:
        cached_history: Dati già presenti in cache
        new_history: Nuovi dati scaricati dall'API

    Returns:
        Lista unificata e ordinata per data
    """
    if not cached_history:
        return new_history

    if not new_history:
        return cached_history

    # Determina quale chiave usare (price o rate)
    value_key = "price" if "price" in (new_history[0] if new_history else cached_history[0]) else "rate"

    # Usa dict per evitare duplicati (chiave=data, valore=prezzo/rate)
    # I nuovi dati sovrascrivono quelli vecchi se stessa data
    merged = {}

    for item in cached_history:
        date_str = item.get("date")
        if date_str:
            merged[date_str] = item.get(value_key)

    for item in new_history:
        date_str = item.get("date")
        if date_str:
            merged[date_str] = item.get(value_key)

    # Converti in lista e ordina
    result = [{"date": d, value_key: v} for d, v in merged.items()]
    result.sort(key=lambda x: x["date"])
    return result


def normalize_chart_history(df):
    """
    Converte il DataFrame restituito da justetf_scraping.load_chart in una lista di dict {date, price}.
    È tollerante ai nomi delle colonne (price/close ecc).
    """
    if df is None:
        return []

    if isinstance(df, pd.Series):
        df = df.to_frame(name="price").reset_index()
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    # se non c'è una colonna data esplicita, usa l'indice se non è RangeIndex
    if not any("date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower() for c in df.columns):
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index().rename(columns={"index": "date"})

    cols_lower = {str(c).lower(): c for c in df.columns}
    date_col = None
    for key in ["date", "time", "timestamp"]:
        if key in cols_lower:
            date_col = cols_lower[key]
            break
    if date_col is None:
        # fallback alla prima colonna
        date_col = df.columns[0]

    # price column: priorità a price/close/nav, altrimenti prima numerica non data
    price_col = None
    for key in ["price", "close", "nav", "adjclose", "value", "quote"]:
        if key in cols_lower:
            price_col = cols_lower[key]
            break
    if price_col is None:
        numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            price_col = numeric_cols[0]
        else:
            # se index numerico e single column, usa quella
            if df.shape[1] == 1:
                price_col = df.columns[0]
            else:
                print(f"[justetf normalize] impossibile trovare price column in {df.columns}")
                return []

    ser = df[date_col]
    if pd.api.types.is_numeric_dtype(ser):
        max_val = pd.to_numeric(ser, errors="coerce").max()
        unit = None
        if pd.notnull(max_val):
            if max_val > 1e12:
                unit = "ms"
            elif max_val > 1e9:
                unit = "s"
        df[date_col] = pd.to_datetime(ser, unit=unit, errors="coerce")
    else:
        df[date_col] = df[date_col].apply(parse_date_input)

    df = df.dropna(subset=[date_col])
    current_year = datetime.now(timezone.utc).year
    df = df[df[date_col].apply(lambda x: x.year if isinstance(x, date) else None).between(2000, current_year + 1)]

    df = df.sort_values(date_col)
    history = []
    for _, row in df.iterrows():
        try:
            ts = row[date_col]
            if isinstance(ts, datetime):
                ts = ts.date()
            price_val = float(row[price_col])
            history.append({"date": ts.strftime(DATE_FMT), "price": price_val})
        except Exception:
            continue

    return history


def get_etf_price_and_history(isin: str, db: Session):
    """
    Restituisce prezzo corrente e storico per ETF con cache intelligente su SQLite.
    Cascade: yfinance (ISIN) → justetf_scraping

    yfinance come prima scelta perché:
    - Molti ETF hanno ISIN riconosciuto da yfinance
    - Dati più affidabili e consistenti
    - justetf come fallback per ETF non supportati

    Cache valida solo se il dato più recente è di oggi (o max 3 giorni fa per weekend).
    """
    if not isin:
        raise HTTPException(status_code=400, detail="Missing ISIN for ETF price lookup")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cache = db.execute(select(ETFPriceCacheModel).where(ETFPriceCacheModel.isin == isin)).scalar_one_or_none()

    cached_history = []
    cached_price = 0.0
    cache_is_fresh = False

    if cache:
        # Carica e normalizza i dati dalla cache
        history_raw = json.loads(cache.history_json or "[]")
        if history_raw:
            # Parse date per ordinamento cronologico corretto
            fixed_with_dates = []
            for h in history_raw:
                d = parse_date_input(h.get("date"))
                if d and d.year >= 2000:  # Filtra date troppo vecchie
                    fixed_with_dates.append({"date_obj": d, "date_str": d.strftime(DATE_FMT), "price": h.get("price")})

            # Ordina cronologicamente (per date_obj, non stringa)
            fixed_with_dates.sort(key=lambda x: x["date_obj"])

            # Converti a formato finale
            cached_history = [{"date": item["date_str"], "price": item["price"]} for item in fixed_with_dates]

        cached_price = cache.last_price or (cached_history[-1]["price"] if cached_history else 0.0)

        # Nuova logica: cache è fresca se il dato più recente è odierno (o max 3gg fa)
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_price:
            print(f"[ETF] {isin}: cache hit ({len(cached_history)} days)")
            return {"last_price": cached_price, "history": cached_history, "currency": cache.currency}

    errors = []

    # Tentativo 1: yfinance con ISIN
    try:
        print(f"[ETF] {isin}: trying yfinance with ISIN...")
        ticker = yf.Ticker(isin)

        # Scarica dati storici
        hist = ticker.history(period="max")

        if not hist.empty:
            # Filtra date future
            today = datetime.now(timezone.utc).date()

            history_with_dates = []
            for idx, row in hist.iterrows():
                try:
                    dt = idx.date() if hasattr(idx, 'date') else idx

                    # Skip date future
                    if dt > today:
                        continue

                    price = float(row["Close"])
                    if price > 0:
                        history_with_dates.append({"date_obj": dt, "price": price})
                except Exception:
                    continue

            if history_with_dates:
                # Ordina per data
                history_with_dates.sort(key=lambda x: x["date_obj"])

                # Converti a formato string
                new_history = [
                    {"date": item["date_obj"].strftime(DATE_FMT), "price": item["price"]}
                    for item in history_with_dates
                ]

                # Merge con cache esistente
                merged_history = merge_historical_data(cached_history, new_history)

                last_price = merged_history[-1]["price"]
                currency = ""

                print(f"[ETF] {isin}: yfinance OK, {len(merged_history)} days, last={last_price:.2f}")

                # Aggiorna cache
                if cache:
                    cache.last_price = last_price
                    cache.currency = currency
                    cache.history_json = json.dumps(merged_history)
                    cache.updated_at = now
                else:
                    cache = ETFPriceCacheModel(
                        isin=isin,
                        last_price=last_price,
                        currency=currency,
                        history_json=json.dumps(merged_history),
                        updated_at=now,
                    )
                    db.add(cache)
                commit_with_retry(db)
                return {"last_price": last_price, "history": merged_history, "currency": currency}
    except Exception as e:
        errors.append(f"yfinance: {str(e)}")
        print(f"[ETF] {isin}: yfinance failed: {str(e)}")

    # Tentativo 2: justetf_scraping
    try:
        import justetf_scraping
    except ImportError:
        error_msg = f"Nessun dato disponibile per {isin}. Errori: {'; '.join(errors)}. justetf_scraping non installato."
        raise HTTPException(status_code=500, detail=error_msg)

    def fetch_chart(identifier: str):
        try:
            return justetf_scraping.load_chart(identifier)
        except Exception:
            return None

    try:
        print(f"[ETF] {isin}: trying justetf_scraping...")
        df_chart = fetch_chart(isin)
        if df_chart is None:
            # prova anche con il ticker se diverso dall'ISIN
            df_chart = fetch_chart(isin.replace(" ", ""))

        new_history = normalize_chart_history(df_chart)
        if not new_history:
            error_msg = f"Nessun dato disponibile per {isin}. Errori: {'; '.join(errors)}"
            raise HTTPException(status_code=400, detail=error_msg)

        # Merge dei nuovi dati con quelli in cache (se presenti)
        merged_history = merge_historical_data(cached_history, new_history)

        latest_point = merged_history[-1]
        last_price = latest_point["price"]
        currency = ""

        print(f"[ETF] {isin}: justetf OK, {len(merged_history)} days, last={last_price:.2f}")

        # Aggiorna la cache con i dati unificati
        if cache:
            cache.last_price = last_price
            cache.currency = currency
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = ETFPriceCacheModel(
                isin=isin,
                last_price=last_price,
                currency=currency,
                history_json=json.dumps(merged_history),
                updated_at=now,
            )
            db.add(cache)
        commit_with_retry(db)
        return {"last_price": last_price, "history": merged_history, "currency": currency}
    except HTTPException:
        # Se abbiamo cache non fresca ma valida, usala come fallback
        if cached_history and cached_price:
            print(f"[ETF] {isin}: using stale cache as fallback")
            return {"last_price": cached_price, "history": cached_history, "currency": cache.currency}
        raise
    except Exception as e:
        errors.append(f"justetf: {str(e)}")
        # Se abbiamo cache non fresca ma valida, usala come fallback
        if cached_history and cached_price:
            print(f"[ETF] {isin}: using stale cache as fallback")
            return {"last_price": cached_price, "history": cached_history, "currency": cache.currency}
        error_msg = f"Nessun dato disponibile per {isin}. Errori: {'; '.join(errors)}"
        raise HTTPException(status_code=500, detail=error_msg)


def get_stock_price_from_alphavantage(symbol: str) -> dict:
    """
    Fallback ad AlphaVantage per ottenere prezzi storici quando FMP non supporta il ticker.
    Usa TIME_SERIES_DAILY con outputsize=compact (100 giorni, free tier).
    """
    if not ALPHAVANTAGE_API_KEY:
        print(f"[STOCK] {symbol}: AlphaVantage API key not configured")
        raise Exception("AlphaVantage API key not configured")

    symbol = symbol.upper()

    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",  # compact = ultimi 100 giorni (free tier)
            "apikey": ALPHAVANTAGE_API_KEY
        }

        print(f"[STOCK] {symbol}: trying AlphaVantage...")
        response = requests.get(AV_BASE, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Verifica errori API
        if "Error Message" in data:
            error_msg = data['Error Message']
            print(f"[STOCK] {symbol}: AlphaVantage error: {error_msg}")
            raise Exception(f"AlphaVantage error: {error_msg}")

        if "Note" in data:
            print(f"[STOCK] {symbol}: AlphaVantage rate limit")
            raise Exception("AlphaVantage rate limit exceeded")

        if "Information" in data:
            # Messaggio informativo (es. premium feature)
            print(f"[STOCK] {symbol}: AlphaVantage info: {data['Information']}")
            raise Exception("AlphaVantage: Premium feature or API limit")

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            print(f"[STOCK] {symbol}: AlphaVantage no time series data")
            raise Exception("AlphaVantage: No time series data")

        # Converti in formato standard, filtra date future
        today = datetime.now(timezone.utc).date()
        history_with_dates = []

        for date_str, values in time_series.items():
            try:
                dt = parse_date_input(date_str)
                if dt and dt <= today:  # Skip date future
                    price = float(values.get("4. close", 0))
                    if price > 0:
                        history_with_dates.append({"date_obj": dt, "price": price})
            except Exception:
                continue

        if not history_with_dates:
            print(f"[STOCK] {symbol}: AlphaVantage no valid history after parsing")
            raise Exception("AlphaVantage: No valid historical data")

        # Ordina cronologicamente per data oggetto
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string dopo ordinamento
        history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "price": item["price"]}
            for item in history_with_dates
        ]

        last_price = history[-1]["price"]
        last_date = history[-1]["date"]
        print(f"[STOCK] {symbol}: AlphaVantage OK, {len(history)} days, last_date={last_date}, last={last_price:.2f}")
        return {"last_price": last_price, "history": history}

    except Exception as e:
        print(f"[STOCK] {symbol}: AlphaVantage failed: {str(e)}")
        raise


def get_stock_price_from_yfinance(symbol: str, days: int = 180) -> dict:
    """
    Fallback finale a yfinance quando FMP e AlphaVantage falliscono.
    """
    try:
        print(f"[STOCK] {symbol}: trying yfinance...")
        ticker = yf.Ticker(symbol)

        # Scarica dati storici
        hist = ticker.history(period="max" if days > 365 else "1y")

        if hist.empty:
            print(f"[STOCK] {symbol}: yfinance no data")
            raise Exception("yfinance: No data available")

        # Filtra date future (yfinance può dare dati pre-market con date future)
        today = datetime.now(timezone.utc).date()

        # Tieni traccia di date come oggetti per ordinamento corretto
        history_with_dates = []
        for idx, row in hist.iterrows():
            try:
                dt = idx.date() if hasattr(idx, 'date') else idx

                # Skip date future
                if dt > today:
                    continue

                price = float(row["Close"])
                if price > 0:
                    history_with_dates.append({"date_obj": dt, "price": price})
            except Exception:
                continue

        if not history_with_dates:
            print(f"[STOCK] {symbol}: yfinance no valid history")
            raise Exception("yfinance: No valid historical data")

        # Ordina per data oggetto (cronologicamente), non stringa
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string dopo ordinamento
        history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "price": item["price"]}
            for item in history_with_dates
        ]

        last_price = history[-1]["price"]
        last_date = history[-1]["date"]
        print(f"[STOCK] {symbol}: yfinance OK, {len(history)} days, last_date={last_date}, last={last_price:.2f}")
        return {"last_price": last_price, "history": history}

    except Exception as e:
        print(f"[STOCK] {symbol}: yfinance failed: {str(e)}")
        raise


def get_stock_price_and_history(symbol: str, days: int = 180):
    """
    Ottiene prezzo corrente e storico di un titolo azionario.
    Cascade: yfinance → FMP → AlphaVantage

    yfinance come prima scelta perché:
    - Dati illimitati e gratuiti
    - Nessun rate limit
    - Supporto eccellente per stock globali
    - FMP e AlphaVantage come fallback per edge cases
    """
    symbol = symbol.upper()

    def parse_history(raw):
        history = []
        for row in raw or []:
            try:
                dt_raw = row.get("date")
                dt_date = parse_date_input(dt_raw)
                dt = dt_date.strftime(DATE_FMT) if dt_date else None
                price_val = row.get("close") or row.get("adjClose") or row.get("open") or row.get("price") or 0
                if dt and price_val is not None:
                    history.append({"date": dt, "price": float(price_val)})
            except Exception:
                continue
        history = [h for h in history if h.get("price") is not None]
        history.sort(key=lambda x: x["date"])
        return history

    errors = []

    # Tentativo 1: yfinance (prima scelta - dati illimitati, nessun rate limit)
    try:
        return get_stock_price_from_yfinance(symbol, days)
    except Exception as e:
        errors.append(f"yfinance: {str(e)}")

    # Tentativo 2: FMP historical-price-eod
    try:
        data = fmp_get("historical-price-eod/full", {"symbol": symbol, "limit": days}, base=FMP_STABLE_BASE)
        if isinstance(data, dict):
            history = parse_history(data.get("historical") or data.get("data"))
        elif isinstance(data, list):
            history = parse_history(data)
        if history:
            print(f"[STOCK] {symbol}: FMP, {len(history)} days")
            last_price = history[-1]["price"]
            return {"last_price": last_price, "history": history}
    except Exception as e:
        errors.append(f"FMP EOD: {str(e)}")

    # Tentativo 3: FMP historical-chart (alternativa FMP)
    try:
        alt = fmp_get(f"historical-chart/1day/{symbol}", {"limit": days}, base=FMP_STABLE_BASE)
        history = parse_history(alt)
        if history:
            print(f"[STOCK] {symbol}: FMP chart, {len(history)} days")
            last_price = history[-1]["price"]
            return {"last_price": last_price, "history": history}
    except Exception as e:
        errors.append(f"FMP chart: {str(e)}")

    # Tentativo 4: AlphaVantage (ultima spiaggia per edge cases)
    try:
        return get_stock_price_from_alphavantage(symbol)
    except Exception as e:
        errors.append(f"AlphaVantage: {str(e)}")

    # Se arriviamo qui, tutti i provider hanno fallito
    error_msg = f"Nessun dato disponibile per {symbol}. Errori: {'; '.join(errors)}"
    print(f"[STOCK] {symbol}: ALL PROVIDERS FAILED")
    raise HTTPException(status_code=400, detail=error_msg)


def get_stock_price_and_history_cached(symbol: str, db: Session, days: int = 180):
    """
    Wrapper con cache intelligente su SQLite per i prezzi storici delle azioni.
    Cache valida solo se il dato più recente è di oggi (o max 3 giorni fa per weekend).
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cache = db.execute(select(StockPriceCacheModel).where(StockPriceCacheModel.symbol == symbol)).scalar_one_or_none()

    cached_history = []
    cached_price = 0.0
    cache_is_fresh = False

    if cache:
        # Carica e normalizza i dati dalla cache
        history_raw = json.loads(cache.history_json or "[]")
        if history_raw:
            # Parse date per ordinamento cronologico corretto
            fixed_with_dates = []
            for h in history_raw:
                d = parse_date_input(h.get("date"))
                if d:
                    fixed_with_dates.append({"date_obj": d, "date_str": d.strftime(DATE_FMT), "price": h.get("price")})

            # Ordina cronologicamente (per date_obj, non stringa)
            fixed_with_dates.sort(key=lambda x: x["date_obj"])

            # Converti a formato finale
            cached_history = [{"date": item["date_str"], "price": item["price"]} for item in fixed_with_dates]

        cached_price = cache.last_price or (cached_history[-1]["price"] if cached_history else 0.0)

        # Nuova logica: cache è fresca se il dato più recente è odierno (o max 3gg fa)
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_price:
            print(f"[STOCK] {symbol}: cache hit ({len(cached_history)} days, last={cached_price:.2f})")
            return {"last_price": cached_price, "history": cached_history}
        else:
            if cached_history:
                latest_date = cached_history[-1].get("date") if cached_history else "none"
                print(f"[STOCK] {symbol}: cache stale (last date={latest_date})")

    # Scarica nuovi dati dall'API
    data = get_stock_price_and_history(symbol, days=days)
    new_history = data.get("history", [])
    last_price = data.get("last_price", 0.0)

    if new_history and last_price:
        # Merge dei nuovi dati con quelli in cache (se presenti)
        merged_history = merge_historical_data(cached_history, new_history)

        # Aggiorna la cache con i dati unificati
        if cache:
            cache.last_price = last_price
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = StockPriceCacheModel(
                symbol=symbol,
                last_price=last_price,
                history_json=json.dumps(merged_history),
                updated_at=now,
            )
            db.add(cache)
        commit_with_retry(db)

        return {"last_price": last_price, "history": merged_history}

    return data


def convert_to_reference_currency(amount: float, from_currency: str, to_currency: str, db: Session) -> float:
    """
    Converte un importo dalla valuta from_currency alla valuta to_currency
    usando l'ultimo tasso di cambio disponibile.
    """
    if from_currency == to_currency or not from_currency or not to_currency:
        return amount

    try:
        fx_data = get_exchange_rate_history(from_currency, to_currency, db)
        fx_rates = fx_data.get("rates", [])
        if fx_rates:
            latest_rate = fx_rates[-1].get("rate", 1.0)
            return amount * latest_rate
    except Exception as e:
        print(f"[FX] Conversion {from_currency}->{to_currency} failed: {e}")

    return amount


def get_exchange_rate_history(from_currency: str, to_currency: str, db: Session) -> dict:
    """
    Ottiene lo storico dei tassi di cambio tra due valute usando yfinance.
    Esempio: USD -> EUR usa il ticker "USDEUR=X" su yfinance

    Ritorna: {"rates": [{"date": "DD-MM-YYYY", "rate": 0.92}, ...]}
    """
    # Se stessa valuta, ritorna rate 1.0
    if from_currency == to_currency:
        return {"rates": []}

    # Crea pair identifier per yfinance
    pair = f"{from_currency}{to_currency}=X"

    now = datetime.now(timezone.utc)

    # Controlla cache
    cache = db.query(ExchangeRateCacheModel).filter_by(pair=pair).first()
    cached_rates = []

    if cache:
        history_raw = json.loads(cache.history_json or "[]")
        if history_raw:
            # Parse e ordina date
            fixed_with_dates = []
            for h in history_raw:
                d = parse_date_input(h.get("date"))
                if d and d.year >= 2000:
                    fixed_with_dates.append({
                        "date_obj": d,
                        "date_str": d.strftime(DATE_FMT),
                        "rate": h.get("rate")
                    })
            fixed_with_dates.sort(key=lambda x: x["date_obj"])
            cached_rates = [{"date": item["date_str"], "rate": item["rate"]} for item in fixed_with_dates]

        # Cache fresca se ultimo dato è recente
        cache_is_fresh = is_cache_data_fresh(cached_rates)

        if cache_is_fresh and cached_rates:
            print(f"[FX] {pair}: cache hit ({len(cached_rates)} days)")
            return {"rates": cached_rates}
        else:
            if cached_rates:
                latest_date = cached_rates[-1].get("date") if cached_rates else "none"
                print(f"[FX] {pair}: cache stale (last date={latest_date})")

    # Scarica dati da yfinance
    try:
        print(f"[FX] {pair}: downloading from yfinance...")
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="max")

        if hist.empty:
            print(f"[FX] {pair}: no data from yfinance")
            # Ritorna cache anche se stale, meglio di niente
            return {"rates": cached_rates} if cached_rates else {"rates": []}

        today = datetime.now(timezone.utc).date()
        rates_with_dates = []

        for idx, row in hist.iterrows():
            try:
                dt = idx.date() if hasattr(idx, 'date') else idx
                if dt > today:
                    continue
                rate = float(row['Close'])
                if rate > 0:
                    rates_with_dates.append({"date_obj": dt, "rate": rate})
            except Exception:
                continue

        if not rates_with_dates:
            print(f"[FX] {pair}: no valid data after parsing")
            return {"rates": cached_rates} if cached_rates else {"rates": []}

        # Ordina cronologicamente
        rates_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string
        new_rates = [
            {"date": item["date_obj"].strftime(DATE_FMT), "rate": item["rate"]}
            for item in rates_with_dates
        ]

        # Merge con cache esistente
        merged_rates = merge_historical_data(cached_rates, new_rates)

        # Aggiorna cache
        if cache:
            cache.history_json = json.dumps(merged_rates)
            cache.updated_at = now
        else:
            cache = ExchangeRateCacheModel(
                pair=pair,
                history_json=json.dumps(merged_rates),
                updated_at=now
            )
            db.add(cache)
        commit_with_retry(db)

        print(f"[FX] {pair}: OK, {len(merged_rates)} days")
        return {"rates": merged_rates}

    except Exception as e:
        print(f"[FX] {pair}: error: {str(e)}")
        # Ritorna cache anche se stale
        return {"rates": cached_rates} if cached_rates else {"rates": []}


def fetch_us_treasury_rate(db: Session) -> dict:
    """
    Scarica i tassi US Treasury da FMP API.
    Usa l'endpoint: https://financialmodelingprep.com/stable/treasury-rates

    Ritorna: {"current_rate": 4.5, "history": [{"date": "DD-MM-YYYY", "rate": 4.5}, ...]}
    """
    currency = "USD"
    now = datetime.now(timezone.utc)

    # Controlla cache
    cache = db.query(RiskFreeRateCacheModel).filter_by(currency=currency).first()
    cached_history = []

    if cache:
        cached_history = json.loads(cache.history_json or "[]")
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_history:
            print(f"[RiskFree] USD Treasury: cache hit ({len(cached_history)} days)")
            return {"current_rate": cache.current_rate, "history": cached_history}

    # Scarica da FMP
    try:
        data = fmp_get("treasury-rates", {}, base=FMP_STABLE_BASE)

        if not data or not isinstance(data, list):
            print(f"[RiskFree] USD Treasury: no data from FMP")
            return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}

        # Estrai il tasso a 10 anni (standard per risk-free)
        history_with_dates = []
        current_rate = 0.0

        for item in data:
            try:
                # FMP ritorna: {"date": "YYYY-MM-DD", "month": X, "year1": X, "year10": X, ...}
                date_str = item.get("date")
                rate_10y = item.get("year10")

                if date_str and rate_10y is not None:
                    dt = parse_date_input(date_str)
                    if dt:
                        history_with_dates.append({
                            "date_obj": dt,
                            "rate": float(rate_10y)
                        })
            except Exception:
                continue

        if not history_with_dates:
            print(f"[RiskFree] USD Treasury: no valid data after parsing")
            return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}

        # Ordina cronologicamente
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string
        new_history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "rate": item["rate"]}
            for item in history_with_dates
        ]

        # Ultimo tasso disponibile
        current_rate = new_history[-1]["rate"] if new_history else 0.0

        # Merge con cache
        merged_history = merge_historical_data(cached_history, new_history)

        # Aggiorna cache
        if cache:
            cache.current_rate = current_rate
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = RiskFreeRateCacheModel(
                currency=currency,
                current_rate=current_rate,
                history_json=json.dumps(merged_history),
                updated_at=now
            )
            db.add(cache)
        commit_with_retry(db)

        print(f"[RiskFree] USD Treasury: OK, current={current_rate}%, {len(merged_history)} days")
        return {"current_rate": current_rate, "history": merged_history}

    except Exception as e:
        print(f"[RiskFree] USD Treasury: error: {str(e)}")
        return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}


def fetch_ecb_rate(db: Session) -> dict:
    """
    Scarica i tassi ECB (European Central Bank) dall'API ufficiale.
    Usa il tasso di riferimento principale (Main Refinancing Operations).

    API: https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.MRR_FR.LEV

    Ritorna: {"current_rate": 4.5, "history": [{"date": "DD-MM-YYYY", "rate": 4.5}, ...]}
    """
    currency = "EUR"
    now = datetime.now(timezone.utc)

    # Controlla cache
    cache = db.query(RiskFreeRateCacheModel).filter_by(currency=currency).first()
    cached_history = []

    if cache:
        cached_history = json.loads(cache.history_json or "[]")
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_history:
            print(f"[RiskFree] EUR ECB: cache hit ({len(cached_history)} days)")
            return {"current_rate": cache.current_rate, "history": cached_history}

    # Scarica da ECB API
    try:
        # API ECB per Main Refinancing Operations rate
        url = "https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.MRR_FR.LEV"
        params = {"format": "jsondata", "detail": "dataonly"}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse ECB JSON structure
        observations = data.get("dataSets", [{}])[0].get("series", {}).get("0:0:0:0:0:0:0", {}).get("observations", {})
        dimensions = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_values = None

        for dim in dimensions:
            if dim.get("id") == "TIME_PERIOD":
                time_values = dim.get("values", [])
                break

        if not observations or not time_values:
            print(f"[RiskFree] EUR ECB: no data from ECB API")
            return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}

        history_with_dates = []

        for idx, value_list in observations.items():
            try:
                time_idx = int(idx)
                if time_idx < len(time_values):
                    date_str = time_values[time_idx].get("id")  # formato: YYYY-MM-DD
                    rate = float(value_list[0])

                    dt = parse_date_input(date_str)
                    if dt:
                        history_with_dates.append({
                            "date_obj": dt,
                            "rate": rate
                        })
            except Exception:
                continue

        if not history_with_dates:
            print(f"[RiskFree] EUR ECB: no valid data after parsing")
            return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}

        # Ordina cronologicamente
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string
        new_history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "rate": item["rate"]}
            for item in history_with_dates
        ]

        # Ultimo tasso disponibile
        current_rate = new_history[-1]["rate"] if new_history else 0.0

        # Merge con cache
        merged_history = merge_historical_data(cached_history, new_history)

        # Aggiorna cache
        if cache:
            cache.current_rate = current_rate
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = RiskFreeRateCacheModel(
                currency=currency,
                current_rate=current_rate,
                history_json=json.dumps(merged_history),
                updated_at=now
            )
            db.add(cache)
        commit_with_retry(db)

        print(f"[RiskFree] EUR ECB: OK, current={current_rate}%, {len(merged_history)} days")
        return {"current_rate": current_rate, "history": merged_history}

    except Exception as e:
        print(f"[RiskFree] EUR ECB: error: {str(e)}")
        return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}


def fetch_market_benchmark(currency: str, db: Session) -> dict:
    """
    Scarica i dati del benchmark di mercato appropriato per la valuta.
    - USD: S&P 500 (^GSPC)
    - EUR: VWCE (VWCE.DE - Vanguard FTSE All-World)

    Ritorna: {"symbol": "^GSPC", "last_price": 4500, "history": [{"date": "DD-MM-YYYY", "price": 4500}, ...]}
    """
    # Mappa valuta -> simbolo
    symbol_map = {
        "USD": "^GSPC",      # S&P 500
        "EUR": "VWCE.DE"     # Vanguard FTSE All-World (quotato in EUR su XETRA)
    }

    symbol = symbol_map.get(currency.upper())
    if not symbol:
        raise HTTPException(status_code=400, detail=f"No market benchmark defined for currency {currency}")

    now = datetime.now(timezone.utc)

    # Controlla cache
    cache = db.query(MarketBenchmarkCacheModel).filter_by(currency=currency.upper()).first()
    cached_history = []

    if cache:
        cached_history = json.loads(cache.history_json or "[]")
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_history:
            print(f"[Benchmark] {currency} ({symbol}): cache hit ({len(cached_history)} days)")
            return {
                "symbol": symbol,
                "last_price": cache.last_price,
                "history": cached_history
            }

    # Scarica da yfinance
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max")

        if hist.empty:
            print(f"[Benchmark] {currency} ({symbol}): no data from yfinance")
            return {
                "symbol": symbol,
                "last_price": cache.last_price if cache else 0.0,
                "history": cached_history
            }

        today = datetime.now(timezone.utc).date()
        history_with_dates = []

        for idx, row in hist.iterrows():
            try:
                dt = idx.date() if hasattr(idx, 'date') else idx
                if dt > today:
                    continue
                price = float(row['Close'])
                if price > 0:
                    history_with_dates.append({"date_obj": dt, "price": price})
            except Exception:
                continue

        if not history_with_dates:
            print(f"[Benchmark] {currency} ({symbol}): no valid data after parsing")
            return {
                "symbol": symbol,
                "last_price": cache.last_price if cache else 0.0,
                "history": cached_history
            }

        # Ordina cronologicamente
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string
        new_history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "price": item["price"]}
            for item in history_with_dates
        ]

        # Ultimo prezzo disponibile
        last_price = new_history[-1]["price"] if new_history else 0.0

        # Merge con cache
        merged_history = merge_historical_data(cached_history, new_history)

        # Aggiorna cache
        if cache:
            cache.symbol = symbol
            cache.last_price = last_price
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = MarketBenchmarkCacheModel(
                currency=currency.upper(),
                symbol=symbol,
                last_price=last_price,
                history_json=json.dumps(merged_history),
                updated_at=now
            )
            db.add(cache)
        commit_with_retry(db)

        print(f"[Benchmark] {currency} ({symbol}): OK, last={last_price}, {len(merged_history)} days")
        return {
            "symbol": symbol,
            "last_price": last_price,
            "history": merged_history
        }

    except Exception as e:
        print(f"[Benchmark] {currency} ({symbol}): error: {str(e)}")
        return {
            "symbol": symbol,
            "last_price": cache.last_price if cache else 0.0,
            "history": cached_history
        }


def get_risk_free_rate(currency: str, db: Session, custom_source: Optional[str] = None) -> float:
    """
    Helper function per ottenere il tasso risk-free corrente.

    Args:
        currency: "USD" o "EUR"
        db: Database session
        custom_source: Fonte personalizzata ("auto", "USD_TREASURY", "EUR_ECB", o valore numerico custom)

    Returns:
        Tasso risk-free annuale in percentuale (es: 4.5)
    """
    # Se custom_source è un numero, usalo direttamente
    if custom_source and custom_source != "auto":
        try:
            return float(custom_source)
        except ValueError:
            # Non è un numero, procedi con la logica normale
            if custom_source == "USD_TREASURY":
                data = fetch_us_treasury_rate(db)
                return data.get("current_rate", 0.0)
            elif custom_source == "EUR_ECB":
                data = fetch_ecb_rate(db)
                return data.get("current_rate", 0.0)

    # Auto mode: determina in base alla currency
    currency = currency.upper()
    if currency == "USD":
        data = fetch_us_treasury_rate(db)
    elif currency == "EUR":
        data = fetch_ecb_rate(db)
    else:
        print(f"[RiskFree] Unknown currency {currency}, defaulting to 0.0")
        return 0.0

    return data.get("current_rate", 0.0)


def get_market_benchmark_data(currency: str, db: Session, custom_benchmark: Optional[str] = None) -> dict:
    """
    Helper function per ottenere i dati del benchmark di mercato.

    Args:
        currency: "USD" o "EUR"
        db: Database session
        custom_benchmark: Benchmark personalizzato ("auto", "SP500", "VWCE", o ticker custom)

    Returns:
        {"symbol": "^GSPC", "last_price": 4500, "history": [...]}
    """
    # Se custom_benchmark è specificato e non è "auto"
    if custom_benchmark and custom_benchmark != "auto":
        if custom_benchmark == "SP500":
            return fetch_market_benchmark("USD", db)
        elif custom_benchmark == "VWCE":
            return fetch_market_benchmark("EUR", db)
        else:
            # Custom ticker - usa yfinance direttamente
            try:
                ticker = yf.Ticker(custom_benchmark)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    last_price = float(hist['Close'].iloc[-1])
                    history = []
                    for idx, row in hist.iterrows():
                        dt = idx.date() if hasattr(idx, 'date') else idx
                        history.append({
                            "date": dt.strftime(DATE_FMT),
                            "price": float(row['Close'])
                        })
                    return {"symbol": custom_benchmark, "last_price": last_price, "history": history}
            except Exception as e:
                print(f"[Benchmark] Error fetching custom ticker {custom_benchmark}: {e}")

    # Auto mode: determina in base alla currency
    return fetch_market_benchmark(currency, db)


# Pydantic models
class UserRegister(BaseModel):
    email: str
    username: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


class Portfolio(BaseModel):
    name: str
    description: Optional[str] = ""
    initial_capital: float = 10000.0
    reference_currency: Optional[str] = "EUR"
    risk_free_source: Optional[str] = "auto"  # "auto", "USD_TREASURY", "EUR_ECB", or custom value
    market_benchmark: Optional[str] = "auto"  # "auto", "SP500", "VWCE", or custom ticker


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


# DB helpers
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db.execute(select(UserModel).where(UserModel.email == email)).scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired user")
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def aggregate_positions(orders: List[OrderModel]):
    positions = {}
    for order in orders:
        symbol = order.symbol.upper()
        if symbol not in positions:
            positions[symbol] = {"quantity": 0.0, "total_cost": 0.0}
        pos = positions[symbol]
        if order.order_type == "buy":
            pos["quantity"] += order.quantity
            pos["total_cost"] += order.quantity * order.price + order.commission
        elif order.order_type == "sell":
            if order.quantity > pos["quantity"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot sell {order.quantity} shares of {symbol}; only {pos['quantity']} available",
                )
            pos["quantity"] -= order.quantity
            pos["total_cost"] -= order.quantity * order.price - order.commission
    positions = {k: v for k, v in positions.items() if v["quantity"] > 0}
    return positions


def xnpv(rate: float, cashflows: list):
    def _as_date(d):
        if isinstance(d, datetime):
            if d.tzinfo:
                return d.astimezone(timezone.utc).date()
            return d.date()
        if isinstance(d, date):
            return d
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            raise ValueError("invalid date")
        if getattr(ts, "tzinfo", None):
            ts = ts.tz_convert("UTC")
        return ts.date()

    if rate <= -1:
        return float("inf")
    try:
        cleaned = [(_as_date(d), cf) for d, cf in cashflows]
    except Exception:
        return float("inf")

    t0 = cleaned[0][0]
    return sum(cf / ((1 + rate) ** ((d - t0).days / 365.0)) for d, cf in cleaned)


def calc_xirr(cashflows: list):
    # cashflows: list of (date, amount)
    if not cashflows:
        return 0.0

    cleaned = []
    for dt, cf in cashflows:
        try:
            if isinstance(dt, datetime):
                if dt.tzinfo:
                    dt = dt.astimezone(timezone.utc).date()
                else:
                    dt = dt.date()
            elif isinstance(dt, date):
                dt = dt
            else:
                ts = pd.to_datetime(dt, errors="coerce")
                if pd.isna(ts):
                    continue
                if hasattr(ts, "tzinfo") and ts.tzinfo:
                    ts = ts.tz_convert("UTC")
                dt = ts.date()
            cleaned.append((dt, cf))
        except Exception:
            continue

    if not cleaned:
        return 0.0

    positive = any(cf > 0 for _, cf in cleaned)
    negative = any(cf < 0 for _, cf in cleaned)
    if not (positive and negative):
        return 0.0

    try:
        low, high = -0.9999, 10.0
        for _ in range(100):
            mid = (low + high) / 2
            npv = xnpv(mid, cleaned)
            if abs(npv) < 1e-6:
                return mid
            if npv > 0:
                low = mid
            else:
                high = mid
        return mid
    except Exception:
        return 0.0


def fmp_get(path: str, params: dict, base: Optional[str] = None):
    if not FMP_API_KEY:
        raise HTTPException(status_code=500, detail="FMP_API_KEY not configured")
    base_url = base or FMP_BASE
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    params = params.copy() if params else {}
    params["apikey"] = FMP_API_KEY
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="FMP rate limit exceeded")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"FMP error {resp.status_code}: {resp.text}")
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


def search_symbol(symbol: str, instrument_type: str):
    instrument_type = instrument_type.lower()
    
    if instrument_type == "stock":
        path = "search-symbol"
        params = {"query": symbol.upper(), "limit": 20}
        return fmp_get(path, params, base=FMP_STABLE_BASE)

    if instrument_type == "etf":
        query = symbol.upper().strip()
        if not query:
            return []
        matches = []
        for item in ETF_UCITS_CACHE:
            tickers = [item.get("ticker", ""), item.get("symbol", "")]
            # match ticker prefix/exact
            if any(query and t and t.upper().startswith(query) for t in tickers):
                sym = (item.get("ticker") or item.get("symbol") or item.get("isin") or "").upper()
                matches.append({
                    "symbol": sym,
                    "name": item.get("name", ""),
                    "exchange": item.get("exchange", "") or item.get("domicile", ""),
                    "currency": item.get("currency", ""),
                    "type": "ETF",
                    "isin": item.get("isin", ""),
                    "ticker": item.get("ticker", ""),
                    "ter": item.get("ter", ""),
                })
                if len(matches) >= 25:
                    break
            # match ISIN esatto (ISIN = 12 chars)
            elif len(query) == 12 and query == str(item.get("isin", "")).upper():
                sym = (item.get("ticker") or item.get("symbol") or item.get("isin") or "").upper()
                matches.append({
                    "symbol": sym,
                    "name": item.get("name", ""),
                    "exchange": item.get("exchange", "") or item.get("domicile", ""),
                    "currency": item.get("currency", ""),
                    "type": "ETF",
                    "isin": item.get("isin", ""),
                    "ticker": item.get("ticker", ""),
                    "ter": item.get("ter", ""),
                })
                if len(matches) >= 25:
                    break
        return matches

    return []


def ensure_symbol_exists(symbol: str, instrument_type: str):
    matches = search_symbol(symbol, instrument_type)
    for m in matches:
        if m.get("symbol", "").upper() == symbol.upper():
            return m
    raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found for type {instrument_type}")


def compute_portfolio_value(
    positions_map: dict,
    orders_by_symbol: dict,
    symbol_type_map: dict,
    symbol_isin_map: dict,
    db: Session,
    include_history: bool = False,
    reference_currency: str = "EUR",
):
    positions = []
    total_value = 0.0
    total_cost = 0.0
    position_histories = {}

    for symbol, data in positions_map.items():
        instrument_type = symbol_type_map.get(symbol, "stock")
        price_history = []
        current_price = 0.0
        fetch_error = None
        try:
            if instrument_type == "etf":
                isin = symbol_isin_map.get(symbol, "")
                etf_data = get_etf_price_and_history(isin, db)
                current_price = etf_data["last_price"]
                price_history = etf_data.get("history", [])
            else:
                stock_data = get_stock_price_and_history_cached(symbol, db, days=180 if include_history else 7)
                current_price = stock_data["last_price"]
                price_history = stock_data.get("history", [])
        except HTTPException as e:
            fetch_error = e.detail
            print(f"Error fetching {symbol}: {e.detail}")
        except Exception as e:
            fetch_error = str(e)
            print(f"Error fetching {symbol}: {e}")

        price_history = price_history or []
        current_price = current_price or 0.0

        # Conversione valuta se necessario
        # Determina valuta ufficiale asset vs valuta ordine
        orders_for_symbol = orders_by_symbol.get(symbol, [])
        if orders_for_symbol:
            order_currency = orders_for_symbol[0].currency or ""
            # Per ETF la valuta ufficiale è nella cache, per stock proviamo a determinarla
            # Per semplicità assumiamo che se non specificato, stock US = USD, altrimenti EUR
            # Idealmente dovremmo avere questo info dall'API
            if instrument_type == "etf":
                # Per ETF la currency è già disponibile dalla cache
                asset_currency = None  # Non abbiamo accesso diretto, skippiamo per ora gli ETF
            else:
                # Per stock, assumiamo USD se ticker US, altrimenti proviamo a determinare
                # Per sicurezza, solo convertiamo se sappiamo con certezza
                # Qui facciamo solo USD<->EUR per ora
                asset_currency = "USD" if order_currency == "EUR" else ("EUR" if order_currency == "USD" else None)

            # Converti se necessario
            if order_currency and asset_currency and order_currency != asset_currency and price_history:
                print(f"[FX] {symbol}: converting {asset_currency} -> {order_currency}")
                try:
                    fx_data = get_exchange_rate_history(asset_currency, order_currency, db)
                    fx_rates = fx_data.get("rates", [])

                    if fx_rates:
                        # Crea mappa data -> tasso
                        fx_map = {}
                        for rate_point in fx_rates:
                            rate_date = parse_date_input(rate_point.get("date"))
                            if rate_date:
                                fx_map[rate_date] = rate_point.get("rate", 1.0)

                        # Converti ogni prezzo storico
                        converted_history = []
                        for price_point in price_history:
                            price_date = parse_date_input(price_point.get("date"))
                            if price_date and price_date in fx_map:
                                original_price = price_point.get("price", 0.0)
                                fx_rate = fx_map[price_date]
                                converted_price = original_price * fx_rate
                                converted_history.append({
                                    "date": price_point.get("date"),
                                    "price": converted_price
                                })
                            else:
                                # Se non abbiamo tasso per quella data, skippiamo
                                pass

                        if converted_history:
                            price_history = converted_history
                            # Converti anche current_price usando ultimo tasso disponibile
                            if fx_map:
                                latest_rate = fx_rates[-1].get("rate", 1.0)
                                current_price = current_price * latest_rate
                                print(f"[FX] {symbol}: converted {len(converted_history)} prices, latest_rate={latest_rate:.4f}")
                except Exception as e:
                    print(f"[FX] {symbol}: conversion failed: {e}")

        if include_history:
            if instrument_type == "etf":
                print(f"[compute] ETF {symbol}: price={current_price}, history_len={len(price_history)}")
            else:
                print(f"[compute] Stock {symbol}: price={current_price}, history_len={len(price_history)}")

        try:
            avg_price = data["total_cost"] / data["quantity"] if data["quantity"] else 0
            market_value = data["quantity"] * current_price
            gain_loss = market_value - data["total_cost"]
            gain_loss_pct = (gain_loss / data["total_cost"] * 100) if data["total_cost"] > 0 else 0

            cashflows = []
            for o in orders_by_symbol.get(symbol, []):
                cf_amount = -(o.quantity * o.price + o.commission) if o.order_type == "buy" else (o.quantity * o.price - o.commission)
                cashflows.append((o.date, cf_amount))
            if data["quantity"] > 0:
                cashflows.append((date.today(), market_value))
            cashflows.sort(key=lambda x: x[0])
            try:
                xirr = calc_xirr(cashflows)
            except Exception as e:
                print(f"XIRR calc failed for {symbol}: {e}")
                xirr = 0.0

            currency = ""
            orders_for_symbol = orders_by_symbol.get(symbol, [])
            if orders_for_symbol:
                currency = orders_for_symbol[0].currency or ""

            # Converti alla valuta di riferimento per i totali
            market_value_ref = convert_to_reference_currency(market_value, currency, reference_currency, db)
            cost_basis_ref = convert_to_reference_currency(data["total_cost"], currency, reference_currency, db)

            positions.append(
                {
                    "symbol": symbol,
                    "quantity": round(data["quantity"], 4),
                    "avg_price": round(avg_price, 2),
                    "current_price": round(current_price, 2),
                    "market_value": round(market_value, 2),
                    "cost_basis": round(data["total_cost"], 2),
                    "gain_loss": round(gain_loss, 2),
                    "gain_loss_pct": round(gain_loss_pct, 2),
                    "instrument_type": instrument_type,
                    "currency": currency,
                    "xirr": round(xirr * 100, 2) if xirr else 0.0,
                    "fetch_error": fetch_error,
                }
            )
            if include_history and price_history:
                position_histories[symbol] = price_history

            # Usa valori convertiti per i totali
            total_value += market_value_ref
            total_cost += cost_basis_ref
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0

    return positions, total_value, total_cost, total_gain_loss, total_gain_loss_pct, position_histories


def aggregate_portfolio_history(position_histories: dict, orders_by_symbol: dict):
    """
    Aggrega il valore del portafoglio nel tempo usando solo le date in cui
    tutti gli strumenti con posizione > 0 hanno un prezzo disponibile.
    """
    if not position_histories:
        return [], {}, []

    price_map = {}
    all_dates = set()
    for symbol, history in position_histories.items():
        m = {}
        for point in history:
            d = parse_date_input(point.get("date"))
            if not d:
                continue
            try:
                m[d] = float(point.get("price"))
                all_dates.add(d)
            except Exception:
                continue
        price_map[symbol] = m

    if not all_dates:
        return [], price_map, []

    orders_sorted = {sym: sorted(orders_by_symbol.get(sym, []), key=lambda o: o.date) for sym in price_map.keys()}
    order_idx = {sym: 0 for sym in price_map.keys()}
    qty_map = {sym: 0.0 for sym in price_map.keys()}

    aggregated = []
    valid_dates = []
    for current_date in sorted(all_dates):
        total_value = 0.0
        any_position = False
        valid = True

        # applica ordini fino a questa data (inclusa)
        for sym, ords in orders_sorted.items():
            idx = order_idx[sym]
            while idx < len(ords) and ords[idx].date <= current_date:
                delta = ords[idx].quantity if ords[idx].order_type == "buy" else -ords[idx].quantity
                qty_map[sym] += delta
                idx += 1
            order_idx[sym] = idx

        for sym, prices in price_map.items():
            qty = qty_map.get(sym, 0.0)
            if qty <= 0:
                continue
            any_position = True
            price = prices.get(current_date)
            if price is None:
                valid = False
                break
            total_value += qty * price

        if valid and any_position:
            aggregated.append({"date": current_date.strftime(DATE_FMT), "value": round(total_value, 2)})
            valid_dates.append(current_date)

    return aggregated, price_map, valid_dates


def compute_portfolio_performance(price_map: dict, orders_by_symbol: dict, common_dates: list):
    """
    Calcola una curva di performance (TWR-like) basata sui rendimenti dei singoli asset
    pesati per il valore del giorno precedente, escludendo l'effetto dei flussi (ordini).
    """
    if not price_map or not common_dates:
        return []

    # ordini per simbolo ordinati per data
    orders_sorted = {sym: sorted(orders_by_symbol.get(sym, []), key=lambda o: o.date) for sym in price_map.keys()}
    order_idx = {sym: 0 for sym in price_map.keys()}
    qty_map = {sym: 0.0 for sym in price_map.keys()}

    perf = []
    nav = 100.0
    prev_date = None

    for current_date in common_dates:
        if prev_date is not None:
            total_prev_value = 0.0
            for sym, prices in price_map.items():
                prev_price = prices.get(prev_date)
                if prev_price and qty_map.get(sym, 0.0) > 0:
                    total_prev_value += qty_map[sym] * prev_price

            if total_prev_value > 0:
                daily_return = 0.0
                for sym, prices in price_map.items():
                    prev_price = prices.get(prev_date)
                    curr_price = prices.get(current_date)
                    qty = qty_map.get(sym, 0.0)
                    if qty > 0 and prev_price and curr_price:
                        weight = (qty * prev_price) / total_prev_value
                        asset_ret = (curr_price / prev_price) - 1
                        daily_return += weight * asset_ret
                nav *= (1 + daily_return)
                perf.append({"date": current_date.strftime(DATE_FMT), "value": round(nav, 4)})
        else:
            perf.append({"date": current_date.strftime(DATE_FMT), "value": round(nav, 4)})

        # applica ordini alla fine del giorno per la giornata successiva
        for sym, ords in orders_sorted.items():
            idx = order_idx[sym]
            while idx < len(ords) and ords[idx].date == current_date:
                delta = ords[idx].quantity if ords[idx].order_type == "buy" else -ords[idx].quantity
                qty_map[sym] = qty_map.get(sym, 0.0) + delta
                idx += 1
            order_idx[sym] = idx

        prev_date = current_date

    return perf


def validate_order_input(order: Order):
    if order.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    if order.price <= 0:
        raise HTTPException(status_code=400, detail="Price must be positive")
    if order.order_type not in {"buy", "sell"}:
        raise HTTPException(status_code=400, detail="order_type must be 'buy' or 'sell'")
    if order.commission < 0:
        raise HTTPException(status_code=400, detail="Commission cannot be negative")
    if order.instrument_type.lower() not in {"stock", "etf"}:
        raise HTTPException(status_code=400, detail="instrument_type must be one of: stock, etf")


# Auth endpoints - NOW IN routers/auth.py
# @app.post("/auth/register", response_model=Token)
# def register(user: UserRegister, db: Session = Depends(get_db)):
#     hashed_password = get_password_hash(user.password)
#     new_user = UserModel(email=user.email.lower(), username=user.username, hashed_password=hashed_password)
#     db.add(new_user)
#     try:
#         db.commit()
#     except IntegrityError:
#         db.rollback()
#         raise HTTPException(status_code=400, detail="Email or username already registered")
#     db.refresh(new_user)

#     access_token = create_access_token(data={"sub": new_user.email})
#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "user": {"email": new_user.email, "username": new_user.username},
#     }

# @app.post("/auth/login", response_model=Token)
# def login(user: UserLogin, db: Session = Depends(get_db)):
#     db_user = db.execute(select(UserModel).where(UserModel.email == user.email.lower())).scalar_one_or_none()
#     if not db_user or not verify_password(user.password, db_user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect email or password")

#     access_token = create_access_token(data={"sub": db_user.email})
#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "user": {"email": db_user.email, "username": db_user.username},
#     }

# @app.get("/auth/me")
# def get_current_user(user: UserModel = Depends(verify_token)):
#     return {"email": user.email, "username": user.username}


# Portfolio endpoints - NOW IN routers/portfolios.py
# @app.post("/portfolios")
# @app.get("/portfolios/count")
# @app.get("/portfolios")
# @app.get("/portfolios/{portfolio_id}")
# @app.put("/portfolios/{portfolio_id}")
# @app.delete("/portfolios/{portfolio_id}")
# All portfolio endpoints have been moved to routers/portfolios.py


# Order endpoints - NOW IN routers/orders.py
# @app.post("/orders")
# @app.get("/orders/{portfolio_id}")
# @app.delete("/orders/{order_id}")
# @app.put("/orders/{order_id}")
# @app.post("/portfolio/optimize")
# All order and optimization endpoints have been moved to routers/orders.py

# (Old order endpoints commented out below - to be removed later)
"""
@app.post("/orders")
def create_order_OLD(order: Order, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    validate_order_input(order)
    try:
        match = ensure_symbol_exists(order.symbol, order.instrument_type)
    except HTTPException:
        # bubble up validation errors, but allow missing key error through HTTPException
        raise
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == order.portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    existing_orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
    current_positions = aggregate_positions(existing_orders)
    symbol_positions = current_positions.get(order.symbol.upper(), {"quantity": 0, "total_cost": 0})

    if order.order_type == "sell" and order.quantity > symbol_positions["quantity"]:
        raise HTTPException(status_code=400, detail="Cannot sell more than current position")

    resolved_isin = order.isin or (match.get("isin") if match else "")
    resolved_ter = order.ter or (match.get("ter") if match else "")

    db_order = OrderModel(
        portfolio_id=portfolio.id,
        symbol=order.symbol.upper(),
        isin=resolved_isin,
        ter=resolved_ter,
        name=order.name or (match.get("name") if match else ""),
        exchange=order.exchange or (match.get("exchange") if match else ""),
        currency=order.currency or (match.get("currency") if match else ""),
        quantity=order.quantity,
        price=order.price,
        commission=order.commission,
        instrument_type=order.instrument_type.lower(),
        order_type=order.order_type,
        date=order.date,
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)

    return {
        "message": "Order created",
        "order": {
            "id": db_order.id,
            "portfolio_id": db_order.portfolio_id,
            "symbol": db_order.symbol,
            "isin": db_order.isin,
            "ter": db_order.ter,
            "name": db_order.name,
            "exchange": db_order.exchange,
            "currency": db_order.currency,
            "quantity": db_order.quantity,
            "price": db_order.price,
            "commission": db_order.commission,
            "instrument_type": db_order.instrument_type,
            "order_type": db_order.order_type,
            "date": format_date(db_order.date),
            "created_at": format_datetime(db_order.created_at),
        },
    }

@app.get("/orders/{portfolio_id}")
def get_orders(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
    return {
        "orders": [
            {
                "id": o.id,
                "portfolio_id": o.portfolio_id,
                "symbol": o.symbol,
                "isin": o.isin,
                "ter": o.ter,
                "name": o.name,
                "exchange": o.exchange,
                "currency": o.currency,
                "quantity": o.quantity,
                "price": o.price,
                "commission": o.commission,
                "instrument_type": o.instrument_type,
                "order_type": o.order_type,
                "date": format_date(o.date),
                "created_at": format_datetime(o.created_at),
            }
            for o in orders
        ]
    }

@app.delete("/orders/{order_id}")
def delete_order(order_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    order = db.execute(
        select(OrderModel).join(PortfolioModel).where(OrderModel.id == order_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    db.delete(order)
    db.commit()
    return {"message": "Order deleted"}

@app.put("/orders/{order_id}")
def update_order(order_id: int, updated: Order, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    validate_order_input(updated)
    try:
        match = ensure_symbol_exists(updated.symbol, updated.instrument_type)
    except HTTPException:
        raise

    order = db.execute(
        select(OrderModel).join(PortfolioModel).where(OrderModel.id == order_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    # Recompute positions excluding this order to validate sells
    existing_orders = (
        db.execute(
            select(OrderModel).where(OrderModel.portfolio_id == order.portfolio_id, OrderModel.id != order.id)
        ).scalars().all()
    )
    positions_before = aggregate_positions(existing_orders)
    symbol_positions = positions_before.get(updated.symbol.upper(), {"quantity": 0, "total_cost": 0})
    if updated.order_type == "sell" and updated.quantity > symbol_positions["quantity"]:
        raise HTTPException(status_code=400, detail="Cannot sell more than current position")

    resolved_isin = updated.isin or (match.get("isin") if match else "")
    resolved_ter = updated.ter or (match.get("ter") if match else "")

    order.symbol = updated.symbol.upper()
    order.isin = resolved_isin
    order.ter = resolved_ter
    order.name = updated.name or (match.get("name") if match else "")
    order.exchange = updated.exchange or (match.get("exchange") if match else "")
    order.currency = updated.currency or (match.get("currency") if match else "")
    order.quantity = updated.quantity
    order.price = updated.price
    order.commission = updated.commission
    order.instrument_type = updated.instrument_type.lower()
    order.order_type = updated.order_type
    order.date = updated.date

    db.commit()
    db.refresh(order)

    return {
        "message": "Order updated",
        "order": {
            "id": order.id,
            "portfolio_id": order.portfolio_id,
            "symbol": order.symbol,
            "isin": order.isin,
            "ter": order.ter,
            "name": order.name,
            "exchange": order.exchange,
            "currency": order.currency,
            "quantity": order.quantity,
            "price": order.price,
            "commission": order.commission,
            "order_type": order.order_type,
            "date": format_date(order.date),
            "created_at": format_datetime(order.created_at),
        },
    }


# Optimization endpoint
@app.post("/portfolio/optimize")
def optimize_portfolio(request: OptimizationRequest, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == request.portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    try:
        symbols = [s.upper() for s in request.symbols]
        data = yf.download(symbols, period="1y", progress=False)["Adj Close"]

        if data.empty:
            raise HTTPException(status_code=400, detail="No data available")

        # Get dynamic risk-free rate based on portfolio settings
        reference_currency = portfolio.reference_currency or "EUR"
        risk_free_rate_pct = get_risk_free_rate(reference_currency, db, portfolio.risk_free_source)
        risk_free_rate = risk_free_rate_pct / 100.0  # Convert from percentage to decimal

        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)

        if request.optimization_type == "max_sharpe":
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif request.optimization_type == "min_volatility":
            ef.min_volatility()
        else:
            ef.efficient_risk(target_volatility=0.15)

        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

        latest_prices = get_latest_prices(data)
        orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
        positions_map = aggregate_positions(orders)
        total_value = sum(
            pos["quantity"] * latest_prices.get(symbol, 0)
            for symbol, pos in positions_map.items()
            if pos["quantity"] > 0 and symbol in latest_prices
        )

        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_value)
        allocation, leftover = da.greedy_portfolio()

        return {
            "weights": {k: round(v, 4) for k, v in cleaned_weights.items() if v > 0.001},
            "allocation": allocation,
            "leftover": round(leftover, 2),
            "expected_return": round(perf[0] * 100, 2),
            "volatility": round(perf[1] * 100, 2),
            "sharpe_ratio": round(perf[2], 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
# End of old order/optimization endpoints - all now in routers/orders.py


@app.get("/market-data/{symbol}")
def get_market_data(symbol: str):
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


@app.get("/portfolio/history/{portfolio_id}/{symbol}")
def get_position_history(
    portfolio_id: int, symbol: str, period: str = "1y", user: UserModel = Depends(verify_token), db: Session = Depends(get_db)
):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        history = []
        for idx, row in hist.iterrows():
            history.append({"date": idx.strftime(DATE_FMT), "close": round(row["Close"], 2), "volume": int(row["Volume"])})

        return {"symbol": symbol, "history": history}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/symbols/search")
def symbols_search(q: str, instrument_type: str = "stock"):
    matches = search_symbol(q, instrument_type)
    formatted = [
        {
            "symbol": item.get("symbol"),
            "name": item.get("name"),
            "exchange": item.get("exchangeShortName") or item.get("exchange"),
            "currency": item.get("currency"),
            "type": item.get("type"),
            "ter": item.get("ter", ""),
        }
        for item in matches
    ]
    return {"results": formatted}


@app.get("/symbols/ucits")
def symbols_ucits():
    formatted = []
    for item in ETF_UCITS_CACHE:
        sym = (item.get("ticker") or item.get("symbol") or item.get("isin") or "").upper()
        formatted.append({
            "symbol": sym,
            "name": item.get("name", ""),
            "exchange": item.get("exchange", "") or item.get("domicile", ""),
            "currency": item.get("currency", ""),
            "type": "ETF",
            "isin": item.get("isin"),
            "ticker": item.get("ticker"),
            "ter": item.get("ter", ""),
        })
    return {"results": formatted}

@app.get("/symbols/etf-list")
def get_etf_list(etf_type: str = "all"):
    """
    Restituisce la lista completa degli ETF nella cache locale.
    
    Args:
        etf_type: "us", "ucits", o "all" (default)
    
    Questo endpoint non richiede chiamate API esterne.
    """
    results = []
    
    if etf_type in ["us", "all"]:
        from etf_cache import get_all_etfs
        us_etfs = get_all_etfs()
        # Aggiungi tag per identificare il tipo
        for etf in us_etfs:
            etf["region"] = "US"
        results.extend(us_etfs)
    
    if etf_type in ["ucits", "all"]:
        try:
            from etf_cache_ucits import get_all_ucits_etfs
            ucits_etfs = get_all_ucits_etfs()
            # Aggiungi tag per identificare il tipo
            for etf in ucits_etfs:
                etf["region"] = "UCITS"
            results.extend(ucits_etfs)
        except ImportError:
            pass
    
    return {
        "etfs": results,
        "count": len(results)
    }

@app.get("/symbols/etf-stats")
def get_etf_stats():
    """
    Restituisce statistiche sulla cache degli ETF
    """
    stats = {
        "us_etfs": 0,
        "ucits_etfs": 0,
        "currencies": {},
        "domiciles": {}
    }
    
    try:
        from etf_cache import get_all_etfs
        us_etfs = get_all_etfs()
        stats["us_etfs"] = len(us_etfs)
        
        for etf in us_etfs:
            curr = etf.get("currency", "Unknown")
            stats["currencies"][curr] = stats["currencies"].get(curr, 0) + 1
    except ImportError:
        pass
    
    try:
        from etf_cache_ucits import get_all_ucits_etfs
        ucits_etfs = get_all_ucits_etfs()
        stats["ucits_etfs"] = len(ucits_etfs)
        
        for etf in ucits_etfs:
            curr = etf.get("currency", "Unknown")
            stats["currencies"][curr] = stats["currencies"].get(curr, 0) + 1
            
            dom = etf.get("domicile", "Unknown")
            stats["domiciles"][dom] = stats["domiciles"].get(dom, 0) + 1
    except ImportError:
        pass
    
    stats["total_etfs"] = stats["us_etfs"] + stats["ucits_etfs"]
    
    return stats


# Analysis endpoint
@app.get("/analysis/{portfolio_id}")
def analyze_portfolio(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    """
    Advanced portfolio analysis including:
    - Correlation matrix
    - Monte Carlo simulation
    - Risk metrics (VaR, CVaR, Sharpe, Sortino, etc.)
    - Drawdown analysis
    """
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
    if not orders:
        return {
            "message": "No orders in portfolio",
            "correlation": None,
            "montecarlo": None,
            "risk_metrics": None,
            "drawdown": None
        }

    # Get positions and historical data from cache
    positions_map = aggregate_positions(orders)
    orders_by_symbol = {}
    symbol_type_map = {}
    symbol_isin_map = {}
    for o in orders:
        orders_by_symbol.setdefault(o.symbol.upper(), []).append(o)
        symbol_type_map[o.symbol.upper()] = (o.instrument_type or "stock").lower()
        symbol_isin_map[o.symbol.upper()] = (o.isin or "").upper()

    reference_currency = portfolio.reference_currency or "EUR"
    positions, total_value, total_cost, _, _, position_histories = compute_portfolio_value(
        positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db,
        include_history=True, reference_currency=reference_currency
    )

    if len(positions) < 2:
        return {
            "message": "Need at least 2 assets for analysis",
            "correlation": None,
            "montecarlo": None,
            "risk_metrics": None,
            "drawdown": None
        }

    try:
        # Use cached historical data instead of downloading again
        if not position_histories:
            return {
                "message": "Insufficient historical data",
                "correlation": None,
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Build a unified DataFrame from cached price histories
        # Find common dates across all symbols
        all_dates_sets = []
        for symbol, history in position_histories.items():
            dates = set()
            for point in history:
                d = parse_date_input(point.get("date"))
                if d:
                    dates.add(d)
            all_dates_sets.append(dates)

        if not all_dates_sets:
            return {
                "message": "No historical data available",
                "correlation": None,
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Get common dates (intersection)
        common_dates = set.intersection(*all_dates_sets) if len(all_dates_sets) > 1 else all_dates_sets[0]

        if len(common_dates) < 30:
            return {
                "message": "Insufficient overlapping historical data (need at least 30 days)",
                "correlation": None,
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Sort dates
        sorted_dates = sorted(list(common_dates))

        # Build price DataFrame
        price_data = {}
        for symbol, history in position_histories.items():
            prices_map = {}
            for point in history:
                d = parse_date_input(point.get("date"))
                if d and d in common_dates:
                    try:
                        prices_map[d] = float(point.get("price", 0))
                    except:
                        pass
            price_data[symbol] = prices_map

        # Create DataFrame
        data_dict = {}
        for symbol in price_data:
            data_dict[symbol] = [price_data[symbol].get(d, np.nan) for d in sorted_dates]

        # Convert sorted_dates to pandas datetime for proper alignment
        data = pd.DataFrame(data_dict, index=pd.to_datetime(sorted_dates))
        data = data.dropna()

        if len(data) < 30:
            return {
                "message": "Insufficient clean historical data after removing gaps",
                "correlation": None,
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Calculate returns
        returns = data.pct_change(fill_method=None).dropna()

        # 1. Correlation Matrix
        corr_matrix = returns.corr().values.tolist()

        # Calculate portfolio weights
        weights = np.array([p["market_value"] / total_value if total_value > 0 else 0 for p in positions])

        # Ensure weights match data columns
        position_symbols = [p["symbol"] for p in positions]
        ordered_weights = []
        for sym in data.columns:
            if sym in position_symbols:
                idx = position_symbols.index(sym)
                ordered_weights.append(weights[idx])
            else:
                ordered_weights.append(0.0)
        weights = np.array(ordered_weights)
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        if len(portfolio_returns) == 0:
            return {
                "message": "No valid portfolio returns calculated",
                "correlation": {
                    "symbols": list(data.columns),
                    "matrix": corr_matrix
                },
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Annual return and volatility
        annual_return = portfolio_returns.mean() * 252 * 100
        annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100

        # Get dynamic risk-free rate based on portfolio settings
        risk_free_rate_pct = get_risk_free_rate(reference_currency, db, portfolio.risk_free_source)
        risk_free_rate = risk_free_rate_pct / 100.0  # Convert from percentage to decimal

        # Sharpe Ratio (using dynamic risk-free rate)
        sharpe_ratio = (annual_return / 100 - risk_free_rate) / (annual_volatility / 100) if annual_volatility > 0 else 0

        # Sortino Ratio (downside deviation only)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino_ratio = (annual_return / 100 - risk_free_rate) / downside_std if downside_std > 0 else 0

        # VaR and CVaR (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5) * 100
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100

        # Max Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Calmar Ratio (Annual Return / Max Drawdown)
        # Use XIRR for more accurate annual return calculation
        calmar_ratio = 0.0
        try:
            from datetime import date as date_class
            # Calculate XIRR
            cashflows = []

            # Cache exchange rates to avoid repeated lookups
            exchange_rate_cache = {}

            for o in orders:
                order_currency = o.currency or reference_currency

                # Get exchange rate (cached)
                if order_currency == reference_currency:
                    rate = 1.0
                elif order_currency not in exchange_rate_cache:
                    # Only fetch once per currency
                    exchange_rate_cache[order_currency] = convert_to_reference_currency(1.0, order_currency, reference_currency, db)
                    rate = exchange_rate_cache[order_currency]
                else:
                    rate = exchange_rate_cache[order_currency]

                if o.order_type == "buy":
                    cf_amount = -(o.quantity * o.price + o.commission) * rate
                else:
                    cf_amount = (o.quantity * o.price - o.commission) * rate

                cashflows.append((o.date, cf_amount))

            if total_value > 0:
                cashflows.append((date_class.today(), total_value))

            if cashflows and max_drawdown < 0:
                cashflows.sort(key=lambda x: x[0])
                portfolio_xirr = calc_xirr(cashflows) * 100
                calmar_ratio = portfolio_xirr / abs(max_drawdown)
        except Exception as e:
            print(f"[CALMAR] Error calculating Calmar Ratio: {e}")
            import traceback
            traceback.print_exc()
            calmar_ratio = 0.0

        # Beta (vs market benchmark - dynamic based on portfolio settings)
        try:
            # Get market benchmark data based on portfolio settings
            benchmark_data = get_market_benchmark_data(reference_currency, db, portfolio.market_benchmark)
            benchmark_history = benchmark_data.get("history", [])
            benchmark_symbol = benchmark_data.get("symbol", "Market")

            print(f"[BETA] Benchmark: {benchmark_symbol}, history length: {len(benchmark_history) if benchmark_history else 0}")

            if benchmark_history and len(benchmark_history) > 30:
                # Build benchmark price series
                benchmark_dates = []
                benchmark_prices = []
                for point in benchmark_history:
                    d = parse_date_input(point.get("date"))
                    if d:
                        benchmark_dates.append(d)
                        benchmark_prices.append(float(point.get("price", 0)))

                print(f"[BETA] Parsed {len(benchmark_dates)} benchmark dates")

                # Create pandas series
                benchmark_series = pd.Series(benchmark_prices, index=pd.to_datetime(benchmark_dates))
                market_returns = benchmark_series.pct_change(fill_method=None).dropna()

                print(f"[BETA] Portfolio returns: {len(portfolio_returns)}, Market returns: {len(market_returns)}")

                # Align dates
                common_dates_beta = portfolio_returns.index.intersection(market_returns.index)
                print(f"[BETA] Common dates for beta: {len(common_dates_beta)}")

                if len(common_dates_beta) > 30:
                    aligned_portfolio = portfolio_returns.loc[common_dates_beta]
                    aligned_market = market_returns.loc[common_dates_beta]

                    covariance = np.cov(aligned_portfolio, aligned_market)[0][1]
                    market_variance = np.var(aligned_market)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                    print(f"[BETA] Calculated vs {benchmark_symbol}: {beta:.3f} (cov={covariance:.6f}, var={market_variance:.6f})")
                else:
                    print(f"[BETA] Not enough common dates ({len(common_dates_beta)}), defaulting to 1.0")
                    beta = 1.0
            else:
                print(f"[BETA] No benchmark history or too short, defaulting to 1.0")
                beta = 1.0
        except Exception as e:
            print(f"[BETA] Error calculating beta: {e}")
            import traceback
            traceback.print_exc()
            beta = 1.0

        # Nomi leggibili per benchmark
        benchmark_name_map = {
            "^GSPC": "S&P 500",
            "VWCE.DE": "VWCE"
        }
        benchmark_display_name = benchmark_name_map.get(benchmark_symbol, benchmark_symbol)

        risk_metrics = {
            "sharpe_ratio": round(sharpe_ratio, 3),
            "sortino_ratio": round(sortino_ratio, 3),
            "var_95": round(var_95, 3),
            "cvar_95": round(cvar_95, 3),
            "calmar_ratio": round(calmar_ratio, 2),
            "volatility": round(annual_volatility, 2),
            "max_drawdown": round(max_drawdown, 2),
            "beta": round(beta, 3),
            "beta_vs": benchmark_display_name,  # Nome del benchmark usato per Beta
            "risk_free_rate": round(risk_free_rate_pct, 2)  # Tasso risk-free usato nei calcoli
        }

        # 3. Monte Carlo Simulation (1 year forward, 10,000 simulations)
        num_simulations = 10000
        num_days = 252  # 1 year

        # Use historical mean and covariance
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        # Generate random returns
        simulations = np.zeros((num_days, num_simulations))

        for i in range(num_simulations):
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
            portfolio_sim_returns = (random_returns * weights).sum(axis=1)
            simulations[:, i] = total_value * (1 + portfolio_sim_returns).cumprod()

        # Calculate percentiles
        dates_future = pd.date_range(start=pd.Timestamp.today(), periods=num_days, freq='B')
        percentiles = {
            "p5": np.percentile(simulations, 5, axis=1).tolist(),
            "p25": np.percentile(simulations, 25, axis=1).tolist(),
            "p50": np.percentile(simulations, 50, axis=1).tolist(),
            "p75": np.percentile(simulations, 75, axis=1).tolist(),
            "p95": np.percentile(simulations, 95, axis=1).tolist(),
        }

        montecarlo = {
            "dates": [d.strftime(DATE_FMT) for d in dates_future],
            "percentiles": percentiles,
            "current_value": round(total_value, 2)
        }

        # 4. Drawdown History
        portfolio_history, price_map, common_dates = aggregate_portfolio_history(position_histories, orders_by_symbol)

        if portfolio_history and len(portfolio_history) > 1:
            values = [p["value"] for p in portfolio_history]
            dates_hist = [p["date"] for p in portfolio_history]

            cumulative_max = np.maximum.accumulate(values)
            drawdown_series = [(val - cmax) / cmax * 100 if cmax > 0 else 0
                             for val, cmax in zip(values, cumulative_max)]

            drawdown_data = {
                "dates": dates_hist,
                "drawdown": drawdown_series
            }
        else:
            drawdown_data = None

        # 5. Performance Attribution
        # Calculate individual asset performance and contribution to portfolio return
        attribution_data = []

        for position in positions:
            symbol = position["symbol"]
            current_value = position["market_value"]
            cost_basis = position["cost_basis"]
            weight = current_value / total_value if total_value > 0 else 0

            # Individual asset return
            asset_return_pct = ((current_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0

            # Contribution to total portfolio return (weight * return)
            contribution_pct = weight * asset_return_pct

            # Get asset price history for period return calculation
            asset_history = position_histories.get(symbol, [])
            period_return_pct = 0.0
            if len(asset_history) >= 2:
                # Calculate return from first to last historical price
                first_price = None
                last_price = None
                for point in asset_history:
                    price = point.get("price")
                    if price and first_price is None:
                        first_price = float(price)
                    if price:
                        last_price = float(price)

                if first_price and last_price and first_price > 0:
                    period_return_pct = ((last_price - first_price) / first_price) * 100

            attribution_data.append({
                "symbol": symbol,
                "name": position.get("name", symbol),
                "weight": round(weight * 100, 2),  # as percentage
                "current_value": round(current_value, 2),
                "cost_basis": round(cost_basis, 2),
                "total_return_pct": round(asset_return_pct, 2),
                "period_return_pct": round(period_return_pct, 2),
                "contribution_to_portfolio": round(contribution_pct, 2),
                "gain_loss": round(current_value - cost_basis, 2)
            })

        # Sort by contribution (descending)
        attribution_data.sort(key=lambda x: x["contribution_to_portfolio"], reverse=True)

        # Calculate portfolio total return for comparison
        portfolio_total_return = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0

        # Build time series for each asset (normalized to start at 100)
        asset_time_series = {}
        all_dates_list = []

        for symbol in position_histories.keys():
            history = position_histories[symbol]
            if history and len(history) > 0:
                date_price_pairs = []
                for point in history:
                    d = parse_date_input(point.get("date"))
                    if d:
                        price = float(point.get("price", 0))
                        date_price_pairs.append((d, price))
                        if d not in [dp[0] for dp in all_dates_list]:
                            all_dates_list.append((d, None))

                # Sort by date
                date_price_pairs.sort(key=lambda x: x[0])

                # Normalize to 100 at start
                if date_price_pairs and date_price_pairs[0][1] > 0:
                    first_price = date_price_pairs[0][1]
                    asset_time_series[symbol] = {
                        d: (price / first_price) * 100
                        for d, price in date_price_pairs
                    }

        # Create aligned time series data with forward-fill for missing values
        # Sort dates properly as date objects, not strings
        sorted_dates = sorted(set([d for d, _ in all_dates_list]))
        aligned_series = []
        last_values = {}  # Track last known value for each symbol

        for date_obj in sorted_dates:
            date_str = date_obj.strftime(DATE_FMT)
            point = {"date": date_str}

            for symbol, date_value_map in asset_time_series.items():
                if date_obj in date_value_map:
                    value = round(date_value_map[date_obj], 2)
                    point[symbol] = value
                    last_values[symbol] = value
                elif symbol in last_values:
                    # Forward-fill: use last known value
                    point[symbol] = last_values[symbol]
                # else: no value yet for this symbol, leave it out (will be null in chart)
            aligned_series.append(point)

        performance_attribution = {
            "assets": attribution_data,
            "portfolio_total_return": round(portfolio_total_return, 2),
            "top_contributors": attribution_data[:3] if len(attribution_data) >= 3 else attribution_data,
            "bottom_contributors": list(reversed(attribution_data[-3:])) if len(attribution_data) >= 3 else [],
            "time_series": aligned_series,
            "symbols": list(position_histories.keys())
        }

        return {
            "correlation": {
                "symbols": list(data.columns),
                "matrix": corr_matrix
            },
            "risk_metrics": risk_metrics,
            "montecarlo": montecarlo,
            "drawdown": drawdown_data,
            "performance_attribution": performance_attribution
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/market-data/risk-free-rate/{currency}")
def get_risk_free_rate_endpoint(currency: str, db: Session = Depends(get_db)):
    """
    Endpoint per ottenere il tasso risk-free per una valuta.
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


@app.get("/market-data/benchmark/{currency}")
def get_benchmark_endpoint(currency: str, db: Session = Depends(get_db)):
    """
    Endpoint per ottenere i dati del benchmark di mercato per una valuta.
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


@app.get("/market-data/portfolio-context/{portfolio_id}")
def get_portfolio_market_context(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    """
    Endpoint per ottenere risk-free rate e benchmark appropriati per un portfolio
    basato sulla sua reference_currency.

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

        # Nomi descrittivi
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
