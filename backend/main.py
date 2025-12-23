import os
import json
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Optional

import jwt
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import requests
from sqlalchemy import (Column, Date, DateTime, Float, ForeignKey, Integer,
                        String, create_engine, select, text)
from sqlalchemy.exc import IntegrityError
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
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

FMP_API_KEY = os.environ.get("FMP_API_KEY")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
AV_BASE = "https://www.alphavantage.co/query"
SYMBOL_CACHE = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Database models
class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    portfolios = relationship("PortfolioModel", back_populates="user", cascade="all, delete")


class PortfolioModel(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    initial_capital = Column(Float, default=0.0)  # kept for schema compatibility, not used in logic
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    user = relationship("UserModel", back_populates="portfolios")
    orders = relationship("OrderModel", back_populates="portfolio", cascade="all, delete")


class OrderModel(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False)
    isin = Column(String, default="")
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


class ETFPriceCacheModel(Base):
    __tablename__ = "etf_price_cache"
    isin = Column(String, primary_key=True, index=True)
    last_price = Column(Float, default=0.0)
    currency = Column(String, default="")
    history_json = Column(String, default="")
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


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


ensure_orders_table_has_isin()


ETF_CACHE_MAX_AGE_HOURS = 6


def normalize_chart_history(df: pd.DataFrame):
    """
    Converte il DataFrame restituito da justetf_scraping.load_chart in una lista di dict {date, price}.
    È tollerante ai nomi delle colonne (price/close ecc).
    """
    if df is None or df.empty:
        return []

    cols_lower = {str(c).lower(): c for c in df.columns}
    date_col = cols_lower.get("date") or cols_lower.get("time") or list(df.columns)[0]
    price_col = cols_lower.get("price") or cols_lower.get("close") or cols_lower.get("nav") or list(df.columns)[-1]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    history = []
    for _, row in df.iterrows():
        try:
            ts = row[date_col]
            price_val = float(row[price_col])
            history.append({"date": ts.strftime("%Y-%m-%d"), "price": price_val})
        except Exception:
            continue
    history = sorted(history, key=lambda x: x["date"])
    return history


def get_etf_price_and_history(isin: str, db: Session):
    """
    Restituisce prezzo corrente e storico per ETF tramite justetf_scraping con cache su SQLite.
    """
    if not isin:
        raise HTTPException(status_code=400, detail="Missing ISIN for ETF price lookup")

    now = datetime.utcnow()
    cache = db.execute(select(ETFPriceCacheModel).where(ETFPriceCacheModel.isin == isin)).scalar_one_or_none()
    if cache and cache.updated_at and (now - cache.updated_at) < timedelta(hours=ETF_CACHE_MAX_AGE_HOURS):
        history = json.loads(cache.history_json or "[]")
        last_price = cache.last_price or (history[-1]["price"] if history else 0.0)
        return {"last_price": last_price, "history": history, "currency": cache.currency}

    try:
        import justetf_scraping
    except ImportError:
        raise HTTPException(status_code=500, detail="justetf_scraping non installato (pip install git+https://github.com/druzsan/justetf-scraping.git)")

    try:
        df_chart = justetf_scraping.load_chart(isin)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore nel recupero prezzi justETF: {e}")

    history = normalize_chart_history(df_chart)
    if not history:
        raise HTTPException(status_code=400, detail="Nessun dato storico disponibile per l'ISIN richiesto")

    last_price = history[-1]["price"]
    currency = ""

    if cache:
        cache.last_price = last_price
        cache.currency = currency
        cache.history_json = json.dumps(history)
        cache.updated_at = now
    else:
        cache = ETFPriceCacheModel(
            isin=isin,
            last_price=last_price,
            currency=currency,
            history_json=json.dumps(history),
            updated_at=now,
        )
        db.add(cache)
    db.commit()
    return {"last_price": last_price, "history": history, "currency": currency}


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


class Order(BaseModel):
    portfolio_id: int
    symbol: str
    name: Optional[str] = ""
    exchange: Optional[str] = ""
    currency: Optional[str] = ""
    isin: Optional[str] = ""
    quantity: float
    price: float
    commission: float = 0.0
    instrument_type: str = "stock"
    order_type: str
    date: date


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
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
    if rate <= -1:
        return float("inf")
    t0 = cashflows[0][0]
    return sum(cf / ((1 + rate) ** ((d - t0).days / 365.0)) for d, cf in cashflows)


def calc_xirr(cashflows: list):
    # cashflows: list of (date, amount)
    if not cashflows:
        return 0.0
    positive = any(cf > 0 for _, cf in cashflows)
    negative = any(cf < 0 for _, cf in cashflows)
    if not (positive and negative):
        return 0.0

    low, high = -0.9999, 10.0
    for _ in range(100):
        mid = (low + high) / 2
        npv = xnpv(mid, cashflows)
        if abs(npv) < 1e-6:
            return mid
        if npv > 0:
            low = mid
        else:
            high = mid
    return mid


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
):
    positions = []
    total_value = 0.0
    total_cost = 0.0
    position_histories = {}

    for symbol, data in positions_map.items():
        instrument_type = symbol_type_map.get(symbol, "stock")
        price_history = []
        current_price = 0.0
        try:
            if instrument_type == "etf":
                isin = symbol_isin_map.get(symbol, "")
                etf_data = get_etf_price_and_history(isin, db)
                current_price = etf_data["last_price"]
                price_history = etf_data.get("history", [])
            else:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                if include_history:
                    hist = ticker.history(period="6mo")
                    price_history = [
                        {"date": idx.strftime("%Y-%m-%d"), "price": float(row["Close"])}
                        for idx, row in hist.iterrows()
                    ]

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
            xirr = calc_xirr(cashflows)

            # pick currency from first order if available
            currency = ""
            orders_for_symbol = orders_by_symbol.get(symbol, [])
            if orders_for_symbol:
                currency = orders_for_symbol[0].currency or ""

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
                }
            )
            if include_history and price_history:
                position_histories[symbol] = price_history

            total_value += market_value
            total_cost += data["total_cost"]
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0

    return positions, total_value, total_cost, total_gain_loss, total_gain_loss_pct, position_histories


def aggregate_portfolio_history(position_histories: dict, positions_map: dict):
    """
    Aggrega le serie storiche dei singoli strumenti usando le quantità correnti per costruire
    una curva del valore di portafoglio.
    """
    if not position_histories:
        return []

    totals = {}
    for symbol, history in position_histories.items():
        qty = positions_map.get(symbol, {}).get("quantity", 0.0)
        for point in history:
            if qty <= 0:
                continue
            date_key = point["date"]
            totals[date_key] = totals.get(date_key, 0.0) + qty * float(point["price"])

    aggregated = [{"date": d, "value": round(v, 2)} for d, v in totals.items()]
    aggregated.sort(key=lambda x: x["date"])
    return aggregated


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


# Auth endpoints
@app.post("/auth/register", response_model=Token)
def register(user: UserRegister, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(user.password)
    new_user = UserModel(email=user.email.lower(), username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email or username already registered")
    db.refresh(new_user)

    access_token = create_access_token(data={"sub": new_user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"email": new_user.email, "username": new_user.username},
    }

@app.post("/auth/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.execute(select(UserModel).where(UserModel.email == user.email.lower())).scalar_one_or_none()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": db_user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"email": db_user.email, "username": db_user.username},
    }

@app.get("/auth/me")
def get_current_user(user: UserModel = Depends(verify_token)):
    return {"email": user.email, "username": user.username}


# Portfolio endpoints
@app.post("/portfolios")
def create_portfolio(portfolio: Portfolio, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    db_portfolio = PortfolioModel(
        user_id=user.id,
        name=portfolio.name,
        description=portfolio.description,
        initial_capital=portfolio.initial_capital,
    )
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return {
        "id": db_portfolio.id,
        "user_email": user.email,
        "name": db_portfolio.name,
        "description": db_portfolio.description,
        "initial_capital": db_portfolio.initial_capital,
        "created_at": db_portfolio.created_at.isoformat() if db_portfolio.created_at else None,
    }

@app.get("/portfolios")
def get_portfolios(user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    results = db.execute(select(PortfolioModel).where(PortfolioModel.user_id == user.id)).scalars().all()
    response = []
    for p in results:
        orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == p.id)).scalars().all()
        positions_map = aggregate_positions(orders)
        orders_by_symbol = {}
        symbol_type_map = {}
        symbol_isin_map = {}
        for o in orders:
            orders_by_symbol.setdefault(o.symbol.upper(), []).append(o)
            symbol_type_map[o.symbol.upper()] = (o.instrument_type or "stock").lower()
            symbol_isin_map[o.symbol.upper()] = (o.isin or "").upper()
        _, total_value, total_cost, total_gain_loss, total_gain_loss_pct, _ = compute_portfolio_value(
            positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db
        )
        response.append(
            {
                "id": p.id,
                "user_email": user.email,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "total_value": round(total_value, 2),
                "total_cost": round(total_cost, 2),
                "total_gain_loss": round(total_gain_loss, 2),
                "total_gain_loss_pct": round(total_gain_loss_pct, 2),
            }
        )
    return {
        "portfolios": response
    }

@app.get("/portfolios/{portfolio_id}")
def get_portfolio(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
    orders_by_symbol = {}
    symbol_type_map = {}
    symbol_isin_map = {}
    for o in orders:
        orders_by_symbol.setdefault(o.symbol.upper(), []).append(o)
        symbol_type_map[o.symbol.upper()] = (o.instrument_type or "stock").lower()
        symbol_isin_map[o.symbol.upper()] = (o.isin or "").upper()
    positions_map = aggregate_positions(orders)
    positions, total_value, total_cost, total_gain_loss, total_gain_loss_pct, position_histories = compute_portfolio_value(
        positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db, include_history=True
    )

    portfolio_history = aggregate_portfolio_history(position_histories, positions_map)

    return {
        "portfolio": {
            "id": portfolio.id,
            "user_email": user.email,
            "name": portfolio.name,
            "description": portfolio.description,
            "created_at": portfolio.created_at.isoformat() if portfolio.created_at else None,
        },
        "positions": positions,
        "summary": {
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain_loss": round(total_gain_loss, 2),
            "total_gain_loss_pct": round(total_gain_loss_pct, 2),
        },
        "history": {
            "portfolio": portfolio_history,
            "positions": position_histories,
        },
        "last_updated": datetime.now().isoformat(),
    }

@app.delete("/portfolios/{portfolio_id}")
def delete_portfolio(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    db.delete(portfolio)
    db.commit()
    return {"message": "Portfolio deleted"}


# Order endpoints
@app.post("/orders")
def create_order(order: Order, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
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

    db_order = OrderModel(
        portfolio_id=portfolio.id,
        symbol=order.symbol.upper(),
        isin=resolved_isin,
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
            "name": db_order.name,
            "exchange": db_order.exchange,
            "currency": db_order.currency,
            "quantity": db_order.quantity,
            "price": db_order.price,
            "commission": db_order.commission,
            "instrument_type": db_order.instrument_type,
            "order_type": db_order.order_type,
            "date": db_order.date.isoformat(),
            "created_at": db_order.created_at.isoformat() if db_order.created_at else None,
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
                "name": o.name,
                "exchange": o.exchange,
                "currency": o.currency,
                "quantity": o.quantity,
                "price": o.price,
                "commission": o.commission,
                "instrument_type": o.instrument_type,
                "order_type": o.order_type,
                "date": o.date.isoformat(),
                "created_at": o.created_at.isoformat() if o.created_at else None,
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

    order.symbol = updated.symbol.upper()
    order.isin = resolved_isin
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
            "name": order.name,
            "exchange": order.exchange,
            "currency": order.currency,
            "quantity": order.quantity,
            "price": order.price,
            "commission": order.commission,
            "order_type": order.order_type,
            "date": order.date.isoformat(),
            "created_at": order.created_at.isoformat() if order.created_at else None,
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

        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)

        if request.optimization_type == "max_sharpe":
            ef.max_sharpe()
        elif request.optimization_type == "min_volatility":
            ef.min_volatility()
        else:
            ef.efficient_risk(target_volatility=0.15)

        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)

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
            history.append({"date": idx.strftime("%Y-%m-%d"), "close": round(row["Close"], 2), "volume": int(row["Volume"])})

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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
