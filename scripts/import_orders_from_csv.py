"""
Utility script to import orders from `investmentOrders2025.csv` into the SQLite DB.

Mapping:
- Data -> date
- Ticker -> symbol
- Order -> order_type (buy/sell)
- Prezzo -> price
- Share -> quantity
- Conto -> portfolio name (creates it if missing)

Columns Tot/Flusso are ignored.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    select,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from etf_cache_ucits import ETF_UCITS_CACHE


CSV_DEFAULT = Path(__file__).with_name("investmentOrders2025.csv")
DEFAULT_USER_EMAIL = "luca.botta44@gmail.com"


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

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./portfolio.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class PortfolioModel(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    initial_capital = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


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
    order_type = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


Base.metadata.create_all(bind=engine)


def parse_number(value: str) -> float:
    """
    Parse euro-style numbers like "€ 1.234,56" or "0,0458" into floats.
    """
    if value is None:
        return 0.0
    cleaned = (
        str(value)
        .replace("€", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace(".", "")
        .replace(",", ".")
        .strip()
    )
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Cannot parse number from '{value}'") from exc


def parse_date(value: str):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except Exception:
            continue
    raise ValueError(f"Cannot parse date '{value}'")


def build_etf_index() -> Dict[str, dict]:
    """
    Quick lookup {symbol|ticker -> etf_info}.
    """
    index: Dict[str, dict] = {}
    for item in ETF_UCITS_CACHE:
        for key in [item.get("symbol"), item.get("ticker")]:
            if not key:
                continue
            key = key.upper()
            index.setdefault(key, item)
    return index


def load_orders(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = ["Data", "Ticker", "Order", "Prezzo", "Share", "Conto"]
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing columns in CSV: {', '.join(missing)}")

        orders = []
        for idx, row in enumerate(reader, start=1):
            try:
                orders.append(
                    {
                        "row_no": idx,
                        "date": parse_date(row["Data"]),
                        "symbol": (row["Ticker"] or "").strip().upper(),
                        "order_type": (row["Order"] or "").strip().lower(),
                        "price": parse_number(row["Prezzo"]),
                        "quantity": parse_number(row["Share"]),
                        "portfolio_name": (row["Conto"] or "").strip(),
                    }
                )
            except Exception as exc:
                raise SystemExit(f"Row {idx}: {exc}") from exc
    # Keep chronological order, stable on file order
    return sorted(orders, key=lambda o: (o["date"], o["row_no"]))


def get_or_create_portfolio(session, user_id: int, name: str, description: str = "") -> PortfolioModel:
    existing = session.execute(
        select(PortfolioModel).where(PortfolioModel.user_id == user_id, PortfolioModel.name == name)
    ).scalar_one_or_none()
    if existing:
        return existing
    portfolio = PortfolioModel(user_id=user_id, name=name, description=description, initial_capital=0.0)
    session.add(portfolio)
    session.commit()
    session.refresh(portfolio)
    return portfolio


def seed_existing_positions(session, portfolio_id: int) -> Dict[str, float]:
    """
    Build a running quantity map for an existing portfolio.
    """
    qty_map: Dict[str, float] = {}
    existing_orders = session.execute(
        select(OrderModel).where(OrderModel.portfolio_id == portfolio_id)
    ).scalars().all()
    for order in existing_orders:
        delta = order.quantity if order.order_type == "buy" else -order.quantity
        qty_map[order.symbol.upper()] = qty_map.get(order.symbol.upper(), 0.0) + delta
    return qty_map


def get_known_symbol_data(session) -> Dict[str, dict]:
    known: Dict[str, dict] = {}
    existing_orders = session.execute(select(OrderModel)).scalars().all()
    for order in existing_orders:
        sym = order.symbol.upper()
        known.setdefault(
            sym,
            {
                "name": order.name or "",
                "exchange": order.exchange or "",
                "currency": order.currency or "",
                "isin": order.isin or "",
                "ter": order.ter or "",
                "instrument_type": (order.instrument_type or "stock").lower(),
            },
        )
    return known


def order_already_present(session, portfolio_id: int, symbol: str, date_obj, quantity: float, price: float, order_type: str):
    stmt = select(OrderModel).where(
        OrderModel.portfolio_id == portfolio_id,
        OrderModel.symbol == symbol,
        OrderModel.date == date_obj,
        OrderModel.quantity == quantity,
        OrderModel.price == price,
        OrderModel.order_type == order_type,
    )
    return session.execute(stmt).scalar_one_or_none() is not None


def resolve_instrument(symbol: str, etf_index: Dict[str, dict], known_symbols: Dict[str, dict]) -> Tuple[str, dict]:
    """
    Return instrument_type and metadata for the symbol.
    Priority: known DB data -> ETF cache -> stock fallback.
    """
    sym = symbol.upper()
    if sym in known_symbols:
        data = known_symbols[sym]
        return data.get("instrument_type", "stock"), data

    etf_info = etf_index.get(sym)
    if etf_info:
        return "etf", etf_info

    return "stock", {}


def main():
    parser = argparse.ArgumentParser(description="Import orders from investmentOrders2025.csv")
    parser.add_argument("--csv", dest="csv_path", default=str(CSV_DEFAULT), help="Path to the CSV file to import")
    parser.add_argument("--user-email", dest="user_email", default=DEFAULT_USER_EMAIL, help="User email owner of the portfolios")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    orders = load_orders(csv_path)
    etf_index = build_etf_index()

    session = SessionLocal()
    user = session.execute(select(UserModel).where(UserModel.email == args.user_email)).scalar_one_or_none()
    if not user:
        raise SystemExit(f"User not found for email {args.user_email}")

    known_symbols = get_known_symbol_data(session)
    portfolios: Dict[str, PortfolioModel] = {}
    running_qty: Dict[int, Dict[str, float]] = {}

    inserted = 0
    skipped = 0

    for order in orders:
        portfolio_name = order["portfolio_name"] or "Imported"
        portfolio = portfolios.get(portfolio_name)
        if not portfolio:
            portfolio = get_or_create_portfolio(
                session,
                user.id,
                portfolio_name,
                description=f"Imported from {csv_path.name}",
            )
            portfolios[portfolio_name] = portfolio
            running_qty[portfolio.id] = seed_existing_positions(session, portfolio.id)

        symbol = order["symbol"]
        if not symbol:
            raise SystemExit(f"Row {order['row_no']}: missing symbol")

        raw_order_type = (order["order_type"] or "").lower()
        if raw_order_type.startswith("b"):
            order_type = "buy"
        elif raw_order_type.startswith("s"):
            order_type = "sell"
        else:
            raise SystemExit(f"Row {order['row_no']}: unsupported order type '{order['order_type']}'")

        portfolio_qty = running_qty[portfolio.id]
        current_qty = portfolio_qty.get(symbol, 0.0)
        if order_type == "sell" and order["quantity"] > current_qty + 1e-9:
            raise SystemExit(
                f"Row {order['row_no']}: sell of {order['quantity']} {symbol} exceeds current qty {current_qty}"
            )

        instrument_type, meta = resolve_instrument(symbol, etf_index, known_symbols)
        isin = meta.get("isin", "") if isinstance(meta, dict) else ""
        name = meta.get("name", "") if isinstance(meta, dict) else ""
        if isinstance(meta, dict):
            exchange = meta.get("exchange") or meta.get("listing") or ""
            currency = meta.get("currency", "")
            ter_val = meta.get("ter", "")
        else:
            exchange = ""
            currency = ""
            ter_val = ""

        if order_already_present(
            session,
            portfolio.id,
            symbol,
            order["date"],
            order["quantity"],
            order["price"],
            order_type,
        ):
            skipped += 1
            continue

        db_order = OrderModel(
            portfolio_id=portfolio.id,
            symbol=symbol,
            isin=isin,
            ter=str(ter_val) if ter_val is not None else "",
            name=name,
            exchange=exchange,
            currency=currency or "EUR",
            quantity=order["quantity"],
            price=order["price"],
            commission=0.0,
            instrument_type=instrument_type,
            order_type=order_type,
            date=order["date"],
        )
        session.add(db_order)

        delta = order["quantity"] if order_type == "buy" else -order["quantity"]
        portfolio_qty[symbol] = current_qty + delta
        inserted += 1

    session.commit()

    print(f"Imported {inserted} orders, skipped {skipped} duplicates.")
    for name, pf in portfolios.items():
        print(f"- Portfolio '{name}' -> id {pf.id}")


if __name__ == "__main__":
    main()
