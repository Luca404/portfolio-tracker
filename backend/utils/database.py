import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./portfolio.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 30})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Database dependency for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
                if attempt < retries - 1:
                    import time
                    time.sleep(base_delay * (2 ** attempt))
            else:
                raise
    raise OperationalError("Database locked after multiple retries", None, None)


# Database migrations
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


def run_migrations():
    """Run all database migrations."""
    ensure_orders_table_has_isin()
    ensure_portfolios_table_has_reference_currency()
    ensure_portfolios_table_has_market_settings()
