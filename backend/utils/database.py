import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# SQLite usato esclusivamente per le cache (etf_price_cache, stock_price_cache, ecc.)
# I dati utente (portfolios, orders, accounts, categories, transactions) sono su Supabase.
CACHE_DB_URL = os.environ.get("CACHE_DB_URL", "sqlite:///./cache.db")
engine = create_engine(CACHE_DB_URL, connect_args={"check_same_thread": False, "timeout": 30})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """SQLite session dependency — solo per le tabelle cache."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def commit_with_retry(db: Session, retries: int = 3, base_delay: float = 0.3):
    from sqlalchemy.exc import OperationalError
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
    from sqlalchemy.exc import OperationalError
    raise OperationalError("Database locked after multiple retries", None, None)


def run_migrations():
    """Placeholder — le tabelle cache vengono create da Base.metadata.create_all in main.py."""
    pass
