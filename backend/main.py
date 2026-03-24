import os
from pathlib import Path

# =============================================================================
# LOAD ENVIRONMENT VARIABLES FIRST (before any other imports!)
# =============================================================================

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

# =============================================================================
# NOW SAFE TO IMPORT OTHER MODULES
# =============================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.database import engine, run_migrations
from routers import auth_router, portfolios_router, orders_router, symbols_router, market_data_router, transactions_router, categories_router, accounts_router

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(title="Portfolio Tracker API")

# =============================================================================
# CORS
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATABASE INITIALIZATION
# Solo tabelle cache su SQLite — i dati utente sono su Supabase
# =============================================================================

from models.base import Base
from models.cache import ETFPriceCacheModel, StockPriceCacheModel, ExchangeRateCacheModel, RiskFreeRateCacheModel, MarketBenchmarkCacheModel  # noqa: ensure cache models are registered

Base.metadata.create_all(bind=engine)
run_migrations()

# Carica la cache ETF UCITS: Supabase come fonte primaria, file statico come fallback
def _load_etf_cache():
    from utils.supabase_client import get_supabase
    from utils.etf_cache import ETF_UCITS_CACHE

    # 1. Prova Supabase
    try:
        sb = get_supabase()
        rows = sb.table("etf_ucits_cache").select("*").limit(10000).execute().data or []
        if len(rows) >= 100:
            for row in rows:
                ETF_UCITS_CACHE.append({
                    "symbol": row["ticker"],
                    "isin": row["isin"],
                    "name": row.get("name", ""),
                    "currency": row.get("currency", ""),
                    "exchange": row.get("exchange", ""),
                    "ter": row.get("ter"),
                    "ticker": row["ticker"],
                    "type": "ETF",
                })
            print(f"[ETF cache] loaded {len(rows)} ETFs from Supabase")
            return
        print(f"[ETF cache] Supabase returned only {len(rows)} rows, falling back to static file")
    except Exception as e:
        print(f"[ETF cache] Supabase load failed, falling back to static file: {e}")

    # 2. Fallback: file statico
    try:
        from etf_cache_ucits import ETF_UCITS_CACHE as static
        ETF_UCITS_CACHE.extend(static)
        print(f"[ETF cache] loaded {len(static)} ETFs from static file")
    except ImportError:
        print("[ETF cache] static file not found, ETF cache is empty")

_load_etf_cache()

# =============================================================================
# ROUTER MOUNTING
# =============================================================================

app.include_router(auth_router)
app.include_router(portfolios_router)
app.include_router(orders_router)
app.include_router(symbols_router)
app.include_router(market_data_router)
app.include_router(transactions_router)
app.include_router(categories_router)
app.include_router(accounts_router)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
