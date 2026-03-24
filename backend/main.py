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

# Carica ETF custom scoperti a runtime (salvati su Supabase) nella cache in-memory
def _load_supabase_etf_cache():
    from utils.supabase_client import get_supabase
    from etf_cache_ucits import ETF_UCITS_CACHE
    try:
        sb = get_supabase()
        rows = sb.table("etf_ucits_cache").select("*").execute().data or []
        existing_keys = {(e.get("isin", "").upper(), e.get("symbol", "").upper()) for e in ETF_UCITS_CACHE}
        added = 0
        for row in rows:
            key = (row["isin"].upper(), row["ticker"].upper())
            if key not in existing_keys:
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
                existing_keys.add(key)
                added += 1
        if added:
            print(f"[ETF cache] loaded {added} custom ETFs from Supabase")
    except Exception as e:
        print(f"[ETF cache] Supabase load failed (non-fatal): {e}")

_load_supabase_etf_cache()

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
