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

from utils.database import engine as db_engine, run_migrations
from routers import auth_router, portfolios_router, orders_router, symbols_router, market_data_router

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
# =============================================================================

from models import Base

Base.metadata.create_all(bind=db_engine)
run_migrations()

# =============================================================================
# ROUTER MOUNTING
# =============================================================================

app.include_router(auth_router)
app.include_router(portfolios_router)
app.include_router(orders_router)
app.include_router(symbols_router)
app.include_router(market_data_router)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
