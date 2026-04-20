# pfTrackr — CLAUDE.md

Full-stack investment portfolio tracker. FastAPI (Python) + React 18 JSX + Supabase. Part of the Trackr ecosystem — see root `CLAUDE.md` and `../docs/supabase-schema.md`.

## Stack

**Backend**: FastAPI, supabase-py, SQLAlchemy 2.0 + SQLite (cache only), Pydantic, yfinance, NumPy/Pandas, PyPortfolioOpt
**Frontend**: React 18 (JSX, not TSX), Vite, Tailwind CSS, Recharts, Lucide React

## Commands

**Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload   # → http://localhost:8000  (Swagger: /docs)
python -m pytest            # run tests
```

**Frontend:**
```bash
cd frontend
npm run dev    # → http://localhost:5173
npm run build
```

## Env vars

**Backend** (`backend/.env`):
```env
SUPABASE_URL=...              # required
SUPABASE_SERVICE_KEY=...      # required (service_role key, not anon)
FMP_API_KEY=...               # optional — USD risk-free rate + stock price fallback
ALPHAVANTAGE_API_KEY=...      # optional — stock price fallback
CACHE_DB_URL=sqlite:///./cache.db  # optional, default: cache.db
```
`SUPABASE_JWT_SECRET` no longer needed — `verify_token()` calls `supabase.auth.get_user(token)` directly.

**Frontend** (`frontend/src/config.js`):
```js
export const API_URL = 'http://localhost:8000';
// prod: 'https://portfolio-tracker-p6ha.onrender.com'
```
No Vite env vars — `API_URL` is in `config.js`, switch manually for dev/prod.

## Architecture constraints

- **`verify_token()` returns `user_id: str` (UUID string)**, not a UserModel. All routers: `user_id: str = Depends(verify_token)`.
- **SQLAlchemy + SQLite = price cache only.** `cache.db` is ephemeral on Render (resets on every redeploy). Never use it for user data.
- **Supabase = source of truth** for all user/app data. Use `get_supabase()` for user data. Use `get_db()` only for SQLite cache.
- **`_order_from_row()`** converts Supabase row dicts → `SimpleNamespace` in portfolio/orders routers, for compatibility with `utils/portfolio.py` utility functions.
- **pfTrackr creates portfolios with `profile_id = user_id`** (main profile only). Full multi-profile support not yet implemented.
- CORS configured in `backend/main.py`.
- `etf_ucits_cache` and `bond_metadata_cache` Supabase tables have RLS disabled — service role key access only.

## Shared DB

Shared tables with trackr: `accounts`, `categories`, `subcategories`, `transactions`, `transfers`, `portfolios`, `orders`, `profiles`, `profile_members`. Full schema → `../docs/supabase-schema.md`.

`/api/accounts`, `/api/categories`, `/api/transactions` backend endpoints are consumed by the Trackr PWA, not by the pfTrackr frontend.

## Deployment

- **Backend**: Render — start: `uvicorn main:app --host 0.0.0.0 --port $PORT`. Required: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.
- **Frontend**: Vercel — root dir: `frontend/`, build: `npm run build`, output: `dist/`.

## Supabase

CLI project at root `../supabase/`. Run all `supabase` CLI commands from `Python/`. Migrations: `../supabase/migrations/`. Schemas: `../supabase/schemas/`.

## Known issues

See `docs/known-issues.md`.
