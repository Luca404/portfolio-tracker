# pfTrackr

Full-stack web app for tracking and analyzing investment portfolios. Built with FastAPI + React, backed by Supabase.

Part of the **Trackrs ecosystem** alongside [Trackr](../trackr) (personal finance) and [fitTrackr](../fitness-tracker) (calorie and nutrition tracking). Shares the same Supabase database with Trackr for investment data.

## Features

- **Portfolio management** — multiple portfolios, multi-currency, configurable benchmarks and risk-free rate sources
- **Order tracking** — buy/sell orders for ETFs, stocks, and bonds; commission support; stock split auto-adjustment
- **Performance** — XIRR (money-weighted return), time-weighted return, gain/loss, asset breakdown
- **Risk analytics** — Sharpe, Sortino, Max Drawdown, volatility, correlation matrix, drawdown chart
- **Monte Carlo** — simulation with confidence intervals
- **Portfolio optimization** — Markowitz / Efficient Frontier, Max Sharpe, discrete allocation
- **ETF discovery** — search across 2800+ UCITS ETFs with ISIN lookup
- **Bond support** — metadata lookup (name, coupon, maturity, YTM, duration) via Borsa Italiana (MOT/EuroMOT/ExtraMOT) and Frankfurt exchange
- **Benchmark comparison** — portfolio vs market benchmark (S&P 500, VWCE, or custom)
- **DCA vs lump-sum** — comparison analysis

## Stack

**Backend** — Python / FastAPI
- Supabase (`supabase-py`) — primary database and auth
- SQLAlchemy 2.0 + SQLite — ephemeral local cache for prices, FX rates, benchmarks
- yfinance, PyPortfolioOpt, NumPy, Pandas, SciPy

**Frontend** — React 18 (JSX, no TypeScript) / Vite
- Tailwind CSS, Recharts, Lucide React

## Getting Started

### Backend

Create `backend/.env`:

```env
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SECRET_KEY=...        # service_role key (not anon)
FMP_API_KEY=...                # optional — USD risk-free rate + stock price fallback
```

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

`frontend/src/config.js` controls `API_URL` — set to `http://localhost:8000` in dev.

## Project Structure

```
portfolio-tracker/
├── backend/
│   ├── main.py                    # FastAPI app, CORS, router registration, SQLite cache init
│   ├── routers/
│   │   ├── auth.py                # /auth/* — wraps Supabase Auth (register, login, me)
│   │   ├── portfolios.py          # /portfolios/* — CRUD, analytics, history, DCA comparison, benchmark comparison
│   │   ├── orders.py              # /orders/* — CRUD, optimization
│   │   ├── symbols.py             # /symbols/* — ETF/stock search, ISIN lookup, bond lookup
│   │   └── market_data.py         # /market-data/* — prices, benchmarks, risk-free rates
│   ├── utils/
│   │   ├── pricing.py             # ETF/stock/FX pricing — yfinance cascade, cache management
│   │   ├── portfolio.py           # XIRR, returns, risk metrics, stock split adjustment
│   │   ├── etf_cache.py           # In-memory ETF_UCITS_CACHE loaded from Supabase at startup
│   │   ├── bond_cache.py          # In-memory BOND_METADATA_CACHE loaded from Supabase at startup
│   │   ├── bond_scraper.py        # Borsa Italiana + Frankfurt scraper for bond metadata
│   │   ├── auth.py                # JWT verification via supabase.auth.get_user()
│   │   └── supabase_client.py     # Supabase singleton
│   ├── models/
│   │   ├── base.py                # SQLAlchemy declarative base
│   │   └── cache.py               # SQLAlchemy models for SQLite cache only (prices, FX, benchmarks)
│   ├── schemas/                   # Pydantic request/response schemas
│   ├── etf_cache_ucits.py         # Static fallback ETF data (used if Supabase returns < 100 rows)
│   └── tests/
│       ├── test_portfolio_logic.py
│       └── test_order_validation.py
├── frontend/
│   └── src/
│       ├── App.jsx                # Global state: token, user, portfolios, currentView
│       ├── config.js              # API_URL
│       ├── pages/                 # AuthPage, DashboardPage, OrdersPage, AnalyzePage, ComparePage, OptimizePage
│       ├── components/            # Navbar, PortfoliosList, MetricCard, charts/, skeletons/
│       └── services/api.js        # Centralized API client with auth headers
└── scripts/
    ├── scrape_justetf.py          # Scrape JustETF for UCITS ETF list
    ├── generate_etf_cache.py      # Build ETF cache CSV from scraped data
    └── import_orders_from_csv.py  # Bulk order import from CSV
```

## Auth flow

1. Login/register via `/auth/login` or `/auth/register` → calls Supabase Auth → returns JWT
2. Frontend sends JWT as `Authorization: Bearer <token>`
3. `verify_token()` calls `supabase.auth.get_user(token)` → returns `user_id` (UUID)
4. All routers use `user_id: str = Depends(verify_token)`

## Caching

**SQLite `cache.db`** — ephemeral on Railway, resets on every redeploy:
- `etf_price_cache`, `stock_price_cache` — historical OHLC data, keyed by ISIN or symbol
- `exchange_rate_cache` — FX pairs (e.g. `USDEUR=X`)
- `risk_free_rate_cache` — per currency
- `market_benchmark_cache` — S&P 500 (`^GSPC`), VWCE (`VWCE.DE`), or custom ticker

**Supabase** (persistent):
- `etf_ucits_cache` — 2800+ UCITS ETF metadata; loaded into memory at startup, new ETFs added on-demand via `/symbols/isin-lookup`
- `bond_metadata_cache` — bond metadata with 24h TTL; populated on-demand via `/symbols/bond-lookup`

**Frontend localStorage** — `pf_summaries_cache` with 24h TTL (5 min if all values are 0).

## Pricing cascade

**ETF** — SQLite cache → yfinance (tries ISIN directly, then `symbol`, `symbol.DE`, `.L`, `.PA`, `.MI`)

**Stock** — SQLite cache → yfinance → FMP (`FMP_API_KEY`) → AlphaVantage

**Risk-free rate** — EUR: ECB API · USD: FMP treasury rates · GBP: hardcoded 4.0% (TODO)

**Bond metadata** — Supabase cache (24h TTL) → Borsa Italiana (6 URL patterns across MOT/EuroMOT/ExtraMOT) → Frankfurt (complementary)

## API endpoints

| Group | Endpoints |
|---|---|
| Auth | `POST /auth/register`, `POST /auth/login`, `GET /auth/me` |
| Portfolios | `GET/POST /portfolios`, `GET/PUT/DELETE /portfolios/{id}`, `GET /portfolios/analysis/{id}`, `GET /portfolios/history/{id}/{symbol}`, `GET /portfolios/compare-dca/{id}`, `GET /portfolios/compare-benchmark/{id}` |
| Orders | `GET /orders/{portfolio_id}`, `POST /orders`, `PUT /orders/{id}`, `DELETE /orders/{id}`, `POST /orders/optimize` |
| Symbols | `GET /symbols/search`, `GET /symbols/ucits`, `GET /symbols/isin-lookup`, `GET /symbols/bonds`, `GET /symbols/bond-lookup` |
| Market data | `GET /market-data/{symbol}`, `GET /market-data/risk-free-rate/{currency}`, `GET /market-data/benchmark/{currency}` |


## Deployment

- **Backend** — Render: `uvicorn main:app --host 0.0.0.0 --port $PORT`; required env vars: `SUPABASE_URL`, `SUPABASE_SECRET_KEY`; optional: `FMP_API_KEY`, `ALPHAVANTAGE_API_KEY`
- **Frontend** — Vercel: `frontend/` as root directory, build command `npm run build`, output `dist/`

## Known limitations

- ETF prices are EOD only — no intraday data for European ETFs via yfinance
- Bond portfolio math (price as % of nominal) not yet integrated into portfolio aggregation
- GBP risk-free rate hardcoded at 4.0%
- ETF UCITS cache needs manual refresh ~monthly via `scripts/`
- `history_mode` field (`full_orders` / `positions_only`) stored on portfolios but not yet consumed by analytics — metrics like XIRR should be hidden for `positions_only` portfolios
