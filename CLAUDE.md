# Portfolio Tracker — CLAUDE.md

App full-stack per il tracking e l'analisi di portafogli di investimento. FastAPI (Python) + React 18 + Supabase.

## Stack

**Backend:**
- FastAPI (Python) — API REST
- Supabase (`supabase-py`) — DB PostgreSQL + Auth
- SQLAlchemy 2.0 (ORM con SQLite) — solo tabelle cache locali (`backend/cache.db`)
- Pydantic — validazione dati
- yfinance — dati di mercato
- NumPy / Pandas — analisi numerica
- PyPortfolioOpt — Modern Portfolio Theory
- JWT (verifica Supabase JWT) — autenticazione

**Frontend:**
- React 18 (JSX, non TSX)
- Vite — build tool, dev server → http://localhost:5173
- Tailwind CSS — styling
- Recharts — grafici
- Lucide React — icone

## Comandi

**Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload   # → http://localhost:8000
```

**Frontend:**
```bash
cd frontend
npm run dev    # → http://localhost:5173
npm run build
npm run preview
```

## Env vars

**Backend** (`.env` in `backend/`):
```env
SUPABASE_URL=https://...          # Supabase project URL (required)
SUPABASE_SERVICE_KEY=...          # Supabase service role key (required)
SUPABASE_JWT_SECRET=...           # Supabase JWT secret (required) — Dashboard → Settings → API → JWT Secret
FMP_API_KEY=...                   # Financial Modeling Prep API (optional)
CACHE_DB_URL=sqlite:///./cache.db # SQLite per cache (opzionale, default: cache.db)
```

**Frontend** (`src/config.js`):
```js
// dev
export const API_URL = 'http://localhost:8000';
// produzione (Railway)
export const API_URL = 'https://portfolio-tracker-production-3bd4.up.railway.app';
```

## Struttura

```
portfolio-tracker/
├── backend/
│   ├── main.py                   # Setup FastAPI, CORS, router registration, cache DB init
│   ├── models/
│   │   ├── base.py
│   │   └── cache.py              # ETFPriceCache, StockPriceCache, ExchangeRateCache, ecc. (SQLite)
│   ├── routers/
│   │   ├── auth.py               # /auth/* — usa Supabase Auth sign_up/sign_in
│   │   ├── portfolios.py         # /portfolios/*
│   │   ├── orders.py             # /orders/* + optimize
│   │   ├── symbols.py            # /symbols/* (search, ucits, etf-list)
│   │   ├── market_data.py        # /market-data/*
│   │   ├── accounts.py           # /api/accounts/* (usato da trackr PWA)
│   │   ├── categories.py         # /api/categories/* + subcategories (usato da trackr PWA)
│   │   └── transactions.py       # /api/transactions/* (usato da trackr PWA)
│   ├── schemas/                  # Pydantic schemas
│   ├── utils/
│   │   ├── database.py           # SQLite per cache, get_db dependency
│   │   ├── auth.py               # Verifica JWT Supabase via HTTP /auth/v1/user, restituisce user_id (UUID)
│   │   ├── supabase_client.py    # Client Supabase singleton (get_supabase())
│   │   ├── etf_cache.py          # ETF_UCITS_CACHE in-memory condivisa (caricata da Supabase all'avvio)
│   │   ├── pricing.py            # Prezzi ETF/stock, valute, dati mercato
│   │   ├── portfolio.py          # XIRR, ritorni, metriche rischio
│   │   ├── symbols.py            # Ricerca e validazione simboli
│   │   ├── dates.py
│   │   ├── cache.py              # Invalidazione cache SQLite
│   │   ├── default_categories.py # Crea categorie di default su Supabase
│   │   └── helpers.py
│   └── etf_cache_ucits.py        # Fallback statico ETF UCITS (usato solo se Supabase non disponibile)
├── frontend/
│   └── src/
│       ├── App.jsx               # State globale (token, user, currentView, portfolios)
│       ├── config.js             # API_URL
│       ├── pages/
│       │   ├── AuthPage.jsx
│       │   ├── DashboardPage.jsx
│       │   ├── OrdersPage.jsx
│       │   ├── AnalyzePage.jsx
│       │   ├── ComparePage.jsx
│       │   └── OptimizePage.jsx
│       ├── components/
│       │   ├── Navbar.jsx
│       │   ├── PortfoliosList.jsx
│       │   ├── MetricCard.jsx
│       │   ├── charts/           # CorrelationHeatmap, MonteCarloChart, DrawdownChart, AssetPerformanceChart
│       │   └── skeletons/        # Loading placeholders
│       ├── services/api.js       # Client API centralizzato
│       └── utils/                # currency.js, dates.js, cache.js, helpers.js
└── scripts/
    ├── etf_cache.py
    └── import_orders_from_csv.py
```

## Architettura dati

```
Frontend (React state in App.jsx)
    → services/api.js  (Authorization: Bearer <supabase_jwt>)
        → FastAPI routers
            → supabase-py → Supabase PostgreSQL (dati utente)
            → SQLAlchemy → SQLite cache.db (cache prezzi, cambi, ecc.)
            → yfinance / external APIs
```

### Auth flow
1. Login/register via `/auth/login` o `/auth/register` → chiama Supabase Auth → ritorna JWT
2. Il frontend manda il JWT in `Authorization: Bearer <token>`
3. `verify_token()` chiama `supabase.auth.get_user(token)` → restituisce `user_id` (UUID)
   - Non usa più HS256 + JWT secret locale: funziona con qualsiasi algoritmo Supabase usa
4. Tutti i router usano `user_id: str = Depends(verify_token)` — non c'è più `UserModel`

### Pricing & Cache

**Tabelle cache SQLite** (effimere su Railway, si resettano ad ogni redeploy):
- `etf_price_cache` — storico prezzi ETF, chiave: ISIN
- `stock_price_cache` — storico prezzi stock, chiave: symbol
- `exchange_rate_cache` — storico cambi, chiave: pair (es. "USDEUR=X")
- `risk_free_rate_cache` — tasso risk-free corrente + storico, chiave: currency
- `market_benchmark_cache` — storico benchmark (S&P 500, VWCE), chiave: currency

**Tabelle Supabase** (persistenti):
- `etf_ucits_cache` — metadati ETF UCITS (isin, ticker, exchange, name, currency, ter); PK: (isin, ticker, exchange)
  - Caricata in memoria all'avvio (`utils/etf_cache.py → ETF_UCITS_CACHE`)
  - Fallback: `etf_cache_ucits.py` statico se Supabase restituisce < 100 righe
  - Nuovi ETF scoperti via `/symbols/isin-lookup` vengono aggiunti automaticamente
- `bond_metadata_cache` — metadati bond (isin PK, name, issuer, coupon, maturity, currency, ytm_gross, ytm_net, duration, coupon_frequency, updated_at)
  - Caricata in memoria all'avvio (`utils/bond_cache.py → BOND_METADATA_CACHE`)
  - Popolata on-demand via `/symbols/bond-lookup`: cascade Supabase (24h TTL) → Borsa Italiana → Frankfurt
  - Scraper: `utils/bond_scraper.py` — `scrape_borsa_italiana_bond(isin)` prova 6 URL (MOT/btp, MOT/altri-titoli-di-stato, MOT/obbligazioni, EuroMOT/titoli-di-stato-esteri, EuroMOT/obbligazioni, ExtraMOT); `fetch_frankfurt_bond_metadata(isin)` complementare

**Invalidazione prezzi**: cache valida se l'ultimo dato copre l'ultimo giorno di mercato aperto:
- Lunedì → accetta dati di venerdì (mercati chiusi sabato/domenica)
- Martedì–Venerdì → richiede dati di ieri
Al miss: scarica dati freschi, merge con storico esistente (no duplicati), salva in SQLite.

**ETF** (`get_etf_price_and_history(isin, db)`):
1. Cache hit? → restituisce subito
2. Tentativo justetf_scraping (non installato su Railway → skip immediato)
3. yfinance cascade per ISIN:
   - Prova con ISIN direttamente → valida che il ticker restituito corrisponda al symbol atteso
   - Se mismatch: prova `symbol` (es. "VWCE") → `symbol.DE` → `symbol.L` → `symbol.PA` → `symbol.MI`
   - Winner: salva in cache e ritorna
4. Last resort: cache stale se disponibile
5. Tutti falliti: HTTP 400

**Stock** (`get_stock_price_and_history_cached(symbol, db)`):
1. Cache hit? → restituisce subito
2. yfinance (prima scelta: gratuito, nessun rate limit, ottimo per stock globali)
3. FMP historical-price-eod (richiede `FMP_API_KEY`)
4. FMP historical-chart (richiede `FMP_API_KEY`)
5. AlphaVantage (richiede `ALPHAVANTAGE_API_KEY`)

**Cambi** (`get_exchange_rate_history(from, to, db)`):
- yfinance con ticker `{FROM}{TO}=X` (es. `USDEUR=X`), storico completo, cache SQLite

**Risk-free rate** (`get_risk_free_rate(currency, db)`):
- EUR → ECB API (Main Refinancing Operations rate, gratuito)
- USD → FMP treasury-rates (richiede `FMP_API_KEY`), tasso 10Y
- GBP → hardcoded 4.0% (TODO: implementare UK Gilt)

**Benchmark** (`fetch_market_benchmark(currency, db)`):
- USD → ^GSPC (S&P 500)
- EUR → VWCE.DE (Vanguard FTSE All-World, quotato su XETRA)
- Custom: ticker yfinance arbitrario

**Stock splits** (`get_stock_splits` + `apply_splits_to_orders`):
- Ad ogni caricamento portfolio, rileva split via `yf.Ticker.splits` dalla data del primo ordine
- Aggiusta retroattivamente: `quantity × ratio`, `price ÷ ratio`
- NON modifica il DB — crea copie degli ordini in memoria (SimpleNamespace)

## Supabase DB (tabelle richieste)

Schema condiviso con il progetto Trackr. Tabelle necessarie con le colonne richieste:

**`portfolios`**: `id`, `user_id` (uuid), `name`, `description`, `initial_capital`, `reference_currency`, `risk_free_source`, `market_benchmark`, `created_at`

**`orders`**: `id`, `portfolio_id`, `user_id` (uuid), `symbol`, `isin`, `ter`, `name`, `exchange`, `currency`, `quantity`, `price`, `commission`, `instrument_type`, `order_type`, `date`, `created_at`

**`accounts`**: `id`, `user_id`, `name`, `icon`, `initial_balance`, `is_favorite`, `created_at`, `updated_at`

**`categories`**: `id`, `user_id`, `name`, `icon`, `category_type`, `created_at`, `updated_at`

**`subcategories`**: `id`, `category_id`, `name`, `icon`

**`transactions`**: `id`, `user_id`, `account_id`, `type`, `category`, `subcategory`, `amount`, `description`, `date`, `ticker`, `quantity`, `price`, `created_at`, `updated_at`

**`etf_ucits_cache`**: `isin`, `ticker`, `exchange` (PK composita), `name`, `currency`, `ter`, `created_at` — RLS disabled (dati pubblici, accesso solo via service role key)

**`bond_metadata_cache`**: `isin` (PK), `name`, `issuer`, `coupon`, `maturity`, `currency`, `ytm_gross`, `ytm_net`, `duration`, `coupon_frequency`, `updated_at` — RLS disabled. TTL 24h in-memory. Aggiornata on-demand via `/symbols/bond-lookup`.
- DDL: `ALTER TABLE bond_metadata_cache ADD COLUMN IF NOT EXISTS name TEXT DEFAULT '';`

## API endpoints principali

**Portfolio tracker frontend:**
- `POST /auth/register` / `POST /auth/login` / `GET /auth/me`
- `GET|POST /portfolios` / `GET|PUT|DELETE /portfolios/{id}`
- `GET /portfolios/analysis/{id}` — analytics avanzate
- `GET|POST /orders/{portfolio_id}` / `POST /orders/optimize`
- `GET /symbols/search` / `GET /symbols/ucits`
- `GET /symbols/bonds` — lista cache bond in-memoria (caricata da Supabase all'avvio)
- `GET /symbols/bond-lookup?isin=...` — scraping on-demand: Supabase cache (24h) → Borsa Italiana (MOT/EuroMOT/ExtraMOT) → Frankfurt (complementare) → 404
- `GET /market-data/{symbol}` / `GET /market-data/benchmark/{currency}`

**Trackr PWA (stesso backend):**
- `GET|POST /api/accounts/` / `GET|PUT|DELETE /api/accounts/{id}`
- `GET|POST /api/categories` / `GET|PUT|DELETE /api/categories/{id}`
- `GET|POST /api/transactions` / `GET|PUT|DELETE /api/transactions/{id}`

Swagger UI: http://localhost:8000/docs

## Features principali

- **Portfolio management**: portafogli multipli, multi-currency, benchmark personalizzabili
- **Order tracking**: buy/sell, ETF e stock, validazione simboli, commissioni, import CSV
- **Performance**: XIRR, time-weighted return, gain/loss, composizione asset
- **Risk analytics**: Sharpe, Sortino, Max Drawdown, volatilità, correlazione, Monte Carlo
- **Optimization**: Markowitz / Efficient Frontier, Max Sharpe, allocazione discreta
- **Expense tracking**: accounts, categories (con subcategories), transactions

## Note importanti

- `verify_token()` restituisce `user_id: str` (UUID Supabase) — non più `UserModel`
- I router usano `get_supabase()` per accedere ai dati utente, `get_db()` solo per la cache SQLite
- Le righe Supabase (dicts) vengono convertite in `SimpleNamespace` via `_order_from_row()` nei router portfolios/orders per compatibilità con le utility functions
- `current_balance` degli account non è su DB, viene calcolato in tempo reale
- Lo stato globale frontend è in `App.jsx` (no Redux/Zustand)
- CORS configurato in `backend/main.py`

## TODO / Problemi noti

- **Prezzi intraday ETF europei**: justetf_scraping e yfinance forniscono solo prezzi EOD (End of Day). Lunedì (e in generale durante la giornata) il prezzo mostrato è l'ultimo close disponibile, non il prezzo live. Nessuna delle fonti attuali espone prezzi intraday affidabili per ETF UCITS su XETRA/Euronext. Possibile soluzione: usare `yf.Ticker.fast_info.last_price` come prezzo corrente separato dallo storico, ma la copertura per ETF europei è limitata.

- **ETF UCITS cache — aggiornamento periodico**: la tabella `etf_ucits_cache` su Supabase è stata popolata con 2834 ETF dal file statico. I nuovi ETF si aggiungono on-demand via `/symbols/isin-lookup`. Per un aggiornamento sistematico: rieseguire `scripts/scrape_justetf.py` (senza filtro size) + `scripts/generate_etf_cache.py` + reimportare il CSV su Supabase ~1 volta al mese.

- **Portfolio history — date gaps**: `aggregate_portfolio_history()` in `utils/portfolio.py` esclude le date in cui almeno un asset attivo non ha dati di prezzo. Il forward-fill (usa l'ultimo prezzo noto) è stato implementato ma non risolve completamente il problema: se yfinance non restituisce storico sufficiente per un simbolo (es. ETF europeo con storia limitata), le date precedenti alla prima quotazione disponibile vengono saltate, causando gap nel grafico. Da investigare: quale simbolo nel portafoglio ha storico limitato, e valutare se usare prezzi parziali (escludere il simbolo senza dati invece di invalidare l'intera data).

- **Bond — US Treasuries e obbligazioni societarie**: non ancora supportati. Per US Treasuries si può usare EODHD (piano gratuito) o TreasuryDirect API. Per corporate bonds europei/americani: EODHD oppure scraping di Borsa Italiana (MOT/ExtraMOT). Da implementare in futuro.

- **Bond — integrazione prezzi in portfolio analytics**: i bond vengono salvati con `price` in % del nominale. Il calcolo del valore di portafoglio deve usare `nominale × price / 100` invece del solito `quantity × price`. Le funzioni `utils/portfolio.py` e `utils/pricing.py` non gestiscono ancora questo caso — da implementare quando si aggiunge il supporto completo ai bond nell'analisi del portafoglio.

## Deployment

- **Backend**: Railway — `https://portfolio-tracker-production-3bd4.up.railway.app`
  - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
  - Env vars da impostare: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (non serve più `SUPABASE_JWT_SECRET`)
  - Opzionali: `FMP_API_KEY` (risk-free USD + fallback stock), `ALPHAVANTAGE_API_KEY` (fallback stock)
  - SQLite cache è effimero (si resetta ad ogni redeploy — accettabile, è solo cache)
- **Frontend**: Vercel — deploy automatico da GitHub, `frontend/` come root directory
  - Build command: `npm run build`, output dir: `dist`
