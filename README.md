# Portfolio Tracker

Un'applicazione web full-stack per tracciare e analizzare portafogli di investimento con funzionalità avanzate di analisi del rischio e ottimizzazione.

## 🏗️ Architettura del Progetto

```
portfolio-tracker/
├── backend/               # Backend FastAPI (Python)
│   ├── main.py           # App setup only (71 righe)
│   ├── models/           # SQLAlchemy models
│   │   ├── user.py      # User model
│   │   ├── portfolio.py # Portfolio model
│   │   ├── order.py     # Order model
│   │   └── cache.py     # Cache models (ETF, Stock, Exchange, etc.)
│   │
│   ├── routers/          # API route handlers (24 endpoints)
│   │   ├── auth.py          # 3 endpoints: /auth/* (register, login, me)
│   │   ├── portfolios.py    # 8 endpoints: /portfolios/* (CRUD, analytics, history)
│   │   ├── orders.py        # 5 endpoints: /orders/* (CRUD, optimize)
│   │   ├── symbols.py       # 4 endpoints: /symbols/* (search, ucits, etf-list, stats)
│   │   └── market_data.py   # 4 endpoints: /market-data/* (prices, rates, benchmarks)
│   │
│   ├── schemas/          # Pydantic schemas (validation)
│   │   ├── user.py      # UserRegister, UserLogin, Token
│   │   ├── portfolio.py # Portfolio schema
│   │   └── order.py     # Order, OptimizationRequest
│   │
│   ├── utils/            # Utility modules
│   │   ├── database.py  # DB connection, migrations
│   │   ├── auth.py      # JWT, password hashing
│   │   ├── dates.py     # Date formatting, parsing
│   │   ├── cache.py     # Cache invalidation
│   │   ├── pricing.py   # ETF/Stock pricing, conversions (1,151 righe)
│   │   ├── portfolio.py # Portfolio calculations, XIRR (452 righe)
│   │   ├── symbols.py   # Symbol search/validation (86 righe)
│   │   └── helpers.py   # Data validation (25 righe)
│   │
│   └── etf_cache_ucits.py # UCITS ETF cache (local data)
│
├── scripts/              # Utility scripts
│   ├── etf_cache.py     # ETF data cache builder
│   └── import_orders_from_csv.py # CSV import utility
│
├── frontend/             # Frontend React
│   ├── public/          # Assets statici
│   └── src/
│       ├── App.jsx                    # Componente principale (170 righe)
│       ├── main.jsx                   # Entry point
│       ├── config.js                  # Configurazione (API_URL)
│       │
│       ├── pages/                     # Componenti pagina (routing)
│       │   ├── index.js              # Export centralizzato
│       │   ├── AuthPage.jsx          # Login/Registrazione
│       │   ├── DashboardPage.jsx     # Dashboard portfolio
│       │   ├── OrdersPage.jsx        # Gestione ordini
│       │   ├── AnalyzePage.jsx       # Analisi avanzate
│       │   ├── ComparePage.jsx       # Confronto (placeholder)
│       │   └── OptimizePage.jsx      # Ottimizzazione MPT
│       │
│       ├── components/                # Componenti riutilizzabili
│       │   ├── Navbar.jsx            # Barra di navigazione
│       │   ├── PortfoliosList.jsx    # Lista portfolio
│       │   ├── MetricCard.jsx        # Card per metriche
│       │   │
│       │   ├── charts/               # Componenti grafici
│       │   │   ├── CorrelationHeatmap.jsx
│       │   │   ├── MonteCarloChart.jsx
│       │   │   ├── DrawdownChart.jsx
│       │   │   └── AssetPerformanceChart.jsx
│       │   │
│       │   └── skeletons/            # Loading skeletons
│       │       ├── PortfolioCardSkeleton.jsx
│       │       ├── DashboardSkeleton.jsx
│       │       └── AnalysisTabSkeleton.jsx
│       │
│       └── utils/                     # Funzioni utility
│           ├── currency.js           # Gestione valute
│           ├── dates.js              # Parsing/formattazione date
│           ├── cache.js              # Cache helpers
│           └── helpers.js            # Utility varie
│
├── scripts/              # Script utility
└── data/                 # Dati locali (ETF cache)
```

## 📦 Backend - Struttura Modulare

Il backend è stato refactorizzato da un singolo file monolitico (3,244 righe) a una struttura modulare organizzata:

### Models (SQLAlchemy ORM)
Database models per persistenza dati:
- **UserModel**: Autenticazione utenti
- **PortfolioModel**: Portfolio investimenti
- **OrderModel**: Ordini buy/sell
- **Cache Models**: ETFPriceCache, StockPriceCache, ExchangeRateCache, RiskFreeRateCache, MarketBenchmarkCache

### Routers (API Endpoints)
24 endpoints organizzati per dominio:
- **auth.py** (3 endpoints): `/auth/register`, `/auth/login`, `/auth/me`
- **portfolios.py** (8 endpoints): CRUD portfolio + analytics avanzate + storico
- **orders.py** (5 endpoints): CRUD ordini + ottimizzazione portfolio (MPT)
- **symbols.py** (4 endpoints): Ricerca simboli, lista UCITS ETF, statistiche
- **market_data.py** (4 endpoints): Prezzi, tassi risk-free, benchmark

### Schemas (Pydantic)
Validazione e serializzazione request/response:
- **UserRegister**, **UserLogin**, **Token**
- **Portfolio** (con validazione campi)
- **Order**, **OptimizationRequest**

### Utils
Funzioni utility condivise:
- **database.py**: Connection pooling, migrations, retry logic
- **auth.py**: JWT tokens, password hashing (bcrypt)
- **dates.py**: Formatting ISO/DMY, date parsing
- **cache.py**: Cache invalidation helpers
- **pricing.py**: ETF/Stock pricing, conversions, risk-free rates, benchmarks (1,151 righe)
- **portfolio.py**: Portfolio calculations, XIRR, aggregations (452 righe)
- **symbols.py**: Symbol search and validation (86 righe)
- **helpers.py**: Data validation and normalization (25 righe)

## 📦 Frontend - Struttura Dettagliata

### Pages (Routing)

Ogni page rappresenta una "schermata" dell'applicazione:

- **AuthPage**: Gestione autenticazione (login/registrazione)
- **DashboardPage**: Vista principale con metriche, grafici performance, holdings
- **OrdersPage**: Creazione e gestione ordini di acquisto/vendita
- **AnalyzePage**: Analisi avanzate (correlazione, Monte Carlo, drawdown, risk metrics)
- **ComparePage**: Confronto con benchmark (coming soon)
- **OptimizePage**: Ottimizzazione portfolio con Modern Portfolio Theory

### Components

#### Componenti Comuni
- **Navbar**: Navigazione principale con logo e menu
- **PortfoliosList**: Gestione CRUD portfolio con settings avanzati
- **MetricCard**: Card informativa con tooltip per metriche di rischio

#### Charts (Grafici)
- **CorrelationHeatmap**: Matrice di correlazione asset
- **MonteCarloChart**: Simulazione Monte Carlo (95°, 50°, 5° percentile)
- **DrawdownChart**: Grafico drawdown massimo
- **AssetPerformanceChart**: Performance normalizzata per asset

#### Skeletons (Loading States)
- **PortfolioCardSkeleton**: Loading card portfolio
- **DashboardSkeleton**: Loading dashboard
- **AnalysisTabSkeleton**: Loading tab analisi

### Utils (Utility Functions)

- **currency.js**:
  - `getCurrencySymbol(currency)` - Simboli valute
  - `formatCurrencyValue(val, currency)` - Formattazione valori
  - `formatTerValue(val)` - Formattazione TER

- **dates.js**:
  - `parseDateDMY(value)` - Parse DD/MM/YYYY
  - `toISODateFromDMY(value)` - Conversione ISO format

- **cache.js**:
  - `invalidatePortfolioCache(portfolioId)` - Invalidazione cache

- **helpers.js**: Tutte le utility sopra re-esportate

## 🔄 Flusso Dati

```
App.jsx (State Management)
    ↓
    ├─→ AuthPage → Login/Register
    │
    ├─→ Navbar (navigation)
    │
    └─→ Pages (views)
         ├─→ DashboardPage → API → Charts + MetricCard
         ├─→ OrdersPage → API → Form + Table
         ├─→ AnalyzePage → API → Charts + MetricCard
         ├─→ ComparePage (placeholder)
         └─→ OptimizePage → API → Results
```

### State Management

Lo state globale è gestito in `App.jsx`:
- `token`: JWT token (localStorage)
- `user`: Dati utente corrente
- `currentView`: Vista attiva (portfolios|dashboard|orders|analyze|compare|optimize)
- `selectedPortfolio`: Portfolio selezionato
- `portfolios`: Lista tutti i portfolio
- `portfoliosLoading`: Loading state

## 🚀 Come Iniziare

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Il backend sarà disponibile su `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Il frontend sarà disponibile su `http://localhost:5173`

## 🛠️ Tecnologie Utilizzate

### Backend
- **FastAPI**: Framework web moderno per Python
- **SQLAlchemy**: ORM per database
- **Pydantic**: Validazione dati
- **NumPy/Pandas**: Analisi dati
- **yfinance**: Dati finanziari

### Frontend
- **React**: Library UI
- **Recharts**: Grafici e visualizzazioni
- **Tailwind CSS**: Styling
- **Lucide React**: Icons
- **Vite**: Build tool

## 📊 Funzionalità Principali

### Portfolio Management
- ✅ Creazione e gestione multipli portfolio
- ✅ Impostazione valuta di riferimento
- ✅ Configurazione risk-free rate e benchmark personalizzati
- ✅ Import ordini da CSV

### Orders Management
- ✅ Tracciamento ordini BUY/SELL
- ✅ Supporto ETF e Stock
- ✅ Autocomplete simboli con ricerca
- ✅ Calcolo automatico P&L

### Analytics
- ✅ **Risk Metrics**: Sharpe, Sortino, Max Drawdown, Volatilità
- ✅ **Correlation Analysis**: Matrice correlazione asset
- ✅ **Monte Carlo Simulation**: Proiezioni future con percentili
- ✅ **Performance Attribution**: Contributo per asset
- ✅ **Drawdown Analysis**: Analisi drawdown storici

### Portfolio Optimization
- ✅ Modern Portfolio Theory (Markowitz)
- ✅ Efficient Frontier
- ✅ Ottimizzazione Sharpe Ratio massimo

## 🎨 Best Practices Implementate

### Frontend
- ✅ **Separazione responsabilità**: Pages, Components, Utils
- ✅ **Component reusability**: Componenti riutilizzabili ben definiti
- ✅ **Loading states**: Skeleton screens per UX migliore
- ✅ **Cache management**: SessionStorage per performance
- ✅ **Error handling**: Gestione errori API consistente

### Organizzazione Codice
- ✅ File piccoli e focalizzati (50-200 righe vs 3000+)
- ✅ Import/Export centralizzati (pages/index.js)
- ✅ Naming conventions consistenti
- ✅ Commenti e documentazione

## 📝 Convenzioni di Codice

### Naming
- **Components**: PascalCase (es. `DashboardPage.jsx`)
- **Utils**: camelCase (es. `formatCurrency.js`)
- **Constants**: UPPER_SNAKE_CASE (es. `API_URL`)

### File Organization
- Un componente principale per file
- Export default per components principali
- Named exports per utils

### Imports Order
1. React e hooks
2. Librerie esterne (recharts, lucide-react)
3. Components locali
4. Utils e config
5. Styles (se presenti)

## 🔐 Sicurezza

- JWT authentication
- Token storage in localStorage
- API authorization headers
- Input validation (frontend + backend)

## 📈 Prossimi Sviluppi

- [ ] Completare ComparePage con confronto benchmark
- [ ] Aggiungere test unitari (Jest, React Testing Library)
- [ ] Implementare React Router per URL routing
- [ ] Aggiungere TypeScript
- [ ] Dark mode
- [ ] Export reports (PDF)
- [ ] Notifiche real-time

## 🤝 Contribuire

1. Fork del progetto
2. Crea un branch per la feature (`git checkout -b feature/AmazingFeature`)
3. Commit delle modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📄 Licenza

Questo progetto è privato e non ha una licenza pubblica.

---

**Sviluppato con ❤️ per il tracking efficiente dei portfolio di investimento**
