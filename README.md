# Portfolio Tracker

A full-stack web application for tracking and analyzing investment portfolios with advanced risk analytics and portfolio optimization capabilities.

## Architecture Overview

Portfolio Tracker is a modern web application built with a clear separation between backend and frontend layers. The backend provides a RESTful API powered by FastAPI, while the frontend delivers a responsive React-based user interface.

### Technology Stack

**Backend:**
- FastAPI (Python web framework)
- SQLAlchemy 2.0 (ORM with SQLite)
- Pydantic (data validation and serialization)
- yfinance (market data)
- NumPy/Pandas (numerical analysis)
- PyPortfolioOpt (Modern Portfolio Theory implementation)
- JWT + bcrypt (authentication and security)

**Frontend:**
- React 18 (UI library)
- Vite (build tool and development server)
- Tailwind CSS (styling framework)
- Recharts (data visualization)
- Lucide React (icon library)

## Project Structure

```
portfolio-tracker/
├── backend/               # FastAPI backend
│   ├── main.py           # Application setup and configuration
│   ├── models/           # SQLAlchemy ORM models
│   │   ├── user.py      # User authentication model
│   │   ├── portfolio.py # Portfolio model
│   │   ├── order.py     # Order model
│   │   └── cache.py     # Cache models (price, exchange rates, benchmarks)
│   │
│   ├── routers/          # API endpoint handlers (24 endpoints total)
│   │   ├── auth.py          # Authentication (register, login, user info)
│   │   ├── portfolios.py    # Portfolio CRUD and analytics
│   │   ├── orders.py        # Order management and optimization
│   │   ├── symbols.py       # Symbol search and ETF listings
│   │   └── market_data.py   # Market data, rates, and benchmarks
│   │
│   ├── schemas/          # Pydantic validation schemas
│   │   ├── user.py      # User registration, login, token
│   │   ├── portfolio.py # Portfolio schema
│   │   └── order.py     # Order and optimization schemas
│   │
│   ├── utils/            # Business logic and utilities
│   │   ├── database.py  # Database connection and migrations
│   │   ├── auth.py      # JWT and password hashing
│   │   ├── pricing.py   # Pricing engine (ETF, stocks, currencies)
│   │   ├── portfolio.py # Portfolio calculations and metrics
│   │   ├── symbols.py   # Symbol search and validation
│   │   ├── dates.py     # Date formatting and parsing
│   │   ├── cache.py     # Cache invalidation helpers
│   │   └── helpers.py   # Data validation utilities
│   │
│   └── etf_cache_ucits.py # UCITS ETF data cache
│
├── frontend/             # React frontend
│   ├── public/          # Static assets
│   └── src/
│       ├── App.jsx                    # Root component with state management
│       ├── main.jsx                   # Application entry point
│       ├── config.js                  # Configuration (API URL)
│       │
│       ├── pages/                     # Page components
│       │   ├── index.js              # Centralized exports
│       │   ├── AuthPage.jsx          # Login and registration
│       │   ├── DashboardPage.jsx     # Portfolio dashboard
│       │   ├── OrdersPage.jsx        # Order management
│       │   ├── AnalyzePage.jsx       # Advanced analytics
│       │   ├── ComparePage.jsx       # Benchmark comparison
│       │   └── OptimizePage.jsx      # Portfolio optimization
│       │
│       ├── components/                # Reusable components
│       │   ├── Navbar.jsx            # Navigation bar
│       │   ├── PortfoliosList.jsx    # Portfolio list and CRUD
│       │   ├── MetricCard.jsx        # Metric display cards
│       │   │
│       │   ├── charts/               # Chart components
│       │   │   ├── CorrelationHeatmap.jsx
│       │   │   ├── MonteCarloChart.jsx
│       │   │   ├── DrawdownChart.jsx
│       │   │   └── AssetPerformanceChart.jsx
│       │   │
│       │   └── skeletons/            # Loading state components
│       │       ├── PortfolioCardSkeleton.jsx
│       │       ├── DashboardSkeleton.jsx
│       │       └── AnalysisTabSkeleton.jsx
│       │
│       ├── services/                  # API client layer
│       │   └── api.js                # Centralized API calls
│       │
│       └── utils/                     # Frontend utilities
│           ├── currency.js           # Currency formatting
│           ├── dates.js              # Date parsing
│           ├── cache.js              # Cache management
│           └── helpers.js            # General utilities
│
└── scripts/              # Utility scripts
    ├── etf_cache.py     # ETF data cache builder
    └── import_orders_from_csv.py # CSV import tool
```

## Backend Architecture

### Database Schema

The application uses SQLAlchemy ORM with SQLite for data persistence:

- **UserModel**: User authentication (email, username, hashed password)
- **PortfolioModel**: Investment portfolios with multi-currency support
- **OrderModel**: Buy/sell orders for ETFs and stocks
- **Cache Models**: Performance optimization caches
  - ETFPriceCache
  - StockPriceCache
  - ExchangeRateCache
  - RiskFreeRateCache
  - MarketBenchmarkCache

### API Endpoints

The API provides 24 RESTful endpoints organized by domain:

**Authentication** (`/auth/*` - 3 endpoints):
- `POST /auth/register` - User registration
- `POST /auth/login` - User authentication
- `GET /auth/me` - Current user information

**Portfolios** (`/portfolios/*` - 8 endpoints):
- `GET /portfolios` - List all user portfolios
- `GET /portfolios/count` - Portfolio count
- `GET /portfolios/{id}` - Portfolio details with positions and performance
- `POST /portfolios` - Create portfolio
- `PUT /portfolios/{id}` - Update portfolio
- `DELETE /portfolios/{id}` - Delete portfolio
- `GET /portfolios/history/{id}/{symbol}` - Position history
- `GET /portfolios/analysis/{id}` - Advanced analytics

**Orders** (`/orders/*` - 5 endpoints):
- `GET /orders/{portfolio_id}` - List orders
- `POST /orders` - Create order
- `PUT /orders/{id}` - Update order
- `DELETE /orders/{id}` - Delete order
- `POST /orders/optimize` - Portfolio optimization

**Symbols** (`/symbols/*` - 4 endpoints):
- `GET /symbols/search` - Search stocks and ETFs
- `GET /symbols/ucits` - UCITS ETF list
- `GET /symbols/etf-list` - Complete ETF list
- `GET /symbols/etf-stats` - ETF cache statistics

**Market Data** (`/market-data/*` - 4 endpoints):
- `GET /market-data/{symbol}` - Symbol market data
- `GET /market-data/risk-free-rate/{currency}` - Risk-free rates
- `GET /market-data/benchmark/{currency}` - Market benchmarks
- `GET /market-data/portfolio-context/{id}` - Portfolio-specific context

### Business Logic

Core business logic is implemented in utility modules:

- **pricing.py**: Pricing engine for ETFs, stocks, currency conversion, and market data fetching
- **portfolio.py**: Portfolio calculations including XIRR, returns, risk metrics, and performance attribution
- **symbols.py**: Symbol search and validation logic
- **auth.py**: JWT token management and password hashing
- **database.py**: Connection pooling, retry logic, and schema migrations

### Caching Strategy

The application implements multi-level caching for optimal performance:

- Database-level caching for API responses
- Session storage caching in the frontend
- Intelligent cache invalidation on data changes
- Configurable cache freshness checks

## Frontend Architecture

### State Management

Global state is managed in `App.jsx`:

- `token`: JWT authentication token (persisted in localStorage)
- `user`: Current user data
- `currentView`: Active page view
- `selectedPortfolio`: Currently selected portfolio
- `portfolios`: List of all portfolios
- `portfoliosLoading`: Loading state indicator

### Page Components

- **AuthPage**: User authentication and registration
- **DashboardPage**: Portfolio overview with metrics, charts, and holdings
- **OrdersPage**: Order creation and management interface
- **AnalyzePage**: Advanced analytics including correlation, Monte Carlo simulations, and risk metrics
- **ComparePage**: Benchmark comparison (in development)
- **OptimizePage**: Portfolio optimization using Modern Portfolio Theory

### Reusable Components

- **Navbar**: Application navigation
- **PortfoliosList**: Portfolio CRUD operations
- **MetricCard**: Metric display with tooltips
- **Charts**: Correlation heatmaps, Monte Carlo simulations, drawdown analysis, performance charts
- **Skeletons**: Loading state placeholders for improved UX

### Service Layer

The `services/api.js` module provides a centralized API client with consistent error handling and authentication header management.

## Features

### Portfolio Management

- Multiple portfolio support with custom names and descriptions
- Configurable reference currency (EUR, USD, GBP, etc.)
- Custom risk-free rate sources
- Custom market benchmarks
- CSV import for bulk order creation

### Order Tracking

- Buy/sell order management
- Support for both ETFs and stocks
- Automatic symbol validation and enrichment
- Autocomplete search across 1000+ ETFs
- Commission tracking
- Position validation

### Performance Analytics

- Portfolio value tracking with historical data
- Gain/loss calculation (absolute and percentage)
- XIRR (money-weighted return)
- Time-weighted return
- Asset composition analysis

### Risk Analytics

- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted return)
- Maximum Drawdown
- Volatility (standard deviation)
- Correlation matrix
- Monte Carlo simulation with confidence intervals
- Drawdown analysis
- Performance attribution by asset

### Portfolio Optimization

- Modern Portfolio Theory (Markowitz) implementation
- Efficient Frontier calculation
- Maximum Sharpe Ratio optimization
- Discrete allocation (whole shares)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with required configuration:
```bash
SECRET_KEY=your-secret-key-here
FMP_API_KEY=your-fmp-api-key  # Optional: For enhanced symbol search
```

5. Start the development server:
```bash
uvicorn main:app --reload
```

The backend API will be available at `http://localhost:8000`

API documentation is automatically generated and accessible at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Configure the API URL in `src/config.js`:
```javascript
export const API_URL = 'http://localhost:8000';
```

4. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Production Build

**Backend:**
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production ASGI server
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

**Frontend:**
```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Configuration

### Environment Variables

**Backend** (`.env` file):
- `SECRET_KEY`: JWT secret key (required)
- `FMP_API_KEY`: Financial Modeling Prep API key (optional)

**Frontend** (`src/config.js`):
- `API_URL`: Backend API URL

### Database

The application uses SQLite by default. The database file is created automatically at `backend/portfolio.db`.

For production deployments, consider migrating to PostgreSQL or MySQL for better concurrency support.

## Security

- JWT-based authentication
- bcrypt password hashing
- SQL injection protection via SQLAlchemy ORM
- Input validation using Pydantic schemas
- CORS configuration
- Secure token storage

## API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API testing interface
- **ReDoc**: `http://localhost:8000/redoc` - API reference documentation

## Development Notes

### Code Organization

The codebase follows modern software engineering practices:

- Modular architecture with clear separation of concerns
- Small, focused files (most under 200 lines)
- Consistent naming conventions (PascalCase for components, camelCase for utilities)
- Centralized imports and exports
- Type validation using Pydantic schemas

### Refactoring History

The backend was refactored from a monolithic 3,244-line file into a modular architecture with dedicated modules for models, routers, schemas, and utilities. This improves maintainability, testability, and scalability.

### Performance Optimizations

- Connection pooling for database operations
- Multi-level caching strategy
- Lazy loading of analytics data
- Retry logic for transient failures
- Efficient data serialization

## Testing

To run tests (when implemented):

**Backend:**
```bash
pytest
```

**Frontend:**
```bash
npm test
```

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is private and does not have a public license.

## Acknowledgments

Built with modern web technologies and financial analysis libraries to provide professional-grade portfolio tracking and analysis capabilities.
