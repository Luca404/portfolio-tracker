from .database import engine, SessionLocal, get_db, commit_with_retry, run_migrations
from .auth import verify_password, get_password_hash, create_access_token, verify_token, security
from .dates import DATE_FMT, format_date, format_datetime, parse_date_input
from .cache import is_cache_data_fresh, merge_historical_data, MAX_DAYS_STALE
from .pricing import (
    get_etf_price_and_history,
    get_stock_price_from_alphavantage,
    get_stock_price_from_yfinance,
    get_stock_price_and_history,
    get_stock_price_and_history_cached,
    convert_to_reference_currency,
    get_exchange_rate_history,
    fetch_us_treasury_rate,
    fetch_ecb_rate,
    fetch_market_benchmark,
    get_risk_free_rate,
    get_market_benchmark_data,
    normalize_chart_history,
)
from .portfolio import (
    aggregate_positions,
    compute_portfolio_value,
    aggregate_portfolio_history,
    compute_portfolio_performance,
    xnpv,
    calc_xirr,
)
from .symbols import search_symbol, ensure_symbol_exists
from .helpers import validate_order_input

__all__ = [
    # Database
    "engine",
    "SessionLocal",
    "get_db",
    "commit_with_retry",
    "run_migrations",
    # Auth
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "verify_token",
    "security",
    # Dates
    "DATE_FMT",
    "format_date",
    "format_datetime",
    "parse_date_input",
    # Cache
    "is_cache_data_fresh",
    "merge_historical_data",
    "MAX_DAYS_STALE",
    # Pricing
    "get_etf_price_and_history",
    "get_stock_price_from_alphavantage",
    "get_stock_price_from_yfinance",
    "get_stock_price_and_history",
    "get_stock_price_and_history_cached",
    "convert_to_reference_currency",
    "get_exchange_rate_history",
    "fetch_us_treasury_rate",
    "fetch_ecb_rate",
    "fetch_market_benchmark",
    "get_risk_free_rate",
    "get_market_benchmark_data",
    "normalize_chart_history",
    # Portfolio
    "aggregate_positions",
    "compute_portfolio_value",
    "aggregate_portfolio_history",
    "compute_portfolio_performance",
    "xnpv",
    "calc_xirr",
    # Symbols
    "search_symbol",
    "ensure_symbol_exists",
    # Helpers
    "validate_order_input",
]
