"""In-memory cache for stock symbol metadata (name, exchange, currency).

Loaded from Supabase `stock_symbol_cache` at startup.
Populated on-demand when users search for stocks via /symbols/search.
"""

STOCK_SYMBOL_CACHE: list = []
