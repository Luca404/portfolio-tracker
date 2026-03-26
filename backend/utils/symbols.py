"""Symbol search and validation utilities."""

from fastapi import HTTPException
from utils.etf_cache import ETF_UCITS_CACHE
from utils.pricing import fmp_get, FMP_STABLE_BASE


def _persist_stock_symbols(results: list):
    """Saves new stock symbols to Supabase and the in-memory cache."""
    from utils.stock_cache import STOCK_SYMBOL_CACHE
    from utils.supabase_client import get_supabase

    existing = {s["symbol"] for s in STOCK_SYMBOL_CACHE}
    new_entries = []
    for s in results:
        sym = (s.get("symbol") or "").upper()
        if not sym or sym in existing:
            continue
        entry = {
            "symbol": sym,
            "name": s.get("name", ""),
            "exchange": s.get("exchangeShortName") or s.get("exchange", ""),
            "currency": s.get("currency", ""),
        }
        STOCK_SYMBOL_CACHE.append(entry)
        existing.add(sym)
        new_entries.append(entry)

    if new_entries:
        try:
            get_supabase().table("stock_symbol_cache").upsert(
                new_entries, on_conflict="symbol"
            ).execute()
            print(f"[Stock cache] saved {len(new_entries)} new symbols to Supabase")
        except Exception as e:
            print(f"[Stock cache] Supabase save failed (non-fatal): {e}")


def search_symbol(symbol: str, instrument_type: str):
    """
    Search for a symbol in local cache or FMP.

    Args:
        symbol: Symbol to search for
        instrument_type: Type of instrument ("stock" or "etf")

    Returns:
        List of matching symbols with metadata
    """
    instrument_type = instrument_type.lower()

    if instrument_type == "stock":
        from utils.stock_cache import STOCK_SYMBOL_CACHE
        q = symbol.upper().strip()

        # 1. Cerca nella cache in-memory
        cached = [
            s for s in STOCK_SYMBOL_CACHE
            if s.get("symbol", "").upper().startswith(q)
            or q in s.get("name", "").upper()
        ][:20]
        if cached:
            print(f"[Stock cache] '{q}': {len(cached)} hits from cache")
            return cached

        # 2. Fallback FMP, poi persisti i risultati
        print(f"[Stock cache] '{q}': cache miss, calling FMP")
        results = fmp_get("search-symbol", {"query": q, "limit": 20}, base=FMP_STABLE_BASE)
        if results:
            _persist_stock_symbols(results)
        return results

    if instrument_type == "etf":
        query = symbol.upper().strip()
        if not query:
            return []
        matches = []
        for item in ETF_UCITS_CACHE:
            tickers = [item.get("ticker", ""), item.get("symbol", "")]
            # match ticker prefix/exact
            if any(query and t and t.upper().startswith(query) for t in tickers):
                sym = (item.get("ticker") or item.get("symbol") or item.get("isin") or "").upper()
                matches.append({
                    "symbol": sym,
                    "name": item.get("name", ""),
                    "exchange": item.get("exchange", "") or item.get("domicile", ""),
                    "currency": item.get("currency", ""),
                    "type": "ETF",
                    "isin": item.get("isin", ""),
                    "ticker": item.get("ticker", ""),
                    "ter": item.get("ter", ""),
                })
                if len(matches) >= 25:
                    break
            # match ISIN esatto (ISIN = 12 chars)
            elif len(query) == 12 and query == str(item.get("isin", "")).upper():
                sym = (item.get("ticker") or item.get("symbol") or item.get("isin") or "").upper()
                matches.append({
                    "symbol": sym,
                    "name": item.get("name", ""),
                    "exchange": item.get("exchange", "") or item.get("domicile", ""),
                    "currency": item.get("currency", ""),
                    "type": "ETF",
                    "isin": item.get("isin", ""),
                    "ticker": item.get("ticker", ""),
                    "ter": item.get("ter", ""),
                })
                if len(matches) >= 25:
                    break
        return matches

    return []


def ensure_symbol_exists(symbol: str, instrument_type: str):
    """
    Validate that a symbol exists and return its metadata.

    Args:
        symbol: Symbol to validate
        instrument_type: Type of instrument ("stock" or "etf")

    Returns:
        Symbol metadata dict

    Raises:
        HTTPException: If symbol not found
    """
    matches = search_symbol(symbol, instrument_type)
    for m in matches:
        if m.get("symbol", "").upper() == symbol.upper():
            return m
    raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found for type {instrument_type}")
