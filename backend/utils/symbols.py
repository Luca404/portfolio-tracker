"""Symbol search and validation utilities."""

from fastapi import HTTPException
from etf_cache_ucits import ETF_UCITS_CACHE
from utils.pricing import fmp_get, FMP_STABLE_BASE


def search_symbol(symbol: str, instrument_type: str):
    """
    Search for a symbol in FMP or ETF cache.

    Args:
        symbol: Symbol to search for
        instrument_type: Type of instrument ("stock" or "etf")

    Returns:
        List of matching symbols with metadata
    """
    instrument_type = instrument_type.lower()

    if instrument_type == "stock":
        path = "search-symbol"
        params = {"query": symbol.upper(), "limit": 20}
        return fmp_get(path, params, base=FMP_STABLE_BASE)

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
