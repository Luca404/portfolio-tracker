from fastapi import APIRouter

from etf_cache_ucits import ETF_UCITS_CACHE
from utils import search_symbol

router = APIRouter(prefix="/symbols", tags=["symbols"])


@router.get("/search")
def symbols_search(q: str, instrument_type: str = "stock"):
    """Search for symbols by query."""
    matches = search_symbol(q, instrument_type)
    formatted = [
        {
            "symbol": item.get("symbol"),
            "name": item.get("name"),
            "exchange": item.get("exchangeShortName") or item.get("exchange"),
            "currency": item.get("currency"),
            "type": item.get("type"),
            "ter": item.get("ter", ""),
        }
        for item in matches
    ]
    return {"results": formatted}


@router.get("/ucits")
def symbols_ucits():
    """Get all UCITS ETFs from cache."""
    formatted = []
    for item in ETF_UCITS_CACHE:
        sym = (item.get("ticker") or item.get("symbol") or item.get("isin") or "").upper()
        formatted.append({
            "symbol": sym,
            "name": item.get("name", ""),
            "exchange": item.get("exchange", "") or item.get("domicile", ""),
            "currency": item.get("currency", ""),
            "type": "ETF",
            "isin": item.get("isin"),
            "ticker": item.get("ticker"),
            "ter": item.get("ter", ""),
        })
    return {"results": formatted}


@router.get("/etf-list")
def get_etf_list(etf_type: str = "all"):
    """
    Get complete list of ETFs from local cache.

    Args:
        etf_type: "us", "ucits", or "all" (default)

    This endpoint doesn't require external API calls.
    """
    results = []

    if etf_type in ["us", "all"]:
        from etf_cache import get_all_etfs
        us_etfs = get_all_etfs()
        # Add tag to identify type
        for etf in us_etfs:
            etf["region"] = "US"
        results.extend(us_etfs)

    if etf_type in ["ucits", "all"]:
        try:
            from etf_cache_ucits import get_all_ucits_etfs
            ucits_etfs = get_all_ucits_etfs()
            # Add tag to identify type
            for etf in ucits_etfs:
                etf["region"] = "UCITS"
            results.extend(ucits_etfs)
        except ImportError:
            pass

    return {
        "etfs": results,
        "count": len(results)
    }


@router.get("/etf-stats")
def get_etf_stats():
    """Get statistics about the ETF cache."""
    stats = {
        "us_etfs": 0,
        "ucits_etfs": 0,
        "currencies": {},
        "domiciles": {}
    }

    try:
        from etf_cache import get_all_etfs
        us_etfs = get_all_etfs()
        stats["us_etfs"] = len(us_etfs)

        for etf in us_etfs:
            curr = etf.get("currency", "Unknown")
            stats["currencies"][curr] = stats["currencies"].get(curr, 0) + 1
    except ImportError:
        pass

    try:
        from etf_cache_ucits import get_all_ucits_etfs
        ucits_etfs = get_all_ucits_etfs()
        stats["ucits_etfs"] = len(ucits_etfs)

        for etf in ucits_etfs:
            curr = etf.get("currency", "Unknown")
            stats["currencies"][curr] = stats["currencies"].get(curr, 0) + 1

            dom = etf.get("domicile", "Unknown")
            stats["domiciles"][dom] = stats["domiciles"].get(dom, 0) + 1
    except ImportError:
        pass

    stats["total_etfs"] = stats["us_etfs"] + stats["ucits_etfs"]

    return stats
