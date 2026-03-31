import re
from io import StringIO

import pandas as pd
import requests as http_requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from utils.database import get_db
from utils.etf_cache import ETF_UCITS_CACHE
from utils.bond_cache import BOND_METADATA_CACHE
from utils.supabase_client import get_supabase
from utils import search_symbol

router = APIRouter(prefix="/symbols", tags=["symbols"])

_JUSTETF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _scrape_justetf(isin: str) -> dict | None:
    """Scarica dati ETF da JustETF per un singolo ISIN."""
    url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
    try:
        r = http_requests.get(url, headers=_JUSTETF_HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        html = r.text

        # Nome ETF — proviamo in ordine di affidabilità:
        name = ""

        # 1. og:title: "EDEF – Amundi MSCI Europe Ex-UK Equity UCITS ETF | justETF"
        m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not m:
            m = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']', html, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            # Rimuovi tutto ciò che viene dopo il primo "|" (WKN, ISIN, "justETF")
            raw = raw.split("|")[0].strip()
            # Rimuovi il ticker iniziale "TICKER – " o "TICKER - "
            raw = re.sub(r'^[A-Z0-9]{1,10}\s*[–-]\s*', '', raw).strip()
            if len(raw) > 5:
                name = raw

        # 2. <h1> della pagina
        if not name:
            m = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
            if m:
                raw = re.sub(r'<[^>]+>', '', m.group(1)).strip()
                if len(raw) > 5:
                    name = raw

        # 3. <title> come ultimo fallback (formato: "TICKER | WKN | justETF")
        if not name:
            m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            if m:
                parts = [p.strip() for p in m.group(1).split("|")]
                parts = [p for p in parts if p.lower() not in ("justetf", "just etf", "") and not re.match(r'^[A-Z0-9]{4,8}$', p)]
                name = parts[0] if parts else ""

        # TER dalla pagina
        ter = None
        m2 = re.search(r"(?:TER|Total expense ratio)[^0-9]*(\d+[.,]\d+)\s*%", html, re.IGNORECASE)
        if m2:
            try:
                ter = float(m2.group(1).replace(",", "."))
            except Exception:
                pass

        # Tabella listings con pd.read_html
        listings = []
        try:
            tables = pd.read_html(StringIO(html))
            for t in tables:
                cols = {str(c).lower(): c for c in t.columns}
                if any("listing" in k for k in cols) and any("ticker" in k for k in cols):
                    listing_col = next(v for k, v in cols.items() if "listing" in k)
                    ticker_col = next(v for k, v in cols.items() if "ticker" in k)
                    currency_col = next((v for k, v in cols.items() if "currency" in k), None)
                    for _, row in t.iterrows():
                        ticker = str(row[ticker_col]).strip()
                        if not ticker or ticker in ("nan", "None", "-"):
                            continue
                        exchange = str(row[listing_col]).strip()
                        currency = str(row[currency_col]).strip() if currency_col else ""
                        if currency in ("nan", "None"):
                            currency = ""
                        listings.append({"ticker": ticker, "exchange": exchange, "currency": currency})
                    break
        except Exception:
            pass

        return {"isin": isin, "name": name, "ter": ter, "listings": listings}
    except Exception:
        return None


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


@router.get("/isin-lookup")
def isin_lookup(isin: str):
    """
    Cerca un ETF per ISIN: prima nella cache in-memory, poi scrapa JustETF.
    Se trovato su JustETF, persiste su Supabase e aggiunge alla cache in-memory.
    """
    isin = isin.upper().strip()

    # 1. Cache in-memory (include statica + ETF scoperti in precedenza)
    cached = [e for e in ETF_UCITS_CACHE if e.get("isin", "").upper() == isin]
    if cached:
        return {
            "source": "cache",
            "listings": [
                {"ticker": e["symbol"], "exchange": e.get("exchange", ""), "currency": e.get("currency", ""), "name": e.get("name", ""), "ter": e.get("ter")}
                for e in cached
            ],
        }

    # 2. Scraping JustETF
    data = _scrape_justetf(isin)
    if not data or not data.get("listings"):
        raise HTTPException(status_code=404, detail="ETF not found on JustETF")

    sb = get_supabase()
    for l in data["listings"]:
        # Cache in-memory
        ETF_UCITS_CACHE.append({
            "symbol": l["ticker"],
            "isin": isin,
            "name": data["name"],
            "currency": l["currency"],
            "exchange": l["exchange"],
            "ter": data.get("ter"),
            "ticker": l["ticker"],
            "type": "ETF",
        })
        # Persistenza Supabase (upsert su chiave isin+ticker+exchange)
        try:
            sb.table("etf_ucits_cache").upsert({
                "isin": isin,
                "ticker": l["ticker"],
                "exchange": l["exchange"],
                "name": data["name"],
                "currency": l["currency"],
                "ter": data.get("ter"),
            }, on_conflict="isin,ticker,exchange").execute()
        except Exception as e:
            print(f"[ETF cache] Supabase save failed (non-fatal): {e}")

    return {
        "source": "justetf",
        "listings": [
            {"ticker": l["ticker"], "exchange": l["exchange"], "currency": l["currency"], "name": data["name"], "ter": data.get("ter")}
            for l in data["listings"]
        ],
    }


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
        ucits_etfs = [dict(e, region="UCITS") for e in ETF_UCITS_CACHE]
        results.extend(ucits_etfs)

    return {
        "etfs": results,
        "count": len(results)
    }


@router.get("/bonds")
def get_bond_cache():
    """Ritorna la cache in-memoria dei bond (caricata da Supabase all'avvio)."""
    return {"results": BOND_METADATA_CACHE, "count": len(BOND_METADATA_CACHE)}


@router.get("/bond-lookup")
def bond_lookup(isin: str, db: Session = Depends(get_db)):
    """
    Cerca un bond per ISIN.
    Cascade: cache Supabase (24h) → Borsa Italiana MOT/EuroMOT/ExtraMOT → 404.
    """
    from utils.bond_scraper import scrape_borsa_italiana_bond
    from datetime import datetime, timezone, timedelta
    isin = isin.upper().strip()
    if not isin:
        raise HTTPException(status_code=400, detail="ISIN obbligatorio")

    sb = get_supabase()

    # 0. Cache Supabase — se il bond è già noto e i dati sono recenti (< 24h), ritorna subito
    try:
        cached = sb.table("bond_metadata_cache").select("*").eq("isin", isin).maybe_single().execute()
        if cached.data:
            row = cached.data
            updated_str = row.get("updated_at", "")
            if updated_str:
                updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                age = datetime.now(timezone.utc) - updated
                if age < timedelta(hours=24):
                    print(f"[Bond] {isin}: Supabase cache hit (age {int(age.total_seconds()//3600)}h)")
                    return {"isin": isin, "metadata": {k: v for k, v in row.items() if v is not None and k != "updated_at"}}
    except Exception as e:
        print(f"[Bond] Supabase cache read failed (non-fatal): {e}")

    meta: dict = {"isin": isin, "currency": "EUR"}

    # 1. Borsa Italiana — MOT/EuroMOT/ExtraMOT (bond italiani ed europei quotati a Milano)
    bi = scrape_borsa_italiana_bond(isin)
    if bi:
        for key in ("ytm_gross", "ytm_net", "accrued_gross", "accrued_net",
                    "duration", "coupon_annual", "coupon_frequency", "price", "name", "issuer"):
            if bi.get(key) is not None:
                meta[key] = bi[key]
        if bi.get("maturity_bi"):
            meta["maturity"] = bi["maturity_bi"]
        if bi.get("coupon_annual"):
            meta["coupon"] = bi["coupon_annual"]

    # Se nessuna fonte ha trovato nulla → 404
    if not meta.get("maturity") and not meta.get("coupon") and not meta.get("ytm_gross"):
        raise HTTPException(status_code=404, detail=f"Bond {isin} non trovato su Borsa Italiana")

    # 3. Persisti su Supabase bond_metadata_cache
    record = {
        "isin": isin,
        "name": meta.get("name", ""),
        "issuer": meta.get("issuer", ""),
        "coupon": meta.get("coupon") or meta.get("coupon_annual"),
        "maturity": meta.get("maturity"),
        "currency": meta.get("currency", "EUR"),
        "ytm_gross": meta.get("ytm_gross"),
        "ytm_net": meta.get("ytm_net"),
        "duration": meta.get("duration"),
        "coupon_frequency": meta.get("coupon_frequency"),
    }
    try:
        sb.table("bond_metadata_cache").upsert(record, on_conflict="isin").execute()
    except Exception as e:
        print(f"[Bond] Supabase save failed (non-fatal): {e}")

    # 4. Aggiorna cache in-memoria
    existing = next((b for b in BOND_METADATA_CACHE if b.get("isin") == isin), None)
    if existing:
        existing.update({k: v for k, v in record.items() if v is not None})
    else:
        BOND_METADATA_CACHE.append({k: v for k, v in record.items() if v is not None})

    return {"isin": isin, "metadata": meta}


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
        stats["ucits_etfs"] = len(ETF_UCITS_CACHE)
        for etf in ETF_UCITS_CACHE:
            curr = etf.get("currency", "Unknown")
            stats["currencies"][curr] = stats["currencies"].get(curr, 0) + 1

            dom = etf.get("domicile", "Unknown")
            stats["domiciles"][dom] = stats["domiciles"].get(dom, 0) + 1
    except ImportError:
        pass

    stats["total_etfs"] = stats["us_etfs"] + stats["ucits_etfs"]

    return stats
