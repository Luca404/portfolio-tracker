"""
Bond data fetching: Borsa Italiana (spot price + metadata + history) → yfinance (fallback).

Borsa Italiana fornisce il Prezzo Ufficiale e diversi metadati utili anche per molti
bond europei quotati su MOT/EuroMOT/ExtraMOT.
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.orm import Session

from models.cache import BondPriceCacheModel
from utils.cache import (
    is_cache_data_fresh,
    merge_historical_data,
    get_latest_history_date,
    get_backfill_period_from_latest_date,
)
from utils.database import commit_with_retry
from utils.dates import DATE_FMT, parse_date_input

# ---------------------------------------------------------------------------
# Borsa Italiana: metadata + prezzi aggiuntivi (solo bond italiani)
# ---------------------------------------------------------------------------

_BI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_BI_CHART_CACHE: dict = {}
_BI_CHART_TTL_HOURS = 6

_BI_FIELD_MAP = {
    "Prezzo ufficiale":                              "price",
    "Rendimento effettivo a scadenza lordo":         "ytm_gross",
    "Rendimento effettivo a scadenza netto":         "ytm_net",
    "Rateo Lordo":                                   "accrued_gross",
    "Rateo Netto":                                   "accrued_net",
    "Duration modificata":                           "duration",
    "Scadenza":                                      "maturity_bi",
    "Tasso Cedola Periodale":                        "coupon_periodic",
    "Periodicità cedola":                            "coupon_frequency",
    "Emittente":                                     "issuer",
    "Stato Emittente":                               "issuer",
    "Denominazione":                                 "name",
}


def _parse_italian_number(s: str) -> Optional[float]:
    """Converte formato numerico italiano (virgola decimale) in float."""
    try:
        return float(s.strip().replace(".", "").replace(",", "."))
    except (ValueError, AttributeError):
        return None


def scrape_borsa_italiana_bond(isin: str) -> Optional[dict]:
    """
    Scrape della scheda obbligazione su Borsa Italiana.
    Supporta bond italiani (MOT) ed europei (EuroMOT/ExtraMOT).
    Restituisce: price, ytm_gross, ytm_net, accrued_gross, duration, maturity, coupon_frequency.
    """
    isin_up = isin.upper()
    base = "https://www.borsaitaliana.it"

    # Pattern URL da provare in ordine (MOT → EuroMOT → ExtraMOT)
    candidates = [
        f"{base}/borsa/obbligazioni/mot/btp/scheda/{isin_up}-MOTX.html?lang=it",
        f"{base}/borsa/obbligazioni/mot/altri-titoli-di-stato/scheda/{isin_up}-MOTX.html?lang=it",
        f"{base}/borsa/obbligazioni/mot/obbligazioni/scheda/{isin_up}-MOTX.html?lang=it",
        f"{base}/borsa/obbligazioni/euromot/titoli-di-stato-esteri/scheda/{isin_up}-XMOT.html?lang=it",
        f"{base}/borsa/obbligazioni/euromot/obbligazioni/scheda/{isin_up}-XMOT.html?lang=it",
        f"{base}/borsa/obbligazioni/extramot/scheda/{isin_up}-EXMX.html?lang=it",
    ]

    resp = None
    for url in candidates:
        try:
            r = requests.get(url, headers=_BI_HEADERS, timeout=10)
            if r.status_code == 200 and "scheda" in r.text and len(r.text) > 5000:
                resp = r
                break
        except Exception:
            continue

    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    result: dict = {}

    # Cerca ogni label nella tabella e prende il valore adiacente
    for strong in soup.find_all("strong"):
        label = (strong.get_text(strip=True)
                 .replace("\xa0", " ")
                 .replace("&agrave;", "à"))
        field = _BI_FIELD_MAP.get(label)
        if not field:
            continue
        td = strong.find_parent("td")
        if not td:
            continue
        next_td = td.find_next_sibling("td")
        if not next_td:
            continue
        raw_val = next_td.get_text(strip=True)
        result[field] = raw_val

    if not result:
        return None

    # Normalizza numerici
    parsed: dict = {"source": "borsa_italiana"}
    for field, raw in result.items():
        if field in ("maturity_bi", "coupon_frequency", "issuer", "name"):
            parsed[field] = raw.strip()
        else:
            parsed[field] = _parse_italian_number(raw)

    # Converti cedola periodale → annuale (solitamente semestrale → ×2)
    if parsed.get("coupon_periodic") and parsed.get("coupon_frequency", "").lower().startswith("semes"):
        parsed["coupon_annual"] = round(parsed["coupon_periodic"] * 2, 4)
    elif parsed.get("coupon_periodic"):
        parsed["coupon_annual"] = parsed["coupon_periodic"]

    # Estrai nome bond dal <title> (es: "BTP 1 MAR 2037 4% | Obbligazioni | Borsa Italiana" → "BTP 1 MAR 2037 4%")
    # o dall'<h1> della pagina, se non trovato nel field map
    if not parsed.get("name"):
        title_tag = soup.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            name_candidate = title_text.split("|")[0].strip()
            if name_candidate and name_candidate.upper() != isin_up:
                parsed["name"] = name_candidate
        if not parsed.get("name"):
            h1 = soup.find("h1")
            if h1:
                parsed["name"] = h1.get_text(strip=True).split("|")[0].strip()

    print(
        f"[Bond] {isin_up}: BI metadata OK "
        f"(price={parsed.get('price')}, ytm={parsed.get('ytm_gross')}, maturity={parsed.get('maturity_bi')})"
    )
    return parsed


def _get_borsa_chart_auth(isin: str) -> Optional[dict]:
    """Fetch token + exchcode from the public Borsa Italiana chart iframe."""
    isin_up = isin.upper()
    now = datetime.now(timezone.utc)
    cached = _BI_CHART_CACHE.get(isin_up)
    if cached and (now - cached["fetched_at"]).total_seconds() < _BI_CHART_TTL_HOURS * 3600:
        return cached["value"]

    iframe_candidates = [
        f"https://grafici.borsaitaliana.it/interactive-chart/{isin_up}-MOTX?lang=it",
        f"https://grafici.borsaitaliana.it/interactive-chart/{isin_up}-XMOT?lang=it",
        f"https://grafici.borsaitaliana.it/interactive-chart/{isin_up}-EXMX?lang=it",
    ]

    for url in iframe_candidates:
        try:
            resp = requests.get(url, headers=_BI_HEADERS, timeout=10)
            if resp.status_code != 200:
                continue

            token_match = re.search(r'token="([^"]+)"', resp.text)
            exch_match = re.search(r'exchcode="([^"]+)"', resp.text)
            code_match = re.search(r'code="([^"]+)"', resp.text)
            if not token_match or not exch_match or not code_match:
                continue

            value = {
                "token": token_match.group(1),
                "exchcode": exch_match.group(1),
                "code": code_match.group(1),
            }
            _BI_CHART_CACHE[isin_up] = {"value": value, "fetched_at": now}
            return value
        except Exception:
            continue

    return None


def fetch_borsa_italiana_bond_history(isin: str, period: str = "5Y") -> list:
    """Fetch bond history from the public Borsa Italiana interactive chart API."""
    auth = _get_borsa_chart_auth(isin)
    if not auth:
        return []

    url = (
        f"https://grafici.borsaitaliana.it/api/instruments/"
        f"{auth['code']},{auth['exchcode']},ISIN/history/period"
    )
    params = {
        "period": period,
        "adjustment": "true",
        "add-last-price": "true",
    }
    headers = {
        "Authorization": f"Bearer {auth['token']}",
        "User-Agent": _BI_HEADERS["User-Agent"],
        "Accept": "application/json, text/plain, */*",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        payload = resp.json()
        rows = (((payload or {}).get("history") or {}).get("historyDt")) or []
        history = []
        for row in rows:
            raw_date = row.get("dt", "")
            parsed_date = None
            if raw_date and len(str(raw_date)) == 8 and str(raw_date).isdigit():
                try:
                    parsed_date = datetime.strptime(str(raw_date), "%Y%m%d").date()
                except ValueError:
                    parsed_date = None
            if parsed_date is None:
                parsed_date = parse_date_input(raw_date)
            price = row.get("closePx")
            if parsed_date and price is not None:
                history.append({"date": parsed_date.strftime(DATE_FMT), "price": float(price)})
        history.sort(key=lambda x: x["date"])
        if history:
            print(f"[Bond] {isin}: BI history {len(history)} days (period={period})")
        return history
    except Exception:
        return []


# ---------------------------------------------------------------------------
# yfinance fallback
# ---------------------------------------------------------------------------

def fetch_yfinance_bond(isin: str) -> Optional[dict]:
    """Fallback: prova yfinance con vari suffix di exchange."""
    country_suffix = f".{isin[:2].upper()}" if len(isin) >= 2 else ""
    suffixes = []
    for suffix in [country_suffix, ".TI", ".F", ".MI", ".PA", ".AS", ""]:
        if suffix not in suffixes:
            suffixes.append(suffix)
    for suffix in suffixes:
        ticker_str = f"{isin}{suffix}"
        try:
            tk = yf.Ticker(ticker_str)
            hist = tk.history(period="2y")
            if hist.empty:
                continue
            history = [
                {"date": d.strftime(DATE_FMT), "price": float(row["Close"])}
                for d, row in hist.iterrows()
                if row["Close"] is not None
            ]
            if not history:
                continue
            last_price = history[-1]["price"]
            print(f"[Bond] {isin}: yfinance fallback OK with {ticker_str} ({len(history)} days)")
            return {"last_price": last_price, "history": history}
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Funzione principale con cache SQLite
# ---------------------------------------------------------------------------

def get_bond_price_and_history(isin: str, db: Session, days: int = 730) -> dict:
    """
    Restituisce {last_price, history, metadata} per un bond, con cache SQLite.

    Cascade prezzi:  Borsa Italiana → Frankfurt → yfinance
    Metadata extra:  Borsa Italiana + Frankfurt
    """
    isin = isin.upper()
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # --- Leggi cache SQLite ---
    cache = db.execute(
        select(BondPriceCacheModel).where(BondPriceCacheModel.isin == isin)
    ).scalar_one_or_none()

    cached_history: list = []
    cached_price: float = 0.0
    cached_meta: dict = {}

    if cache:
        cached_history = json.loads(cache.history_json or "[]")
        cached_price = cache.last_price or (cached_history[-1]["price"] if cached_history else 0.0)
        cached_meta = json.loads(cache.metadata_json or "{}")
        history_is_too_short = len(cached_history) < 5
        if is_cache_data_fresh(cached_history, cache.updated_at) and cached_price and not history_is_too_short:
            print(f"[Bond] {isin}: cache hit ({len(cached_history)} days)")
            return {"last_price": cached_price, "history": cached_history, "metadata": cached_meta}
        if cached_history:
            print(f"[Bond] {isin}: cache refresh ({len(cached_history)} days cached)")

    # --- Borsa Italiana: prezzo spot + metadata utili ---
    bi_data = scrape_borsa_italiana_bond(isin)
    meta = cached_meta.copy()
    if bi_data:
        for key in (
            "ytm_gross",
            "ytm_net",
            "accrued_gross",
            "accrued_net",
            "duration",
            "coupon_annual",
            "coupon_frequency",
            "maturity_bi",
            "name",
            "issuer",
        ):
            if bi_data.get(key) is not None:
                meta[key] = bi_data[key]

    # --- Storico prezzi: Borsa Italiana → yfinance ---
    new_history: list = []
    last_price: float = 0.0
    latest_cached_date = get_latest_history_date(cached_history)

    if bi_data and bi_data.get("price") is not None:
        last_price = float(bi_data["price"])
        today_point = {"date": datetime.now(timezone.utc).date().strftime(DATE_FMT), "price": last_price}
        new_history = [today_point]
        print(f"[Bond] {isin}: BI spot price {last_price:.4f}")

    # New policy:
    # - first fetch: grab the widest useful history
    # - subsequent refreshes: request only the recent missing tail
    bi_period = get_backfill_period_from_latest_date(latest_cached_date)
    borsa_history = fetch_borsa_italiana_bond_history(isin, period=bi_period)
    if borsa_history:
        new_history = borsa_history
        last_price = borsa_history[-1]["price"]

    if not new_history:
        yf_data = fetch_yfinance_bond(isin)
        if yf_data:
            yf_history = yf_data.get("history", [])
            if yf_history:
                new_history = yf_history
            if not last_price:
                last_price = yf_data.get("last_price", 0.0)

    if not new_history and cached_history:
        print(f"[Bond] {isin}: all sources failed, using stale cache")
        return {"last_price": cached_price, "history": cached_history, "metadata": cached_meta}

    if not new_history:
        raise ValueError(f"Nessun dato prezzo disponibile per il bond {isin}")

    merged_history = merge_historical_data(cached_history, new_history)
    print(f"[Bond] {isin}: history ready ({len(merged_history)} points)")

    # --- Metadata: Borsa Italiana come fonte primaria ---

    # --- Aggiorna cache SQLite ---
    meta_json = json.dumps(meta)
    if cache:
        cache.last_price = last_price
        cache.history_json = json.dumps(merged_history)
        cache.metadata_json = meta_json
        cache.updated_at = now
    else:
        db.add(BondPriceCacheModel(
            isin=isin,
            last_price=last_price,
            history_json=json.dumps(merged_history),
            metadata_json=meta_json,
            updated_at=now,
        ))
    commit_with_retry(db)

    return {"last_price": last_price, "history": merged_history, "metadata": meta}
