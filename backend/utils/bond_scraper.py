"""
Bond data fetching: Börse Frankfurt (primary) → Borsa Italiana (metadata IT) → yfinance (fallback).

Prezzi Frankfurt restituiti in % del nominale (tradedInPercent=true).
Borsa Italiana aggiunge YTM, duration e rateo cedolare per i BTP italiani.
"""

import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.orm import Session

from models.cache import BondPriceCacheModel
from utils.cache import is_cache_data_fresh, merge_historical_data
from utils.database import commit_with_retry
from utils.dates import DATE_FMT, parse_date_input

# ---------------------------------------------------------------------------
# Börse Frankfurt — headers dinamici
# ---------------------------------------------------------------------------

_FRANKFURT_BASE = "https://api.boerse-frankfurt.de/v1/data"
_FRANKFURT_ORIGIN = "https://www.boerse-frankfurt.de"
_SALT_CACHE: dict = {"value": None, "fetched_at": None}
_SALT_TTL_HOURS = 6  # ricarica il salt ogni 6 ore


def _get_frankfurt_salt() -> str:
    """Estrae il salt dal bundle JS di Börse Frankfurt (cache in memoria 6h)."""
    now = datetime.now(timezone.utc)
    cached = _SALT_CACHE
    if cached["value"] and cached["fetched_at"] and (now - cached["fetched_at"]).total_seconds() < _SALT_TTL_HOURS * 3600:
        return cached["value"]
    try:
        # Trova il nome del bundle JS principale
        r = requests.get(_FRANKFURT_ORIGIN, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        match = re.search(r'src="(/main\.[a-f0-9]+\.js)"', r.text)
        if not match:
            raise ValueError("JS bundle not found")
        js_url = _FRANKFURT_ORIGIN + match.group(1)
        js_r = requests.get(js_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        salt_match = re.search(r'salt:"([a-f0-9]{30,40})"', js_r.text)
        if not salt_match:
            raise ValueError("salt not found in JS bundle")
        salt = salt_match.group(1)
        _SALT_CACHE["value"] = salt
        _SALT_CACHE["fetched_at"] = now
        print(f"[Bond] Frankfurt salt refreshed: {salt[:8]}...")
        return salt
    except Exception as e:
        print(f"[Bond] Frankfurt salt fetch failed: {e}")
        # Fallback: salt noto — aggiornare se smette di funzionare
        fallback = "af5a8d16eb5dc49f8a72b26fd9185475c7a"
        _SALT_CACHE["value"] = fallback
        _SALT_CACHE["fetched_at"] = now
        return fallback


def _frankfurt_headers(url: str) -> dict:
    """Calcola gli header dinamici richiesti dall'API di Börse Frankfurt."""
    salt = _get_frankfurt_salt()
    # Usa ora tedesca (CET/CEST) — Frankfurt valida x-security su ora locale
    import zoneinfo
    berlin = zoneinfo.ZoneInfo("Europe/Berlin")
    now_berlin = datetime.now(berlin)
    now_utc = datetime.now(timezone.utc)

    ms = now_utc.microsecond // 1000
    client_date = now_utc.strftime(f"%Y-%m-%dT%H:%M:%S.{ms:03d}Z")
    x_security = hashlib.md5(now_berlin.strftime("%Y%m%d%H%M").encode()).hexdigest()
    x_traceid = hashlib.md5((client_date + url + salt).encode()).hexdigest()
    return {
        "authority": "api.boerse-frankfurt.de",
        "origin": _FRANKFURT_ORIGIN,
        "referer": _FRANKFURT_ORIGIN + "/",
        "accept": "application/json, text/plain, */*",
        "accept-language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "client-date": client_date,
        "x-client-traceid": x_traceid,
        "x-security": x_security,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    }


def _frankfurt_get(path: str, params: dict) -> Optional[dict]:
    url = f"{_FRANKFURT_BASE}/{path}"
    full_url = url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    try:
        resp = requests.get(url, params=params, headers=_frankfurt_headers(full_url), timeout=15)
        if resp.status_code == 200:
            return resp.json()
        print(f"[Bond] Frankfurt {path} → HTTP {resp.status_code}")
        return None
    except Exception as e:
        print(f"[Bond] Frankfurt {path} error: {e}")
        return None


# ---------------------------------------------------------------------------
# Frankfurt: metadati bond
# ---------------------------------------------------------------------------

def fetch_frankfurt_bond_metadata(isin: str) -> Optional[dict]:
    """Restituisce cedola, scadenza, emittente e valuta dal master_data_bond di Frankfurt."""
    data = _frankfurt_get("master_data_bond", {"isin": isin.upper(), "mic": "XFRA"})
    if not data:
        return None
    maturity = data.get("maturity")
    return {
        "isin": isin.upper(),
        "issuer": data.get("issuer", ""),
        "coupon": data.get("cupon"),          # può essere null per alcuni bond
        "maturity": maturity,                  # "YYYY-MM-DD"
        "currency": data.get("issueCurrency", "EUR"),
        "issue_date": data.get("issueDate"),
        "min_investment": data.get("minimumInvestmentAmount", 1000.0),
    }


# ---------------------------------------------------------------------------
# Frankfurt: storico prezzi
# ---------------------------------------------------------------------------

def fetch_frankfurt_bond_history(isin: str, days: int = 730) -> list:
    """
    Scarica lo storico OHLCV da Frankfurt.
    I prezzi sono in % del nominale (tradedInPercent=true).
    Restituisce lista [{date, price}] dove price = close in % del par.
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    data = _frankfurt_get("price_history", {
        "isin": isin.upper(),
        "mic": "XFRA",
        "minDate": start.isoformat(),
        "maxDate": end.isoformat(),
        "limit": days,
        "cleanSplit": "false",
        "cleanPayout": "false",
        "cleanSubscription": "false",
    })
    if not data or not data.get("data"):
        return []
    history = []
    for row in data["data"]:
        raw_date = row.get("date", "")
        close = row.get("close")
        if not raw_date or close is None:
            continue
        d = parse_date_input(raw_date)
        if d:
            history.append({"date": d.strftime(DATE_FMT), "price": float(close)})
    history.sort(key=lambda x: x["date"])
    print(f"[Bond] Frankfurt history {isin}: {len(history)} days")
    return history


# ---------------------------------------------------------------------------
# Borsa Italiana: metadata + prezzi aggiuntivi (solo bond italiani)
# ---------------------------------------------------------------------------

_BI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

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
}


def _parse_italian_number(s: str) -> Optional[float]:
    """Converte formato numerico italiano (virgola decimale) in float."""
    try:
        return float(s.strip().replace(".", "").replace(",", "."))
    except (ValueError, AttributeError):
        return None


def scrape_borsa_italiana_bond(isin: str) -> Optional[dict]:
    """
    Scrape della scheda BTP su Borsa Italiana.
    Disponibile solo per bond italiani (MOT).
    Restituisce: price, ytm_gross, ytm_net, accrued_gross, duration, maturity, coupon_frequency.
    """
    url = f"https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/scheda/{isin.upper()}-MOTX.html?lang=it"
    try:
        resp = requests.get(url, headers=_BI_HEADERS, timeout=15)
        if resp.status_code != 200:
            # prova anche senza -MOTX (altri segmenti MOT: BOT, BTP€i, CCT...)
            url2 = f"https://www.borsaitaliana.it/borsa/obbligazioni/mot/altri-titoli-di-stato/scheda/{isin.upper()}-MOTX.html?lang=it"
            resp = requests.get(url2, headers=_BI_HEADERS, timeout=15)
            if resp.status_code != 200:
                return None
    except Exception as e:
        print(f"[Bond] Borsa Italiana fetch error {isin}: {e}")
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
        if field in ("maturity_bi", "coupon_frequency"):
            parsed[field] = raw
        else:
            parsed[field] = _parse_italian_number(raw)

    # Converti cedola periodale → annuale (solitamente semestrale → ×2)
    if parsed.get("coupon_periodic") and parsed.get("coupon_frequency", "").lower().startswith("semes"):
        parsed["coupon_annual"] = round(parsed["coupon_periodic"] * 2, 4)
    elif parsed.get("coupon_periodic"):
        parsed["coupon_annual"] = parsed["coupon_periodic"]

    print(f"[Bond] Borsa Italiana {isin}: price={parsed.get('price')}, ytm={parsed.get('ytm_gross')}, maturity={parsed.get('maturity_bi')}")
    return parsed


# ---------------------------------------------------------------------------
# yfinance fallback
# ---------------------------------------------------------------------------

def fetch_yfinance_bond(isin: str) -> Optional[dict]:
    """Fallback: prova yfinance con vari suffix di exchange."""
    suffixes = [".TI", ".F", ".MI", ".PA", ".AS", ""]
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
            print(f"[Bond] yfinance {ticker_str}: {len(history)} days, last={last_price:.4f}")
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

    Cascade prezzi:  Frankfurt → yfinance
    Metadata extra:  Borsa Italiana (solo ISIN italiani, IT*)
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
        if is_cache_data_fresh(cached_history, cache.updated_at) and cached_price:
            print(f"[Bond] {isin}: cache hit ({len(cached_history)} days)")
            return {"last_price": cached_price, "history": cached_history, "metadata": cached_meta}
        if cached_history:
            print(f"[Bond] {isin}: cache stale, refreshing")

    # --- Storico prezzi: Frankfurt → yfinance ---
    new_history: list = []
    last_price: float = 0.0

    frankfurt_history = fetch_frankfurt_bond_history(isin, days)
    if frankfurt_history:
        new_history = frankfurt_history
        last_price = new_history[-1]["price"]

    if not new_history:
        yf_data = fetch_yfinance_bond(isin)
        if yf_data:
            new_history = yf_data.get("history", [])
            last_price = yf_data.get("last_price", 0.0)

    if not new_history and cached_history:
        print(f"[Bond] {isin}: all sources failed, using stale cache")
        return {"last_price": cached_price, "history": cached_history, "metadata": cached_meta}

    if not new_history:
        raise ValueError(f"Nessun dato prezzo disponibile per il bond {isin}")

    merged_history = merge_historical_data(cached_history, new_history)

    # --- Metadata: Frankfurt master_data + Borsa Italiana (IT*) ---
    meta = cached_meta.copy()

    frankfurt_meta = fetch_frankfurt_bond_metadata(isin)
    if frankfurt_meta:
        meta.update({k: v for k, v in frankfurt_meta.items() if v is not None})

    if isin.startswith("IT"):
        bi_data = scrape_borsa_italiana_bond(isin)
        if bi_data:
            # Borsa Italiana ha più dettagli: YTM, duration, rateo
            for key in ("ytm_gross", "ytm_net", "accrued_gross", "accrued_net", "duration",
                        "coupon_annual", "coupon_frequency", "maturity_bi"):
                if bi_data.get(key) is not None:
                    meta[key] = bi_data[key]
            # Usa il prezzo BI come last_price se Frankfurt non ha dati intraday
            if bi_data.get("price") and not last_price:
                last_price = bi_data["price"]

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
