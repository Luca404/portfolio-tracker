"""
Bond data fetching: Borsa Italiana (spot price + metadata) → Börse Frankfurt
(history/metadata) → yfinance (fallback).

Prezzi Frankfurt restituiti in % del nominale (tradedInPercent=true).
Borsa Italiana fornisce il Prezzo Ufficiale e diversi metadati utili anche per molti
bond europei quotati su MOT/EuroMOT/ExtraMOT.
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
    matched_url = None
    for url in candidates:
        try:
            r = requests.get(url, headers=_BI_HEADERS, timeout=10)
            if r.status_code == 200 and "scheda" in r.text and len(r.text) > 5000:
                resp = r
                matched_url = url
                break
        except Exception:
            continue

    if resp is None:
        print(f"[Bond] Borsa Italiana {isin_up}: non trovato su nessun segmento MOT/EuroMOT/ExtraMOT")
        return None

    print(f"[Bond] Borsa Italiana {isin_up}: found at {matched_url}")
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

    print(f"[Bond] Borsa Italiana {isin_up}: name={parsed.get('name')}, issuer={parsed.get('issuer')}, price={parsed.get('price')}, ytm={parsed.get('ytm_gross')}, maturity={parsed.get('maturity_bi')}")
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
            print(f"[Bond] Borsa chart auth {isin_up}: exch={value['exchcode']}")
            return value
        except Exception as e:
            print(f"[Bond] Borsa chart auth {isin_up} failed ({url}): {e}")

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
            print(f"[Bond] Borsa chart history {isin}: HTTP {resp.status_code}")
            return []
        payload = resp.json()
        rows = (((payload or {}).get("history") or {}).get("historyDt")) or []
        history = []
        for row in rows:
            raw_date = row.get("dt", "")
            parsed_date = parse_date_input(raw_date)
            price = row.get("closePx")
            if parsed_date and price is not None:
                history.append({"date": parsed_date.strftime(DATE_FMT), "price": float(price)})
        history.sort(key=lambda x: x["date"])
        print(f"[Bond] Borsa history {isin}: {len(history)} days found")
        return history
    except Exception as e:
        print(f"[Bond] Borsa chart history {isin} error: {e}")
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
            print(f"[Bond] yfinance {ticker_str}: {len(history)} days found, last={last_price:.4f}")
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
        if is_cache_data_fresh(cached_history, cache.updated_at) and cached_price:
            print(f"[Bond] {isin}: cache hit ({len(cached_history)} days)")
            return {"last_price": cached_price, "history": cached_history, "metadata": cached_meta}
        if cached_history:
            print(f"[Bond] {isin}: cache stale, refreshing")

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

    # --- Storico prezzi: Borsa Italiana → Frankfurt → yfinance ---
    new_history: list = []
    last_price: float = 0.0

    if bi_data and bi_data.get("price") is not None:
        last_price = float(bi_data["price"])
        today_point = {"date": datetime.now(timezone.utc).date().strftime(DATE_FMT), "price": last_price}
        new_history = [today_point]
        print(f"[Bond] Borsa Italiana {isin}: using official price {last_price:.4f} (1 day found)")

    # Per richieste spot, il prezzo ufficiale BI basta.
    # Per history vera, proviamo prima il chart API di Borsa Italiana.
    if days > 7:
        if days >= 365 * 5:
            bi_period = "5Y"
        elif days >= 365:
            bi_period = "1Y"
        else:
            bi_period = "1M"
        borsa_history = fetch_borsa_italiana_bond_history(isin, period=bi_period)
        if borsa_history:
            new_history = borsa_history
            last_price = borsa_history[-1]["price"]

    if not new_history and not (bi_data and bi_data.get("price") is not None and days <= 7):
        frankfurt_history = fetch_frankfurt_bond_history(isin, days)
        if frankfurt_history:
            new_history = frankfurt_history
            last_price = new_history[-1]["price"]

        if not frankfurt_history:
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
    print(f"[Bond] {isin}: merged history points={len(merged_history)}")

    # --- Metadata: Frankfurt come complemento se disponibile ---
    if not (bi_data and bi_data.get("price") is not None and days <= 7):
        frankfurt_meta = fetch_frankfurt_bond_metadata(isin)
        if frankfurt_meta:
            for key in ("issuer", "currency", "issue_date", "min_investment"):
                if frankfurt_meta.get(key) and not meta.get(key):
                    meta[key] = frankfurt_meta[key]
            if frankfurt_meta.get("maturity") and not meta.get("maturity"):
                meta["maturity"] = frankfurt_meta["maturity"]
            if frankfurt_meta.get("coupon") and not meta.get("coupon"):
                meta["coupon"] = frankfurt_meta["coupon"]

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
