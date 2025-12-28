"""Pricing and market data utilities."""

import json
import os
from datetime import datetime, timezone, date
from typing import Optional, Dict, List

import pandas as pd
import requests
import yfinance as yf
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import (
    ETFPriceCacheModel,
    StockPriceCacheModel,
    ExchangeRateCacheModel,
    RiskFreeRateCacheModel,
    MarketBenchmarkCacheModel
)
from utils.cache import is_cache_data_fresh, merge_historical_data
from utils.database import commit_with_retry
from utils.dates import DATE_FMT, parse_date_input
from etf_cache_ucits import ETF_UCITS_CACHE

# API Configuration
FMP_API_KEY = os.environ.get("FMP_API_KEY")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
AV_BASE = "https://www.alphavantage.co/query"


def fmp_get(path: str, params: dict, base: Optional[str] = None):
    """Make a request to Financial Modeling Prep API."""
    if not FMP_API_KEY:
        raise HTTPException(status_code=500, detail="FMP_API_KEY not configured")
    base_url = base or FMP_BASE
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    params = params.copy() if params else {}
    params["apikey"] = FMP_API_KEY
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="FMP rate limit exceeded")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"FMP error {resp.status_code}: {resp.text}")
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


def normalize_chart_history(df):
    """
    Converte il DataFrame restituito da justetf_scraping.load_chart in una lista di dict {date, price}.
    È tollerante ai nomi delle colonne (price/close ecc).
    """
    if df is None:
        return []

    if isinstance(df, pd.Series):
        df = df.to_frame(name="price").reset_index()
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    # se non c'è una colonna data esplicita, usa l'indice se non è RangeIndex
    if not any("date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower() for c in df.columns):
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index().rename(columns={"index": "date"})

    cols_lower = {str(c).lower(): c for c in df.columns}
    date_col = None
    for key in ["date", "time", "timestamp"]:
        if key in cols_lower:
            date_col = cols_lower[key]
            break
    if date_col is None:
        # fallback alla prima colonna
        date_col = df.columns[0]

    # price column: priorità a price/close/nav, altrimenti prima numerica non data
    price_col = None
    for key in ["price", "close", "nav", "adjclose", "value", "quote"]:
        if key in cols_lower:
            price_col = cols_lower[key]
            break
    if price_col is None:
        numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            price_col = numeric_cols[0]
        else:
            # se index numerico e single column, usa quella
            if df.shape[1] == 1:
                price_col = df.columns[0]
            else:
                print(f"[justetf normalize] impossibile trovare price column in {df.columns}")
                return []

    ser = df[date_col]
    if pd.api.types.is_numeric_dtype(ser):
        max_val = pd.to_numeric(ser, errors="coerce").max()
        unit = None
        if pd.notnull(max_val):
            if max_val > 1e12:
                unit = "ms"
            elif max_val > 1e9:
                unit = "s"
        df[date_col] = pd.to_datetime(ser, unit=unit, errors="coerce")
    else:
        df[date_col] = df[date_col].apply(parse_date_input)

    df = df.dropna(subset=[date_col])
    current_year = datetime.now(timezone.utc).year
    df = df[df[date_col].apply(lambda x: x.year if isinstance(x, datetime) or hasattr(x, 'year') else None).between(2000, current_year + 1)]

    df = df.sort_values(date_col)
    history = []
    for _, row in df.iterrows():
        try:
            ts = row[date_col]
            if isinstance(ts, datetime):
                ts = ts.date()
            price_val = float(row[price_col])
            history.append({"date": ts.strftime(DATE_FMT), "price": price_val})
        except Exception:
            continue

    return history


def get_etf_price_and_history(isin: str, db: Session):
    """
    Restituisce prezzo corrente e storico per ETF con cache intelligente su SQLite.
    Cascade: justetf_scraping → yfinance (con validazione ISIN)

    JustETF come prima scelta perché:
    - Più affidabile per ETF europei
    - Dati completi e consistenti
    - Nessun rischio di match errati con ticker simili

    yfinance come fallback solo se JustETF non funziona:
    - Prova con ISIN e valida che il ticker restituito corrisponda al symbol atteso
    - Previene match errati con ticker simili

    Cache valida solo se il dato più recente è di oggi (o max 3 giorni fa per weekend).
    """
    if not isin:
        raise HTTPException(status_code=400, detail="Missing ISIN for ETF price lookup")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cache = db.execute(select(ETFPriceCacheModel).where(ETFPriceCacheModel.isin == isin)).scalar_one_or_none()

    cached_history = []
    cached_price = 0.0
    cache_is_fresh = False

    if cache:
        # Carica e normalizza i dati dalla cache
        history_raw = json.loads(cache.history_json or "[]")
        if history_raw:
            # Parse date per ordinamento cronologico corretto
            fixed_with_dates = []
            for h in history_raw:
                d = parse_date_input(h.get("date"))
                if d and d.year >= 2000:  # Filtra date troppo vecchie
                    fixed_with_dates.append({"date_obj": d, "date_str": d.strftime(DATE_FMT), "price": h.get("price")})

            # Ordina cronologicamente (per date_obj, non stringa)
            fixed_with_dates.sort(key=lambda x: x["date_obj"])

            # Converti a formato finale
            cached_history = [{"date": item["date_str"], "price": item["price"]} for item in fixed_with_dates]

        cached_price = cache.last_price or (cached_history[-1]["price"] if cached_history else 0.0)

        # Nuova logica: cache è fresca se il dato più recente è odierno (o max 3gg fa)
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_price:
            print(f"[ETF] {isin}: cache hit ({len(cached_history)} days)")
            return {"last_price": cached_price, "history": cached_history, "currency": cache.currency}

    errors = []

    # Cerca il ticker dal cache UCITS (è una lista, non un dict)
    etf_info = None
    for entry in ETF_UCITS_CACHE:
        if entry.get("isin") == isin:
            etf_info = entry
            break

    symbol = etf_info.get("symbol", "") if etf_info else ""

    # Tentativo 1: justetf_scraping (più affidabile per ETF europei)
    justetf_success = False
    try:
        import justetf_scraping

        def fetch_chart(identifier: str):
            try:
                return justetf_scraping.load_chart(identifier)
            except Exception:
                return None

        print(f"[ETF] {isin}: trying justetf_scraping...")
        df_chart = fetch_chart(isin)
        if df_chart is None:
            # prova anche con il ticker se diverso dall'ISIN
            df_chart = fetch_chart(isin.replace(" ", ""))

        new_history = normalize_chart_history(df_chart)
        if new_history and len(new_history) > 0:
            # Merge dei nuovi dati con quelli in cache (se presenti)
            merged_history = merge_historical_data(cached_history, new_history)

            latest_point = merged_history[-1]
            last_price = latest_point["price"]
            currency = ""

            print(f"[ETF] {isin}: justetf OK, {len(merged_history)} days, last={last_price:.2f}")

            # Aggiorna la cache con i dati unificati
            if cache:
                cache.last_price = last_price
                cache.currency = currency
                cache.history_json = json.dumps(merged_history)
                cache.updated_at = now
            else:
                cache = ETFPriceCacheModel(
                    isin=isin,
                    last_price=last_price,
                    currency=currency,
                    history_json=json.dumps(merged_history),
                    updated_at=now,
                )
                db.add(cache)
            commit_with_retry(db)
            justetf_success = True
            return {"last_price": last_price, "history": merged_history, "currency": currency}
        else:
            print(f"[ETF] {isin}: justetf returned no data")
            errors.append("justetf: no data returned")
    except ImportError:
        print(f"[ETF] {isin}: justetf_scraping not installed")
        errors.append("justetf: module not installed")
    except Exception as e:
        print(f"[ETF] {isin}: justetf failed: {str(e)}")
        errors.append(f"justetf: {str(e)}")

    if not justetf_success:
        print(f"[ETF] {isin}: justetf failed, trying yfinance as fallback...")

    # Tentativo 2: yfinance (fallback) - prova con ISIN e validazione ticker
    tickers_to_try = []
    # ISIN come prima scelta (univoco, ma validato)
    tickers_to_try.append(isin)

    # Poi prova con ticker se disponibile
    if symbol:
        # Ticker così com'è
        tickers_to_try.append(symbol)
        # Se non ha suffisso di mercato, prova le varianti europee comuni
        if "." not in symbol:
            tickers_to_try.extend([f"{symbol}.DE", f"{symbol}.L", f"{symbol}.PA", f"{symbol}.MI"])

    yfinance_success = False
    for ticker_attempt in tickers_to_try:
        try:
            print(f"[ETF] {isin}: trying yfinance with {ticker_attempt}...")
            ticker = yf.Ticker(ticker_attempt)

            # Verifica che il ticker restituito sia quello giusto
            # Se stiamo cercando per ISIN, yfinance potrebbe restituire un ticker diverso
            ticker_info = ticker.info
            actual_ticker = ticker_info.get("symbol", "").upper()

            # Se stiamo cercando per ISIN e abbiamo un symbol atteso, verifica che il ticker trovato corrisponda
            if ticker_attempt == isin and symbol:
                # Estrai il base symbol (senza suffisso di mercato)
                expected_base = symbol.split(".")[0].upper()
                actual_base = actual_ticker.split(".")[0].upper() if actual_ticker else ""

                # Se il ticker trovato non contiene il symbol atteso, skippa questo risultato
                if actual_base and expected_base not in actual_base and actual_base != expected_base:
                    print(f"[ETF] {isin}: yfinance returned wrong ticker {actual_ticker} (expected {symbol}), skipping...")
                    raise ValueError(f"Wrong ticker: got {actual_ticker}, expected {symbol}")

            # Scarica dati storici
            hist = ticker.history(period="max")

            if not hist.empty:
                # Filtra date future
                today = datetime.now(timezone.utc).date()

                history_with_dates = []
                for idx, row in hist.iterrows():
                    try:
                        dt = idx.date() if hasattr(idx, 'date') else idx

                        # Skip date future
                        if dt > today:
                            continue

                        price = float(row["Close"])
                        if price > 0:
                            history_with_dates.append({"date_obj": dt, "price": price})
                    except Exception:
                        continue

                if history_with_dates:
                    # Ordina per data
                    history_with_dates.sort(key=lambda x: x["date_obj"])

                    # Converti a formato string
                    new_history = [
                        {"date": item["date_obj"].strftime(DATE_FMT), "price": item["price"]}
                        for item in history_with_dates
                    ]

                    # Merge con cache esistente
                    merged_history = merge_historical_data(cached_history, new_history)

                    last_price = merged_history[-1]["price"]
                    currency = ""

                    print(f"[ETF] {isin}: yfinance OK with {ticker_attempt}, {len(merged_history)} days, last={last_price:.2f}")

                    # Aggiorna cache
                    if cache:
                        cache.last_price = last_price
                        cache.currency = currency
                        cache.history_json = json.dumps(merged_history)
                        cache.updated_at = now
                    else:
                        cache = ETFPriceCacheModel(
                            isin=isin,
                            last_price=last_price,
                            currency=currency,
                            history_json=json.dumps(merged_history),
                            updated_at=now,
                        )
                        db.add(cache)
                    commit_with_retry(db)
                    yfinance_success = True
                    return {"last_price": last_price, "history": merged_history, "currency": currency}
        except Exception as e:
            errors.append(f"yfinance ({ticker_attempt}): {str(e)}")
            print(f"[ETF] {isin}: yfinance failed with {ticker_attempt}: {str(e)}")
            continue

    # Se nessuna fonte ha funzionato, prova cache stale come ultima risorsa
    if cached_history and cached_price:
        print(f"[ETF] {isin}: using stale cache as last resort")
        return {"last_price": cached_price, "history": cached_history, "currency": cache.currency}

    # Nessuna fonte disponibile
    error_msg = f"Nessun dato disponibile per {isin}. Errori: {'; '.join(errors)}"
    raise HTTPException(status_code=400, detail=error_msg)


def get_stock_price_from_alphavantage(symbol: str) -> dict:
    """
    Fallback ad AlphaVantage per ottenere prezzi storici quando FMP non supporta il ticker.
    Usa TIME_SERIES_DAILY con outputsize=compact (100 giorni, free tier).
    """
    if not ALPHAVANTAGE_API_KEY:
        print(f"[STOCK] {symbol}: AlphaVantage API key not configured")
        raise Exception("AlphaVantage API key not configured")

    symbol = symbol.upper()

    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",  # compact = ultimi 100 giorni (free tier)
            "apikey": ALPHAVANTAGE_API_KEY
        }

        print(f"[STOCK] {symbol}: trying AlphaVantage...")
        response = requests.get(AV_BASE, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Verifica errori API
        if "Error Message" in data:
            error_msg = data['Error Message']
            print(f"[STOCK] {symbol}: AlphaVantage error: {error_msg}")
            raise Exception(f"AlphaVantage error: {error_msg}")

        if "Note" in data:
            print(f"[STOCK] {symbol}: AlphaVantage rate limit")
            raise Exception("AlphaVantage rate limit exceeded")

        if "Information" in data:
            # Messaggio informativo (es. premium feature)
            print(f"[STOCK] {symbol}: AlphaVantage info: {data['Information']}")
            raise Exception("AlphaVantage: Premium feature or API limit")

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            print(f"[STOCK] {symbol}: AlphaVantage no time series data")
            raise Exception("AlphaVantage: No time series data")

        # Converti in formato standard, filtra date future
        today = datetime.now(timezone.utc).date()
        history_with_dates = []

        for date_str, values in time_series.items():
            try:
                dt = parse_date_input(date_str)
                if dt and dt <= today:  # Skip date future
                    price = float(values.get("4. close", 0))
                    if price > 0:
                        history_with_dates.append({"date_obj": dt, "price": price})
            except Exception:
                continue

        if not history_with_dates:
            print(f"[STOCK] {symbol}: AlphaVantage no valid history after parsing")
            raise Exception("AlphaVantage: No valid historical data")

        # Ordina cronologicamente per data oggetto
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string dopo ordinamento
        history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "price": item["price"]}
            for item in history_with_dates
        ]

        last_price = history[-1]["price"]
        last_date = history[-1]["date"]
        print(f"[STOCK] {symbol}: AlphaVantage OK, {len(history)} days, last_date={last_date}, last={last_price:.2f}")
        return {"last_price": last_price, "history": history}

    except Exception as e:
        print(f"[STOCK] {symbol}: AlphaVantage failed: {str(e)}")
        raise


def get_stock_price_from_yfinance(symbol: str, days: int = 180) -> dict:
    """
    Fallback finale a yfinance quando FMP e AlphaVantage falliscono.
    """
    try:
        print(f"[STOCK] {symbol}: trying yfinance...")
        ticker = yf.Ticker(symbol)

        # Scarica dati storici
        hist = ticker.history(period="max" if days > 365 else "1y")

        if hist.empty:
            print(f"[STOCK] {symbol}: yfinance no data")
            raise Exception("yfinance: No data available")

        # Filtra date future (yfinance può dare dati pre-market con date future)
        today = datetime.now(timezone.utc).date()

        # Tieni traccia di date come oggetti per ordinamento corretto
        history_with_dates = []
        for idx, row in hist.iterrows():
            try:
                dt = idx.date() if hasattr(idx, 'date') else idx

                # Skip date future
                if dt > today:
                    continue

                price = float(row["Close"])
                if price > 0:
                    history_with_dates.append({"date_obj": dt, "price": price})
            except Exception:
                continue

        if not history_with_dates:
            print(f"[STOCK] {symbol}: yfinance no valid history")
            raise Exception("yfinance: No valid historical data")

        # Ordina per data oggetto (cronologicamente), non stringa
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string dopo ordinamento
        history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "price": item["price"]}
            for item in history_with_dates
        ]

        last_price = history[-1]["price"]
        last_date = history[-1]["date"]
        print(f"[STOCK] {symbol}: yfinance OK, {len(history)} days, last_date={last_date}, last={last_price:.2f}")
        return {"last_price": last_price, "history": history}

    except Exception as e:
        print(f"[STOCK] {symbol}: yfinance failed: {str(e)}")
        raise


def get_stock_price_and_history(symbol: str, days: int = 180):
    """
    Ottiene prezzo corrente e storico di un titolo azionario.
    Cascade: yfinance → FMP → AlphaVantage

    yfinance come prima scelta perché:
    - Dati illimitati e gratuiti
    - Nessun rate limit
    - Supporto eccellente per stock globali
    - FMP e AlphaVantage come fallback per edge cases
    """
    symbol = symbol.upper()

    def parse_history(raw):
        history = []
        for row in raw or []:
            try:
                dt_raw = row.get("date")
                dt_date = parse_date_input(dt_raw)
                dt = dt_date.strftime(DATE_FMT) if dt_date else None
                price_val = row.get("close") or row.get("adjClose") or row.get("open") or row.get("price") or 0
                if dt and price_val is not None:
                    history.append({"date": dt, "price": float(price_val)})
            except Exception:
                continue
        history = [h for h in history if h.get("price") is not None]
        history.sort(key=lambda x: x["date"])
        return history

    errors = []

    # Tentativo 1: yfinance (prima scelta - dati illimitati, nessun rate limit)
    try:
        return get_stock_price_from_yfinance(symbol, days)
    except Exception as e:
        errors.append(f"yfinance: {str(e)}")

    # Tentativo 2: FMP historical-price-eod
    try:
        data = fmp_get("historical-price-eod/full", {"symbol": symbol, "limit": days}, base=FMP_STABLE_BASE)
        if isinstance(data, dict):
            history = parse_history(data.get("historical") or data.get("data"))
        elif isinstance(data, list):
            history = parse_history(data)
        if history:
            print(f"[STOCK] {symbol}: FMP, {len(history)} days")
            last_price = history[-1]["price"]
            return {"last_price": last_price, "history": history}
    except Exception as e:
        errors.append(f"FMP EOD: {str(e)}")

    # Tentativo 3: FMP historical-chart (alternativa FMP)
    try:
        alt = fmp_get(f"historical-chart/1day/{symbol}", {"limit": days}, base=FMP_STABLE_BASE)
        history = parse_history(alt)
        if history:
            print(f"[STOCK] {symbol}: FMP chart, {len(history)} days")
            last_price = history[-1]["price"]
            return {"last_price": last_price, "history": history}
    except Exception as e:
        errors.append(f"FMP chart: {str(e)}")

    # Tentativo 4: AlphaVantage (ultima spiaggia per edge cases)
    try:
        return get_stock_price_from_alphavantage(symbol)
    except Exception as e:
        errors.append(f"AlphaVantage: {str(e)}")

    # Se arriviamo qui, tutti i provider hanno fallito
    error_msg = f"Nessun dato disponibile per {symbol}. Errori: {'; '.join(errors)}"
    print(f"[STOCK] {symbol}: ALL PROVIDERS FAILED")
    raise HTTPException(status_code=400, detail=error_msg)


def get_stock_price_and_history_cached(symbol: str, db: Session, days: int = 180):
    """
    Wrapper con cache intelligente su SQLite per i prezzi storici delle azioni.
    Cache valida solo se il dato più recente è di oggi (o max 3 giorni fa per weekend).
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cache = db.execute(select(StockPriceCacheModel).where(StockPriceCacheModel.symbol == symbol)).scalar_one_or_none()

    cached_history = []
    cached_price = 0.0
    cache_is_fresh = False

    if cache:
        # Carica e normalizza i dati dalla cache
        history_raw = json.loads(cache.history_json or "[]")
        if history_raw:
            # Parse date per ordinamento cronologico corretto
            fixed_with_dates = []
            for h in history_raw:
                d = parse_date_input(h.get("date"))
                if d:
                    fixed_with_dates.append({"date_obj": d, "date_str": d.strftime(DATE_FMT), "price": h.get("price")})

            # Ordina cronologicamente (per date_obj, non stringa)
            fixed_with_dates.sort(key=lambda x: x["date_obj"])

            # Converti a formato finale
            cached_history = [{"date": item["date_str"], "price": item["price"]} for item in fixed_with_dates]

        cached_price = cache.last_price or (cached_history[-1]["price"] if cached_history else 0.0)

        # Nuova logica: cache è fresca se il dato più recente è odierno (o max 3gg fa)
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_price:
            print(f"[STOCK] {symbol}: cache hit ({len(cached_history)} days, last={cached_price:.2f})")
            return {"last_price": cached_price, "history": cached_history}
        else:
            if cached_history:
                latest_date = cached_history[-1].get("date") if cached_history else "none"
                print(f"[STOCK] {symbol}: cache stale (last date={latest_date})")

    # Scarica nuovi dati dall'API
    data = get_stock_price_and_history(symbol, days=days)
    new_history = data.get("history", [])
    last_price = data.get("last_price", 0.0)

    if new_history and last_price:
        # Merge dei nuovi dati con quelli in cache (se presenti)
        merged_history = merge_historical_data(cached_history, new_history)

        # Aggiorna la cache con i dati unificati
        if cache:
            cache.last_price = last_price
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = StockPriceCacheModel(
                symbol=symbol,
                last_price=last_price,
                history_json=json.dumps(merged_history),
                updated_at=now,
            )
            db.add(cache)
        commit_with_retry(db)

        return {"last_price": last_price, "history": merged_history}

    return data


def convert_to_reference_currency(amount: float, from_currency: str, to_currency: str, db: Session) -> float:
    """
    Converte un importo dalla valuta from_currency alla valuta to_currency
    usando l'ultimo tasso di cambio disponibile.
    """
    if from_currency == to_currency or not from_currency or not to_currency:
        return amount

    try:
        fx_data = get_exchange_rate_history(from_currency, to_currency, db)
        fx_rates = fx_data.get("rates", [])
        if fx_rates:
            latest_rate = fx_rates[-1].get("rate", 1.0)
            return amount * latest_rate
    except Exception as e:
        print(f"[FX] Conversion {from_currency}->{to_currency} failed: {e}")

    return amount


def get_exchange_rate_history(from_currency: str, to_currency: str, db: Session) -> dict:
    """
    Ottiene lo storico dei tassi di cambio tra due valute usando yfinance.
    Esempio: USD -> EUR usa il ticker "USDEUR=X" su yfinance

    Ritorna: {"rates": [{"date": "DD-MM-YYYY", "rate": 0.92}, ...]}
    """
    # Se stessa valuta, ritorna rate 1.0
    if from_currency == to_currency:
        return {"rates": []}

    # Crea pair identifier per yfinance
    pair = f"{from_currency}{to_currency}=X"

    now = datetime.now(timezone.utc)

    # Controlla cache
    cache = db.query(ExchangeRateCacheModel).filter_by(pair=pair).first()
    cached_rates = []

    if cache:
        history_raw = json.loads(cache.history_json or "[]")
        if history_raw:
            # Parse e ordina date
            fixed_with_dates = []
            for h in history_raw:
                d = parse_date_input(h.get("date"))
                if d and d.year >= 2000:
                    fixed_with_dates.append({
                        "date_obj": d,
                        "date_str": d.strftime(DATE_FMT),
                        "rate": h.get("rate")
                    })
            fixed_with_dates.sort(key=lambda x: x["date_obj"])
            cached_rates = [{"date": item["date_str"], "rate": item["rate"]} for item in fixed_with_dates]

        # Cache fresca se ultimo dato è recente
        cache_is_fresh = is_cache_data_fresh(cached_rates)

        if cache_is_fresh and cached_rates:
            print(f"[FX] {pair}: cache hit ({len(cached_rates)} days)")
            return {"rates": cached_rates}
        else:
            if cached_rates:
                latest_date = cached_rates[-1].get("date") if cached_rates else "none"
                print(f"[FX] {pair}: cache stale (last date={latest_date})")

    # Scarica dati da yfinance
    try:
        print(f"[FX] {pair}: downloading from yfinance...")
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="max")

        if hist.empty:
            print(f"[FX] {pair}: no data from yfinance")
            # Ritorna cache anche se stale, meglio di niente
            return {"rates": cached_rates} if cached_rates else {"rates": []}

        today = datetime.now(timezone.utc).date()
        rates_with_dates = []

        for idx, row in hist.iterrows():
            try:
                dt = idx.date() if hasattr(idx, 'date') else idx
                if dt > today:
                    continue
                rate = float(row['Close'])
                if rate > 0:
                    rates_with_dates.append({"date_obj": dt, "rate": rate})
            except Exception:
                continue

        if not rates_with_dates:
            print(f"[FX] {pair}: no valid data after parsing")
            return {"rates": cached_rates} if cached_rates else {"rates": []}

        # Ordina cronologicamente
        rates_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string
        new_rates = [
            {"date": item["date_obj"].strftime(DATE_FMT), "rate": item["rate"]}
            for item in rates_with_dates
        ]

        # Merge con cache esistente
        merged_rates = merge_historical_data(cached_rates, new_rates)

        # Aggiorna cache
        if cache:
            cache.history_json = json.dumps(merged_rates)
            cache.updated_at = now
        else:
            cache = ExchangeRateCacheModel(
                pair=pair,
                history_json=json.dumps(merged_rates),
                updated_at=now
            )
            db.add(cache)
        commit_with_retry(db)

        print(f"[FX] {pair}: OK, {len(merged_rates)} days")
        return {"rates": merged_rates}

    except Exception as e:
        print(f"[FX] {pair}: error: {str(e)}")
        # Ritorna cache anche se stale
        return {"rates": cached_rates} if cached_rates else {"rates": []}


def fetch_us_treasury_rate(db: Session) -> dict:
    """
    Scarica i tassi US Treasury da FMP API.
    Usa l'endpoint: https://financialmodelingprep.com/stable/treasury-rates

    Ritorna: {"current_rate": 4.5, "history": [{"date": "DD-MM-YYYY", "rate": 4.5}, ...]}
    """
    currency = "USD"
    now = datetime.now(timezone.utc)

    # Controlla cache
    cache = db.query(RiskFreeRateCacheModel).filter_by(currency=currency).first()
    cached_history = []

    if cache:
        cached_history = json.loads(cache.history_json or "[]")
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_history:
            print(f"[RiskFree] USD Treasury: cache hit ({len(cached_history)} days)")
            return {"current_rate": cache.current_rate, "history": cached_history}

    # Scarica da FMP
    try:
        data = fmp_get("treasury-rates", {}, base=FMP_STABLE_BASE)

        if not data or not isinstance(data, list):
            print(f"[RiskFree] USD Treasury: no data from FMP")
            return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}

        # Estrai il tasso a 10 anni (standard per risk-free)
        history_with_dates = []
        current_rate = 0.0

        for item in data:
            try:
                # FMP ritorna: {"date": "YYYY-MM-DD", "month": X, "year1": X, "year10": X, ...}
                date_str = item.get("date")
                rate_10y = item.get("year10")

                if date_str and rate_10y is not None:
                    dt = parse_date_input(date_str)
                    if dt:
                        history_with_dates.append({
                            "date_obj": dt,
                            "rate": float(rate_10y)
                        })
            except Exception:
                continue

        if not history_with_dates:
            print(f"[RiskFree] USD Treasury: no valid data after parsing")
            return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}

        # Ordina cronologicamente
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string
        new_history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "rate": item["rate"]}
            for item in history_with_dates
        ]

        # Ultimo tasso disponibile
        current_rate = new_history[-1]["rate"] if new_history else 0.0

        # Merge con cache
        merged_history = merge_historical_data(cached_history, new_history)

        # Aggiorna cache
        if cache:
            cache.current_rate = current_rate
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = RiskFreeRateCacheModel(
                currency=currency,
                current_rate=current_rate,
                history_json=json.dumps(merged_history),
                updated_at=now
            )
            db.add(cache)
        commit_with_retry(db)

        print(f"[RiskFree] USD Treasury: OK, current={current_rate}%, {len(merged_history)} days")
        return {"current_rate": current_rate, "history": merged_history}

    except Exception as e:
        print(f"[RiskFree] USD Treasury: error: {str(e)}")
        return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}


def fetch_ecb_rate(db: Session) -> dict:
    """
    Scarica i tassi ECB (European Central Bank) dall'API ufficiale.
    Usa il tasso di riferimento principale (Main Refinancing Operations).

    API: https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.MRR_FR.LEV

    Ritorna: {"current_rate": 4.5, "history": [{"date": "DD-MM-YYYY", "rate": 4.5}, ...]}
    """
    currency = "EUR"
    now = datetime.now(timezone.utc)

    # Controlla cache
    cache = db.query(RiskFreeRateCacheModel).filter_by(currency=currency).first()
    cached_history = []

    if cache:
        cached_history = json.loads(cache.history_json or "[]")
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_history:
            print(f"[RiskFree] EUR ECB: cache hit ({len(cached_history)} days)")
            return {"current_rate": cache.current_rate, "history": cached_history}

    # Scarica da ECB API
    try:
        # API ECB per Main Refinancing Operations rate
        url = "https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.MRR_FR.LEV"
        params = {"format": "jsondata", "detail": "dataonly"}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse ECB JSON structure
        observations = data.get("dataSets", [{}])[0].get("series", {}).get("0:0:0:0:0:0:0", {}).get("observations", {})
        dimensions = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_values = None

        for dim in dimensions:
            if dim.get("id") == "TIME_PERIOD":
                time_values = dim.get("values", [])
                break

        if not observations or not time_values:
            print(f"[RiskFree] EUR ECB: no data from ECB API")
            return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}

        history_with_dates = []

        for idx, value_list in observations.items():
            try:
                time_idx = int(idx)
                if time_idx < len(time_values):
                    date_str = time_values[time_idx].get("id")  # formato: YYYY-MM-DD
                    rate = float(value_list[0])

                    dt = parse_date_input(date_str)
                    if dt:
                        history_with_dates.append({
                            "date_obj": dt,
                            "rate": rate
                        })
            except Exception:
                continue

        if not history_with_dates:
            print(f"[RiskFree] EUR ECB: no valid data after parsing")
            return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}

        # Ordina cronologicamente
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string
        new_history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "rate": item["rate"]}
            for item in history_with_dates
        ]

        # Ultimo tasso disponibile
        current_rate = new_history[-1]["rate"] if new_history else 0.0

        # Merge con cache
        merged_history = merge_historical_data(cached_history, new_history)

        # Aggiorna cache
        if cache:
            cache.current_rate = current_rate
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = RiskFreeRateCacheModel(
                currency=currency,
                current_rate=current_rate,
                history_json=json.dumps(merged_history),
                updated_at=now
            )
            db.add(cache)
        commit_with_retry(db)

        print(f"[RiskFree] EUR ECB: OK, current={current_rate}%, {len(merged_history)} days")
        return {"current_rate": current_rate, "history": merged_history}

    except Exception as e:
        print(f"[RiskFree] EUR ECB: error: {str(e)}")
        return {"current_rate": cache.current_rate if cache else 0.0, "history": cached_history}


def fetch_market_benchmark(currency: str, db: Session) -> dict:
    """
    Scarica i dati del benchmark di mercato appropriato per la valuta.
    - USD: S&P 500 (^GSPC)
    - EUR: VWCE (VWCE.DE - Vanguard FTSE All-World)

    Ritorna: {"symbol": "^GSPC", "last_price": 4500, "history": [{"date": "DD-MM-YYYY", "price": 4500}, ...]}
    """
    # Mappa valuta -> simbolo
    symbol_map = {
        "USD": "^GSPC",      # S&P 500
        "EUR": "VWCE.DE"     # Vanguard FTSE All-World (quotato in EUR su XETRA)
    }

    symbol = symbol_map.get(currency.upper())
    if not symbol:
        raise HTTPException(status_code=400, detail=f"No market benchmark defined for currency {currency}")

    now = datetime.now(timezone.utc)

    # Controlla cache
    cache = db.query(MarketBenchmarkCacheModel).filter_by(currency=currency.upper()).first()
    cached_history = []

    if cache:
        cached_history = json.loads(cache.history_json or "[]")
        cache_is_fresh = is_cache_data_fresh(cached_history)

        if cache_is_fresh and cached_history:
            print(f"[Benchmark] {currency} ({symbol}): cache hit ({len(cached_history)} days)")
            return {
                "symbol": symbol,
                "last_price": cache.last_price,
                "history": cached_history
            }

    # Scarica da yfinance
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max")

        if hist.empty:
            print(f"[Benchmark] {currency} ({symbol}): no data from yfinance")
            return {
                "symbol": symbol,
                "last_price": cache.last_price if cache else 0.0,
                "history": cached_history
            }

        today = datetime.now(timezone.utc).date()
        history_with_dates = []

        for idx, row in hist.iterrows():
            try:
                dt = idx.date() if hasattr(idx, 'date') else idx
                if dt > today:
                    continue
                price = float(row['Close'])
                if price > 0:
                    history_with_dates.append({"date_obj": dt, "price": price})
            except Exception:
                continue

        if not history_with_dates:
            print(f"[Benchmark] {currency} ({symbol}): no valid data after parsing")
            return {
                "symbol": symbol,
                "last_price": cache.last_price if cache else 0.0,
                "history": cached_history
            }

        # Ordina cronologicamente
        history_with_dates.sort(key=lambda x: x["date_obj"])

        # Converti a formato string
        new_history = [
            {"date": item["date_obj"].strftime(DATE_FMT), "price": item["price"]}
            for item in history_with_dates
        ]

        # Ultimo prezzo disponibile
        last_price = new_history[-1]["price"] if new_history else 0.0

        # Merge con cache
        merged_history = merge_historical_data(cached_history, new_history)

        # Aggiorna cache
        if cache:
            cache.symbol = symbol
            cache.last_price = last_price
            cache.history_json = json.dumps(merged_history)
            cache.updated_at = now
        else:
            cache = MarketBenchmarkCacheModel(
                currency=currency.upper(),
                symbol=symbol,
                last_price=last_price,
                history_json=json.dumps(merged_history),
                updated_at=now
            )
            db.add(cache)
        commit_with_retry(db)

        print(f"[Benchmark] {currency} ({symbol}): OK, last={last_price}, {len(merged_history)} days")
        return {
            "symbol": symbol,
            "last_price": last_price,
            "history": merged_history
        }

    except Exception as e:
        print(f"[Benchmark] {currency} ({symbol}): error: {str(e)}")
        return {
            "symbol": symbol,
            "last_price": cache.last_price if cache else 0.0,
            "history": cached_history
        }


def get_risk_free_rate(currency: str, db: Session, custom_source: Optional[str] = None) -> float:
    """
    Helper function per ottenere il tasso risk-free corrente.

    Args:
        currency: "USD", "EUR", o "GBP"
        db: Database session
        custom_source: Fonte personalizzata ("auto", "USD_TREASURY", "EUR_ECB", "EUR_BUND", "GBP_GILT", o valore numerico custom)

    Returns:
        Tasso risk-free annuale in percentuale (es: 4.5)
    """
    # Se custom_source è un numero, usalo direttamente
    if custom_source and custom_source != "auto":
        try:
            return float(custom_source)
        except ValueError:
            # Non è un numero, procedi con la logica normale
            if custom_source == "USD_TREASURY":
                data = fetch_us_treasury_rate(db)
                return data.get("current_rate", 0.0)
            elif custom_source == "EUR_ECB":
                data = fetch_ecb_rate(db)
                return data.get("current_rate", 0.0)
            elif custom_source == "EUR_BUND":
                # German Bund 10Y - per ora usa ECB come fallback
                # TODO: Implementare fetch specifico per Bund tramite yfinance (^TNX equivalente EU)
                print("[RiskFree] EUR_BUND: using ECB as fallback")
                data = fetch_ecb_rate(db)
                return data.get("current_rate", 0.0)
            elif custom_source == "GBP_GILT":
                # UK Gilt 10Y - per ora usa un valore di default
                # TODO: Implementare fetch specifico per UK Gilt
                print("[RiskFree] GBP_GILT: using default 4.0%")
                return 4.0

    # Auto mode: determina in base alla currency
    currency = currency.upper()
    if currency == "USD":
        data = fetch_us_treasury_rate(db)
    elif currency == "EUR":
        data = fetch_ecb_rate(db)
    elif currency == "GBP":
        # UK Gilt come default per GBP
        print("[RiskFree] GBP auto: using default 4.0%")
        return 4.0
    else:
        print(f"[RiskFree] Unknown currency {currency}, defaulting to 0.0")
        return 0.0

    return data.get("current_rate", 0.0)


def get_market_benchmark_data(currency: str, db: Session, custom_benchmark: Optional[str] = None) -> dict:
    """
    Helper function per ottenere i dati del benchmark di mercato.

    Args:
        currency: "USD" o "EUR"
        db: Database session
        custom_benchmark: Benchmark personalizzato ("auto", "SP500", "VWCE", o ticker custom)

    Returns:
        {"symbol": "^GSPC", "last_price": 4500, "history": [...]}
    """
    # Se custom_benchmark è specificato e non è "auto"
    if custom_benchmark and custom_benchmark != "auto":
        if custom_benchmark == "SP500":
            return fetch_market_benchmark("USD", db)
        elif custom_benchmark == "VWCE":
            return fetch_market_benchmark("EUR", db)
        else:
            # Custom ticker - usa yfinance direttamente
            try:
                ticker = yf.Ticker(custom_benchmark)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    last_price = float(hist['Close'].iloc[-1])
                    history = []
                    for idx, row in hist.iterrows():
                        dt = idx.date() if hasattr(idx, 'date') else idx
                        history.append({
                            "date": dt.strftime(DATE_FMT),
                            "price": float(row['Close'])
                        })
                    return {"symbol": custom_benchmark, "last_price": last_price, "history": history}
            except Exception as e:
                print(f"[Benchmark] Error fetching custom ticker {custom_benchmark}: {e}")

    # Auto mode: determina in base alla currency
    return fetch_market_benchmark(currency, db)


def get_stock_splits(symbol: str, from_date: date) -> Dict[date, float]:
    """
    Recupera gli stock splits per un simbolo a partire da una data.

    Args:
        symbol: Simbolo dello stock (es. "NFLX")
        from_date: Data da cui cercare gli split (tipicamente la data del primo ordine)

    Returns:
        Dict mapping date -> split ratio (es. {date(2024, 7, 15): 10.0} per split 10:1)
    """
    try:
        ticker = yf.Ticker(symbol)
        splits = ticker.splits

        if splits is None or splits.empty:
            return {}

        # Filtra solo gli split dopo from_date e converti in dict
        result = {}
        for split_datetime, ratio in splits.items():
            split_date = split_datetime.date() if hasattr(split_datetime, 'date') else split_datetime
            if split_date >= from_date:
                result[split_date] = float(ratio)
                print(f"[SPLIT] {symbol}: {split_date} -> {ratio}:1 split")

        return result
    except Exception as e:
        print(f"[SPLIT] Error fetching splits for {symbol}: {e}")
        return {}


def apply_splits_to_orders(orders: List, symbol_splits: Dict[str, Dict[date, float]]):
    """
    Applica gli stock splits agli ordini, aggiustando quantità e prezzi.

    Logic:
    - Per ogni ordine, controlla se ci sono stati split DOPO la data dell'ordine
    - Se sì, moltiplica la quantità per il ratio cumulativo e dividi il prezzo

    Args:
        orders: Lista di OrderModel
        symbol_splits: Dict mapping symbol -> {date: ratio}

    Returns:
        Lista di ordini con quantità e prezzi aggiustati (crea copie, non modifica gli originali)
    """
    adjusted_orders = []

    for order in orders:
        symbol = order.symbol.upper()
        splits = symbol_splits.get(symbol, {})

        if not splits:
            # Nessuno split, usa ordine originale
            adjusted_orders.append(order)
            continue

        # Calcola il ratio cumulativo di tutti gli split DOPO questo ordine
        cumulative_ratio = 1.0
        for split_date, ratio in splits.items():
            if split_date > order.date:
                cumulative_ratio *= ratio

        if cumulative_ratio == 1.0:
            # Nessuno split dopo questo ordine
            adjusted_orders.append(order)
        else:
            # Crea una copia dell'ordine con valori aggiustati
            # Nota: non modifichiamo l'oggetto DB originale, creiamo una copia
            import copy
            adjusted_order = copy.copy(order)
            adjusted_order.quantity = order.quantity * cumulative_ratio
            adjusted_order.price = order.price / cumulative_ratio

            print(f"[SPLIT] Adjusted {symbol} order from {order.date}: "
                  f"qty {order.quantity} -> {adjusted_order.quantity}, "
                  f"price ${order.price:.2f} -> ${adjusted_order.price:.2f} "
                  f"(ratio: {cumulative_ratio}:1)")

            adjusted_orders.append(adjusted_order)

    return adjusted_orders
