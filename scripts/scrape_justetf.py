import time
import random
import requests
import pandas as pd
from pathlib import Path

# =========================
# PATHS
# =========================
DATA_DIR = Path("../backend/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "etf_ucits.csv"
OUTPUT_FILTERED = DATA_DIR / "etf_overview_filtered_size.csv"
OUTPUT_LISTINGS = DATA_DIR / "etf_listings.csv"

# =========================
# CONFIG
# =========================
MIN_SIZE = 1000
KEEP_INSTRUMENTS = {"ETF", "ETC"}

BASE_URL = "https://www.justetf.com/en/etf-profile.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
}

# Retry/backoff
MAX_RETRIES = 8
BASE_SLEEP = 1.2            # base sleep tra richieste normali
JITTER = 0.6                # random jitter
BACKOFF_FACTOR = 2.0        # exponential backoff per errori (403/429/5xx)

# Flush progress
BATCH_FLUSH = 10            # salva su csv ogni N ISIN riusciti

# =========================
# UTILS
# =========================
def sleep_with_jitter(seconds: float) -> None:
    time.sleep(max(0.0, seconds + random.uniform(0, JITTER)))

def load_and_filter_overview(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df[df["size"] >= MIN_SIZE]
    df = df[df["instrument"].isin(KEEP_INSTRUMENTS)]
    return df.reset_index(drop=True)

def read_done_isins(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    try:
        done = pd.read_csv(output_csv, usecols=["isin"])
        return set(done["isin"].dropna().astype(str).unique())
    except Exception:
        # se file è corrotto/mezzo scritto, meglio non rischiare:
        return set()

def append_rows_to_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)

# =========================
# SCRAPING
# =========================
def fetch_html(isin: str, session: requests.Session) -> str:
    """
    Fetch pagina ETF. Gestisce retry/backoff se 403/429/5xx o errori rete.
    """
    url = f"{BASE_URL}?isin={isin}"
    attempt = 0
    backoff = BASE_SLEEP

    while True:
        attempt += 1
        try:
            r = session.get(url, headers=HEADERS, timeout=30)

            # Successo
            if r.status_code == 200:
                return r.text

            # Errori "rate limit / blocked"
            if r.status_code in (403, 429, 503, 502, 500):
                if attempt > MAX_RETRIES:
                    raise requests.HTTPError(f"{r.status_code} after {MAX_RETRIES} retries", response=r)

                # backoff esponenziale + un po' di pausa extra per 403
                extra = 8.0 if r.status_code == 403 else 0.0
                wait_s = backoff + extra
                print(f"  -> HTTP {r.status_code} for {isin}, retry {attempt}/{MAX_RETRIES}, sleeping {wait_s:.1f}s")
                sleep_with_jitter(wait_s)
                backoff *= BACKOFF_FACTOR
                continue

            # altri status code: fallisci subito
            r.raise_for_status()

        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt > MAX_RETRIES:
                raise
            wait_s = backoff
            print(f"  -> Network error for {isin} ({e}), retry {attempt}/{MAX_RETRIES}, sleeping {wait_s:.1f}s")
            sleep_with_jitter(wait_s)
            backoff *= BACKOFF_FACTOR


def parse_listings_from_html(isin: str, html: str) -> pd.DataFrame:
    """
    Estrae la tabella 'Listings' dalla pagina justETF usando pandas.read_html.
    """
    tables = pd.read_html(html)

    def is_listings_table(df: pd.DataFrame) -> bool:
        cols = " ".join(str(c).lower() for c in df.columns)
        return "listing" in cols and "ticker" in cols

    candidates = [t for t in tables if is_listings_table(t)]
    if not candidates:
        return pd.DataFrame(columns=["isin", "listing", "trade_currency", "ticker"])

    df = candidates[0].copy()

    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith("listing"):
            rename[c] = "listing"
        elif "trade" in cl and "currency" in cl:
            rename[c] = "trade_currency"
        elif cl.startswith("ticker"):
            rename[c] = "ticker"
        elif "reuters" in cl:
            rename[c] = "reuters_ric"
        elif "bloomberg" in cl:
            rename[c] = "bloomberg"
        elif "market maker" in cl:
            rename[c] = "market_maker"

    df = df.rename(columns=rename)

    keep_cols = [
        c for c in ["listing", "trade_currency", "ticker", "reuters_ric", "bloomberg", "market_maker"]
        if c in df.columns
    ]
    df = df[keep_cols]
    df.insert(0, "isin", isin)

    # pulizia
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.replace({"nan": None, "None": None})

    # rimuovi righe completamente vuote
    df = df.dropna(how="all", subset=[c for c in ["listing", "trade_currency", "ticker"] if c in df.columns])

    return df

# =========================
# MAIN
# =========================
def main():
    print("Loading & filtering overview...")
    overview = load_and_filter_overview(INPUT_CSV)
    overview.to_csv(OUTPUT_FILTERED, index=False)
    print(f"Filtered rows: {len(overview)}")
    print(f"Saved -> {OUTPUT_FILTERED}")

    all_isins = overview["isin"].dropna().astype(str).unique().tolist()

    done_isins = read_done_isins(OUTPUT_LISTINGS)
    todo_isins = [x for x in all_isins if x not in done_isins]

    print(f"Total ISIN after filter: {len(all_isins)}")
    print(f"Already done (from {OUTPUT_LISTINGS.name}): {len(done_isins)}")
    print(f"To do: {len(todo_isins)}")

    buffer = []
    success_count = 0

    with requests.Session() as session:
        for idx, isin in enumerate(todo_isins, 1):
            print(f"[{idx}/{len(todo_isins)}] {isin}")
            try:
                html = fetch_html(isin, session)
                df_list = parse_listings_from_html(isin, html)

                # salva anche se è vuoto? io direi no: vuoto spesso = pagina diversa/blocco
                if df_list.empty:
                    print("  -> No listings table found (skipping save)")
                else:
                    buffer.append(df_list)
                    success_count += 1
                    print(f"  -> listings rows: {len(df_list)}")

                # pausa base tra richieste riuscite
                sleep_with_jitter(BASE_SLEEP)

                # flush periodico
                if success_count > 0 and (success_count % BATCH_FLUSH == 0) and buffer:
                    out = pd.concat(buffer, ignore_index=True)
                    append_rows_to_csv(out, OUTPUT_LISTINGS)
                    print(f"  -> FLUSH saved {len(out)} rows to {OUTPUT_LISTINGS}")
                    buffer.clear()

            except Exception as e:
                print(f"  -> FAILED {isin}: {e}")
                # pausa extra se fallisce, per calmare la situazione
                sleep_with_jitter(10.0)

    # flush finale
    if buffer:
        out = pd.concat(buffer, ignore_index=True)
        append_rows_to_csv(out, OUTPUT_LISTINGS)
        print(f"Final flush saved {len(out)} rows to {OUTPUT_LISTINGS}")

    print("DONE ✅")

if __name__ == "__main__":
    main()
