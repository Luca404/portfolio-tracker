# scripts/generate_etf_cache.py
"""
Genera il file backend/etf_cache_ucits.py unendo:
- backend/data/etf_ucits.csv (overview: name, ter, domicile, ecc.)
- backend/data/etf_listings.csv (listing per exchange con ticker e currency)

Output: lista flatten di listing (una entry per ticker/exchange) con i campi utili a ricerca/autocomplete.
"""

import json
from pathlib import Path
import pandas as pd


def generate_etf_cache():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    overview_path = data_dir / "etf_ucits.csv"
    listings_path = data_dir / "etf_listings.csv"
    output_file = base_dir.parent / "backend" / "etf_cache_ucits.py"

    if not overview_path.exists():
        raise FileNotFoundError(f"Manca {overview_path}")
    if not listings_path.exists():
        raise FileNotFoundError(f"Manca {listings_path}")

    overview_df = pd.read_csv(overview_path)
    listings_df = pd.read_csv(listings_path)

    # Mappa info principali per ISIN
    ucits_map = {}
    for _, row in overview_df.iterrows():
        isin = str(row.get("isin", "") or row.name)
        ucits_map[isin] = {
            "name": str(row.get("name", "")),
            "currency": str(row.get("currency", "")),
            "domicile": str(row.get("domicile_country", "")),
            "exchange": str(row.get("exchange", "")),
            "ter": row.get("ter", None),
            "ticker": str(row.get("ticker", "")),
            "wkn": str(row.get("wkn", "")),
            "instrument": str(row.get("instrument", "ETF")),
        }

    etf_list = []
    seen = set()

    # Aggiungi entry per ogni listing (solo ISIN presenti in listings)
    for _, row in listings_df.iterrows():
        isin = str(row.get("isin", "")).strip()
        if not isin:
            continue
        base = ucits_map.get(isin, {})
        ticker = str(row.get("ticker", "") or "").strip() or base.get("ticker", "")
        exchange = str(row.get("listing", "") or "").strip() or base.get("exchange", "")
        currency = str(row.get("trade_currency", "") or "").strip() or base.get("currency", "")
        if not ticker:
            continue
        sym = ticker or isin
        key = (sym, exchange, currency)
        if key in seen:
            continue
        seen.add(key)
        etf_list.append({
            "symbol": sym,
            "isin": isin,
            "name": base.get("name", ""),
            "currency": currency,
            "exchange": exchange,
            "domicile": base.get("domicile", ""),
            "ter": base.get("ter", None),
            "ticker": ticker,
            "wkn": base.get("wkn", ""),
            "instrument": base.get("instrument", "ETF"),
            "type": "ETF",
            "listing": exchange,
        })

    # Non aggiungiamo ISIN senza listing: la cache deve avere solo voci presenti in listings

    # Scrivi file Python
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write('ETF UCITS Cache - Lista flatten di listing UCITS\n')
        f.write('Generato automaticamente da scripts/generate_etf_cache.py\n')
        f.write(f'Numero di listing: {len(etf_list)}\n')
        f.write('"""\n\n')
        f.write('ETF_UCITS_CACHE = ')
        f.write(json.dumps(etf_list, indent=2, ensure_ascii=False))
        f.write('\n\n\n')
        f.write('''
def search_etf_ucits_cache(query: str, limit: int = 20):
    """
    Cerca nella cache locale degli ETF UCITS (listing) per ISIN/ticker/WKN/nome.
    """
    query = query.upper().strip()
    results = []

    exact = [etf for etf in ETF_UCITS_CACHE if etf.get("isin", "").upper() == query or etf.get("ticker", "").upper() == query]
    if exact:
        return exact[:limit]

    for etf in ETF_UCITS_CACHE:
        if etf.get("symbol", "").upper() == query or etf.get("wkn", "").upper() == query:
            results.append(etf)
        elif etf.get("symbol", "").upper().startswith(query):
            results.append(etf)
        elif query in etf.get("name", "").upper():
            results.append(etf)
        if len(results) >= limit:
            break
    return results[:limit]


def get_all_ucits_etfs():
    """Restituisce la cache degli ETF UCITS (listing)."""
    return ETF_UCITS_CACHE.copy()
''')

    print(f"File generato: {output_file} (entries: {len(etf_list)})")


if __name__ == "__main__":
    generate_etf_cache()
