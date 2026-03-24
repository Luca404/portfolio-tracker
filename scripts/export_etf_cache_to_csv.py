"""
Genera un CSV da etf_cache_ucits.py pronto per l'import su Supabase
nella tabella etf_ucits_cache (colonne: isin, ticker, exchange, name, currency, ter).

Uso:
    cd portfolio-tracker
    python scripts/export_etf_cache_to_csv.py
    # output: scripts/data/etf_ucits_supabase.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from etf_cache_ucits import ETF_UCITS_CACHE

output = Path(__file__).parent / "data" / "etf_ucits_supabase.csv"

seen = set()
rows = []
for e in ETF_UCITS_CACHE:
    isin = (e.get("isin") or "").strip()
    ticker = (e.get("ticker") or e.get("symbol") or "").strip()
    exchange = (e.get("exchange") or e.get("listing") or "").strip()
    if not isin or not ticker:
        continue
    key = (isin, ticker, exchange)
    if key in seen:
        continue
    seen.add(key)
    rows.append({
        "isin": isin,
        "ticker": ticker,
        "exchange": exchange,
        "name": (e.get("name") or "").strip(),
        "currency": (e.get("currency") or "").strip(),
        "ter": e.get("ter") if e.get("ter") is not None else "",
    })

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["isin", "ticker", "exchange", "name", "currency", "ter"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Esportati {sum(1 for _ in open(output)) - 1} ETF → {output}")
