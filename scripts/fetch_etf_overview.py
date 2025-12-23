"""
Scarica l'overview UCITS da justETF e salva in scripts/data/etf_ucits.csv
Richiede: pip install git+https://github.com/druzsan/justetf-scraping.git
"""

from pathlib import Path
import pandas as pd
import justetf_scraping


def fetch_overview():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / "etf_ucits.csv"

    print("Scarico overview UCITS da justETF...")
    df = justetf_scraping.load_overview(enrich=True)
    df.to_csv(output, index=False)
    print(f"Salvato {len(df)} righe in {output}")


if __name__ == "__main__":
    fetch_overview()
