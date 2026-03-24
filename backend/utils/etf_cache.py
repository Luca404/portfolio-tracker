"""
ETF UCITS cache in-memory condivisa.

Caricata all'avvio di main.py da Supabase (fonte primaria).
Se Supabase non è disponibile o restituisce < 100 righe, cade back al file
statico etf_cache_ucits.py (se presente nel repo come fallback).
Tutti i moduli importano ETF_UCITS_CACHE da qui.
"""

ETF_UCITS_CACHE: list = []
