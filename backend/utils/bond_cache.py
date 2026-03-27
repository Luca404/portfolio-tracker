"""In-memory cache for bond metadata (name, issuer, coupon, maturity, currency).

Loaded from Supabase `bond_metadata_cache` at startup.
Populated on-demand when users look up bonds via /symbols/bond-lookup.
"""

BOND_METADATA_CACHE: list = []
