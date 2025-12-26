from datetime import datetime, timezone
from .dates import parse_date_input

# Cache invalidation: valida solo se il dato più recente è odierno o max 3 giorni fa
# (per gestire weekend/festivi)
MAX_DAYS_STALE = 3  # Accetta dati fino a 3 giorni fa come "freschi"


def is_cache_data_fresh(history: list) -> bool:
    """
    Verifica se i dati storici in cache sono ancora validi.
    Cache è valida se il dato più recente è:
    - Di oggi
    - Di ieri/l'altro ieri (max 3 giorni fa) per gestire weekend/festivi

    Args:
        history: Lista di dict con chiave "date" in formato DATE_FMT

    Returns:
        True se la cache è ancora fresca, False altrimenti
    """
    if not history:
        return False

    try:
        # Prendi la data più recente dalla history
        latest_date_str = history[-1].get("date")
        if not latest_date_str:
            return False

        latest_date = parse_date_input(latest_date_str)
        if not latest_date:
            return False

        # Calcola differenza in giorni
        today = datetime.now(timezone.utc).date()
        days_old = (today - latest_date).days

        # Cache valida se dato più recente ha max MAX_DAYS_STALE giorni
        return days_old <= MAX_DAYS_STALE

    except Exception:
        return False


def merge_historical_data(cached_history: list, new_history: list) -> list:
    """
    Unisce i dati storici esistenti in cache con nuovi dati scaricati,
    evitando duplicati e mantenendo solo i dati più recenti per ogni data.

    Args:
        cached_history: Dati già presenti in cache
        new_history: Nuovi dati scaricati dall'API

    Returns:
        Lista unificata e ordinata per data
    """
    if not cached_history:
        return new_history

    if not new_history:
        return cached_history

    # Determina quale chiave usare (price o rate)
    value_key = "price" if "price" in (new_history[0] if new_history else cached_history[0]) else "rate"

    # Usa dict per evitare duplicati (chiave=data, valore=prezzo/rate)
    # I nuovi dati sovrascrivono quelli vecchi se stessa data
    merged = {}

    for item in cached_history:
        date_str = item.get("date")
        if date_str:
            merged[date_str] = item.get(value_key)

    for item in new_history:
        date_str = item.get("date")
        if date_str:
            merged[date_str] = item.get(value_key)

    # Converti in lista e ordina
    result = [{"date": d, value_key: v} for d, v in merged.items()]
    result.sort(key=lambda x: x["date"])
    return result
