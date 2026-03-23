from datetime import datetime, timezone, timedelta
from .dates import parse_date_input

MAX_DAYS_STALE = 1  # lasciato per compatibilità con import esistenti


def is_cache_data_fresh(history: list, updated_at=None) -> bool:
    """
    Cache fresca se è stata scaricata oggi (feriali) o venerdì/dopo (weekend).

    Con updated_at (preferito): controlla quando è stata scaricata.
    Senza updated_at (fallback): controlla la data dell'ultimo prezzo.
    """
    if not history:
        return False

    today = datetime.now(timezone.utc).date()
    weekday = today.weekday()  # 0=Lun, 5=Sab, 6=Dom

    # Data minima richiesta per considerare la cache fresca
    if weekday == 5:    # sabato → basta che sia stata scaricata venerdì
        required = today - timedelta(days=1)
    elif weekday == 6:  # domenica → basta che sia stata scaricata venerdì
        required = today - timedelta(days=2)
    else:               # lunedì–venerdì → deve essere stata scaricata oggi
        required = today

    if updated_at is not None:
        try:
            cache_date = updated_at.date() if isinstance(updated_at, datetime) else updated_at
            return cache_date >= required
        except Exception:
            return False

    # Fallback: controlla la data dell'ultimo prezzo in history
    try:
        latest_date_str = history[-1].get("date")
        if not latest_date_str:
            return False
        latest_date = parse_date_input(latest_date_str)
        return latest_date >= required if latest_date else False
    except Exception:
        return False


def merge_historical_data(cached_history: list, new_history: list) -> list:
    """
    Unisce i dati storici esistenti in cache con nuovi dati scaricati,
    evitando duplicati e mantenendo solo i dati più recenti per ogni data.
    """
    if not cached_history:
        return new_history

    if not new_history:
        return cached_history

    value_key = "price" if "price" in (new_history[0] if new_history else cached_history[0]) else "rate"

    merged = {}

    for item in cached_history:
        date_str = item.get("date")
        if date_str:
            merged[date_str] = item.get(value_key)

    for item in new_history:
        date_str = item.get("date")
        if date_str:
            merged[date_str] = item.get(value_key)

    result = [{"date": d, value_key: v} for d, v in merged.items()]
    result.sort(key=lambda x: x["date"])
    return result
