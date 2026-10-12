from datetime import datetime, timezone, timedelta
from .dates import DATE_FMT, parse_date_input

MAX_DAYS_STALE = 1  # lasciato per compatibilità con import esistenti


def get_latest_history_date(history: list):
    """Return the latest valid date present in cached history."""
    if not history:
        return None
    try:
        dates = (parse_date_input(point.get("date")) for point in history)
        return max((value for value in dates if value is not None), default=None)
    except Exception:
        return None


def get_backfill_period_from_latest_date(latest_date):
    """
    Heuristic refresh window:
    - no history / very old gap -> 5Y
    - medium gap -> 1Y
    - recent gap -> 1M
    """
    if not latest_date:
        return "5Y"

    today = datetime.now(timezone.utc).date()
    gap_days = max((today - latest_date).days, 0)
    if gap_days > 365:
        return "5Y"
    if gap_days > 31:
        return "1Y"
    return "1M"


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
        latest_date = get_latest_history_date(history)
        return latest_date >= required if latest_date else False
    except Exception:
        return False


def merge_historical_data(cached_history: list, new_history: list) -> list:
    """
    Unisce i dati storici esistenti in cache con nuovi dati scaricati,
    evitando duplicati e mantenendo solo i dati più recenti per ogni data.
    """
    if not cached_history and not new_history:
        return []

    value_key = "price" if "price" in (new_history[0] if new_history else cached_history[0]) else "rate"

    merged = {}

    # Parse dates before deduplication and sorting: DD-MM-YYYY text order can
    # select an old December price as the latest quote in October.
    for item in [*cached_history, *new_history]:
        day = parse_date_input(item.get("date"))
        if day is not None:
            merged[day] = item.get(value_key)

    return [{"date": day.strftime(DATE_FMT), value_key: merged[day]} for day in sorted(merged)]
