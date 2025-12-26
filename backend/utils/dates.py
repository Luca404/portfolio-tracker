from datetime import date, datetime
from typing import Optional

import pandas as pd

DATE_FMT = "%d-%m-%Y"


def format_date(d: Optional[date]):
    """Format a date object to string."""
    return d.strftime(DATE_FMT) if d else None


def format_datetime(dt: Optional[datetime]):
    """Format a datetime object to string."""
    return dt.strftime(f"{DATE_FMT} %H:%M:%S") if dt else None


def parse_date_input(val) -> Optional[date]:
    """
    Converte una data in date (day-first) cercando vari formati comuni.
    """
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        for fmt in [DATE_FMT, "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"]:
            try:
                return datetime.strptime(val, fmt).date()
            except Exception:
                continue
        try:
            return pd.to_datetime(val, dayfirst=True).date()
        except Exception:
            return None
    try:
        return pd.to_datetime(val, dayfirst=True).date()
    except Exception:
        return None
