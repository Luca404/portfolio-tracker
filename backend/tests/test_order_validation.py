from datetime import date
from pathlib import Path
import sys

import pytest
from fastapi import HTTPException

# Allow pytest to import backend modules when running from backend/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from schemas.order import Order
from utils.helpers import validate_order_input


def make_order(**overrides):
    base = {
        "portfolio_id": 1,
        "symbol": "VWCE",
        "quantity": 1,
        "price": 100,
        "commission": 0,
        "instrument_type": "etf",
        "order_type": "buy",
        "date": date(2024, 1, 1),
    }
    base.update(overrides)
    return Order(**base)


def test_validate_order_input_accepts_valid_order():
    order = make_order()

    validate_order_input(order)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("quantity", 0, "Quantity must be positive"),
        ("price", 0, "Price must be positive"),
        ("commission", -1, "Commission cannot be negative"),
        ("order_type", "hold", "order_type must be 'buy' or 'sell'"),
        ("instrument_type", "crypto", "instrument_type must be one of: stock, etf"),
    ],
)
def test_validate_order_input_rejects_invalid_values(field, value, message):
    order = make_order(**{field: value})

    with pytest.raises(HTTPException, match=message):
        validate_order_input(order)
