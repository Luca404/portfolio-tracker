from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

# Allow pytest to import backend modules when running from backend/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.portfolio import aggregate_positions, calc_xirr
from utils.pricing import apply_splits_to_orders


def make_order(symbol: str, quantity: float, price: float, order_type: str, when: date, commission: float = 0.0):
    return SimpleNamespace(
        symbol=symbol,
        quantity=quantity,
        price=price,
        commission=commission,
        order_type=order_type,
        date=when,
    )


def test_aggregate_positions_accumulates_buy_orders():
    orders = [
        make_order("VWCE", 10, 100, "buy", date(2024, 1, 10), commission=1),
        make_order("VWCE", 5, 110, "buy", date(2024, 2, 10), commission=1),
    ]

    positions = aggregate_positions(orders)

    assert positions == {"VWCE": {"quantity": 15.0, "total_cost": 1552.0}}


def test_aggregate_positions_rejects_sell_beyond_holdings():
    orders = [
        make_order("AAPL", 3, 100, "buy", date(2024, 1, 10)),
        make_order("AAPL", 4, 110, "sell", date(2024, 2, 10)),
    ]

    with pytest.raises(HTTPException, match="Cannot sell 4 shares of AAPL; only 3.0 available"):
        aggregate_positions(orders)


def test_calc_xirr_returns_zero_for_one_sided_cashflows():
    cashflows = [
        (date(2024, 1, 1), -1000.0),
        (date(2024, 2, 1), -500.0),
    ]

    assert calc_xirr(cashflows) == 0.0


def test_calc_xirr_is_positive_for_simple_profitable_case():
    cashflows = [
        (date(2024, 1, 1), -1000.0),
        (date(2025, 1, 1), 1200.0),
    ]

    result = calc_xirr(cashflows)

    assert 0.19 < result < 0.21


def test_apply_splits_to_orders_adjusts_only_orders_before_split():
    orders = [
        make_order("NVDA", 10, 100, "buy", date(2024, 1, 1)),
        make_order("NVDA", 2, 120, "buy", date(2024, 7, 1)),
    ]
    splits = {"NVDA": {date(2024, 6, 10): 10.0}}

    adjusted = apply_splits_to_orders(orders, splits)

    assert adjusted[0].quantity == 100
    assert adjusted[0].price == 10
    assert adjusted[1].quantity == 2
    assert adjusted[1].price == 120


def test_apply_splits_to_orders_rejects_invalid_ratio():
    orders = [make_order("TSLA", 10, 100, "buy", date(2024, 1, 1))]
    splits = {"TSLA": {date(2024, 6, 10): 0.0}}

    with pytest.raises(ValueError, match="Invalid split ratio"):
        apply_splits_to_orders(orders, splits)
