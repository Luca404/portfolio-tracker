"""Regressions for current prices disagreeing with chronologically sorted charts."""

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import Base, ETFPriceCacheModel
from utils.cache import get_latest_history_date, merge_historical_data
from utils.portfolio import aggregate_portfolio_history, compute_portfolio_value
from utils.pricing import get_etf_price_and_history


@pytest.mark.parametrize("value_key", ["price", "rate"])
def test_merge_uses_calendar_order_and_replaces_same_day(value_key):
    cached = [
        {"date": "31-12-2025", value_key: 82.96},
        {"date": "01-10-2026", value_key: 90},
    ]
    fresh = [
        {"date": "12-10-2026", value_key: 98.95},
        {"date": "2026-10-01", value_key: 95},
    ]
    assert merge_historical_data(cached, fresh) == [
        {"date": "31-12-2025", value_key: 82.96},
        {"date": "01-10-2026", value_key: 95},
        {"date": "12-10-2026", value_key: 98.95},
    ]


@pytest.mark.parametrize("cached,fresh", [([], [{"date": "01-02-2026", "price": 2}, {"date": "31-01-2026", "price": 1}]),
                                         ([{"date": "01-02-2026", "price": 2}, {"date": "31-01-2026", "price": 1}], [])])
def test_merge_sorts_even_when_one_side_is_empty(cached, fresh):
    assert merge_historical_data(cached, fresh)[-1] == {"date": "01-02-2026", "price": 2}


def test_latest_date_does_not_trust_the_last_list_entry():
    assert get_latest_history_date([
        {"date": "12-10-2026", "price": 98.95},
        {"date": "not-a-date", "price": 0},
        {"date": "31-12-2025", "price": 82.96},
    ]) == date(2026, 10, 12)


def test_cached_etf_recovers_wrong_last_price_and_matches_chart(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    today = datetime.now(timezone.utc)
    # Synthetic EUR position: an older date sorts after today as plain text.
    history = [{"date": today.date().isoformat(), "price": 98.95},
               {"date": "31-12-2025", "price": 82.96}]
    with Session(engine) as db:
        cached = ETFPriceCacheModel(isin="TEST-ISIN", last_price=82.96,
                                   history_json=json.dumps(history), currency="EUR", updated_at=today.replace(tzinfo=None))
        db.add(cached)
        db.commit()

        def forbid_provider(*args, **kwargs):
            raise AssertionError("A fresh cached history must not fetch market data")

        monkeypatch.setattr("utils.pricing.yf.Ticker", forbid_provider)
        result = get_etf_price_and_history("TEST-ISIN", db)
        assert result["last_price"] == 98.95
        db.refresh(cached)
        assert cached.last_price == 98.95
        assert cached.updated_at == today.replace(tzinfo=None)

        orders = {"TEST": [SimpleNamespace(quantity=100, price=80, commission=0,
                                           order_type="buy", currency="EUR", date=date(2025, 1, 1))]}
        positions, total, cost, gain, percentage, histories = compute_portfolio_value(
            {"TEST": {"quantity": 100, "total_cost": 8000}}, orders,
            {"TEST": "etf"}, {"TEST": "TEST-ISIN"}, db,
            include_history=True, reference_currency="EUR",
        )
        chart, _, _ = aggregate_portfolio_history(histories, orders)
        assert total == pytest.approx(9895)
        assert positions[0]["market_value"] == pytest.approx(total)
        assert chart[-1]["value"] == pytest.approx(total)
        assert cost == 8000
        assert gain == pytest.approx(1895)
        assert percentage == pytest.approx(23.6875)
    engine.dispose()


@pytest.mark.parametrize("provider", ["justetf", "yfinance"])
def test_etf_refresh_uses_newest_quote_after_merging_old_december_data(monkeypatch, provider):
    today = datetime.now(timezone.utc)
    chart = pd.DataFrame({"date": [today.date()], "price": [98.95]})
    monkeypatch.setitem(sys.modules, "justetf_scraping", SimpleNamespace(
        load_chart=lambda identifier: chart if provider == "justetf" else None))
    market_data = pd.DataFrame({"Close": [98.95]}, index=pd.to_datetime([today.date()]))
    monkeypatch.setattr("utils.pricing.yf.Ticker", lambda symbol: SimpleNamespace(
        info={"symbol": "TEST"}, history=lambda **kwargs: market_data))
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        cached = ETFPriceCacheModel(isin="TEST-ISIN", last_price=82.96,
                                   history_json=json.dumps([{"date": "31-12-2025", "price": 82.96}]),
                                   currency="EUR", updated_at=(today - timedelta(days=7)).replace(tzinfo=None))
        db.add(cached)
        db.commit()
        result = get_etf_price_and_history("TEST-ISIN", db)
        assert result["last_price"] == 98.95
        assert result["history"][-1]["price"] == 98.95
        db.refresh(cached)
        assert cached.last_price == 98.95
    engine.dispose()
