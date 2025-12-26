"""Portfolio calculation and aggregation utilities."""

from datetime import date, datetime, timezone
from typing import List

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from models import OrderModel
from utils.dates import DATE_FMT, parse_date_input
from utils.pricing import (
    get_etf_price_and_history,
    get_stock_price_and_history_cached,
    convert_to_reference_currency,
    get_exchange_rate_history
)


def aggregate_positions(orders: List[OrderModel]):
    """
    Aggregate orders into current positions (quantity and cost basis).

    Args:
        orders: List of OrderModel instances

    Returns:
        Dictionary mapping symbol to position data (quantity, total_cost)
    """
    positions = {}
    for order in orders:
        symbol = order.symbol.upper()
        if symbol not in positions:
            positions[symbol] = {"quantity": 0.0, "total_cost": 0.0}
        pos = positions[symbol]
        if order.order_type == "buy":
            pos["quantity"] += order.quantity
            pos["total_cost"] += order.quantity * order.price + order.commission
        elif order.order_type == "sell":
            if order.quantity > pos["quantity"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot sell {order.quantity} shares of {symbol}; only {pos['quantity']} available",
                )
            pos["quantity"] -= order.quantity
            pos["total_cost"] -= order.quantity * order.price - order.commission
    positions = {k: v for k, v in positions.items() if v["quantity"] > 0}
    return positions


def xnpv(rate: float, cashflows: list):
    """
    Calculate Net Present Value for irregular cashflows.

    Args:
        rate: Discount rate
        cashflows: List of (date, amount) tuples

    Returns:
        Net present value
    """
    def _as_date(d):
        if isinstance(d, datetime):
            if d.tzinfo:
                return d.astimezone(timezone.utc).date()
            return d.date()
        if isinstance(d, date):
            return d
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            raise ValueError("invalid date")
        if getattr(ts, "tzinfo", None):
            ts = ts.tz_convert("UTC")
        return ts.date()

    if rate <= -1:
        return float("inf")
    try:
        cleaned = [(_as_date(d), cf) for d, cf in cashflows]
    except Exception:
        return float("inf")

    t0 = cleaned[0][0]
    return sum(cf / ((1 + rate) ** ((d - t0).days / 365.0)) for d, cf in cleaned)


def calc_xirr(cashflows: list):
    """
    Calculate Internal Rate of Return for irregular cashflows using Newton's method.

    Args:
        cashflows: List of (date, amount) tuples

    Returns:
        Annualized internal rate of return
    """
    if not cashflows:
        return 0.0

    cleaned = []
    for dt, cf in cashflows:
        try:
            if isinstance(dt, datetime):
                if dt.tzinfo:
                    dt = dt.astimezone(timezone.utc).date()
                else:
                    dt = dt.date()
            elif isinstance(dt, date):
                dt = dt
            else:
                ts = pd.to_datetime(dt, errors="coerce")
                if pd.isna(ts):
                    continue
                if hasattr(ts, "tzinfo") and ts.tzinfo:
                    ts = ts.tz_convert("UTC")
                dt = ts.date()
            cleaned.append((dt, cf))
        except Exception:
            continue

    if not cleaned:
        return 0.0

    positive = any(cf > 0 for _, cf in cleaned)
    negative = any(cf < 0 for _, cf in cleaned)
    if not (positive and negative):
        return 0.0

    try:
        low, high = -0.9999, 10.0
        for _ in range(100):
            mid = (low + high) / 2
            npv = xnpv(mid, cleaned)
            if abs(npv) < 1e-6:
                return mid
            if npv > 0:
                low = mid
            else:
                high = mid
        return mid
    except Exception:
        return 0.0


def compute_portfolio_value(
    positions_map: dict,
    orders_by_symbol: dict,
    symbol_type_map: dict,
    symbol_isin_map: dict,
    db: Session,
    include_history: bool = False,
    reference_currency: str = "EUR",
):
    """
    Compute current portfolio value with prices, gains/losses, and optional history.

    Args:
        positions_map: Dict of symbol -> {quantity, total_cost}
        orders_by_symbol: Dict of symbol -> list of orders
        symbol_type_map: Dict of symbol -> instrument_type
        symbol_isin_map: Dict of symbol -> ISIN
        db: Database session
        include_history: Whether to include price history
        reference_currency: Currency for totals

    Returns:
        Tuple of (positions, total_value, total_cost, total_gain_loss, total_gain_loss_pct, position_histories)
    """
    positions = []
    total_value = 0.0
    total_cost = 0.0
    position_histories = {}

    for symbol, data in positions_map.items():
        instrument_type = symbol_type_map.get(symbol, "stock")
        price_history = []
        current_price = 0.0
        fetch_error = None
        try:
            if instrument_type == "etf":
                isin = symbol_isin_map.get(symbol, "")
                etf_data = get_etf_price_and_history(isin, db)
                current_price = etf_data["last_price"]
                price_history = etf_data.get("history", [])
            else:
                stock_data = get_stock_price_and_history_cached(symbol, db, days=180 if include_history else 7)
                current_price = stock_data["last_price"]
                price_history = stock_data.get("history", [])
        except HTTPException as e:
            fetch_error = e.detail
            print(f"Error fetching {symbol}: {e.detail}")
        except Exception as e:
            fetch_error = str(e)
            print(f"Error fetching {symbol}: {e}")

        price_history = price_history or []
        current_price = current_price or 0.0

        # Conversione valuta se necessario
        # Determina valuta ufficiale asset vs valuta ordine
        orders_for_symbol = orders_by_symbol.get(symbol, [])
        if orders_for_symbol:
            order_currency = orders_for_symbol[0].currency or ""
            # Per ETF la valuta ufficiale è nella cache, per stock proviamo a determinarla
            # Per semplicità assumiamo che se non specificato, stock US = USD, altrimenti EUR
            # Idealmente dovremmo avere questo info dall'API
            if instrument_type == "etf":
                # Per ETF la currency è già disponibile dalla cache
                asset_currency = None  # Non abbiamo accesso diretto, skippiamo per ora gli ETF
            else:
                # Per stock, assumiamo USD se ticker US, altrimenti proviamo a determinare
                # Per sicurezza, solo convertiamo se sappiamo con certezza
                # Qui facciamo solo USD<->EUR per ora
                asset_currency = "USD" if order_currency == "EUR" else ("EUR" if order_currency == "USD" else None)

            # Converti se necessario
            if order_currency and asset_currency and order_currency != asset_currency and price_history:
                print(f"[FX] {symbol}: converting {asset_currency} -> {order_currency}")
                try:
                    fx_data = get_exchange_rate_history(asset_currency, order_currency, db)
                    fx_rates = fx_data.get("rates", [])

                    if fx_rates:
                        # Crea mappa data -> tasso
                        fx_map = {}
                        for rate_point in fx_rates:
                            rate_date = parse_date_input(rate_point.get("date"))
                            if rate_date:
                                fx_map[rate_date] = rate_point.get("rate", 1.0)

                        # Converti ogni prezzo storico
                        converted_history = []
                        for price_point in price_history:
                            price_date = parse_date_input(price_point.get("date"))
                            if price_date and price_date in fx_map:
                                original_price = price_point.get("price", 0.0)
                                fx_rate = fx_map[price_date]
                                converted_price = original_price * fx_rate
                                converted_history.append({
                                    "date": price_point.get("date"),
                                    "price": converted_price
                                })
                            else:
                                # Se non abbiamo tasso per quella data, skippiamo
                                pass

                        if converted_history:
                            price_history = converted_history
                            # Converti anche current_price usando ultimo tasso disponibile
                            if fx_map:
                                latest_rate = fx_rates[-1].get("rate", 1.0)
                                current_price = current_price * latest_rate
                                print(f"[FX] {symbol}: converted {len(converted_history)} prices, latest_rate={latest_rate:.4f}")
                except Exception as e:
                    print(f"[FX] {symbol}: conversion failed: {e}")

        if include_history:
            if instrument_type == "etf":
                print(f"[compute] ETF {symbol}: price={current_price}, history_len={len(price_history)}")
            else:
                print(f"[compute] Stock {symbol}: price={current_price}, history_len={len(price_history)}")

        try:
            avg_price = data["total_cost"] / data["quantity"] if data["quantity"] else 0
            market_value = data["quantity"] * current_price
            gain_loss = market_value - data["total_cost"]
            gain_loss_pct = (gain_loss / data["total_cost"] * 100) if data["total_cost"] > 0 else 0

            cashflows = []
            for o in orders_by_symbol.get(symbol, []):
                cf_amount = -(o.quantity * o.price + o.commission) if o.order_type == "buy" else (o.quantity * o.price - o.commission)
                cashflows.append((o.date, cf_amount))
            if data["quantity"] > 0:
                cashflows.append((date.today(), market_value))
            cashflows.sort(key=lambda x: x[0])
            try:
                xirr = calc_xirr(cashflows)
            except Exception as e:
                print(f"XIRR calc failed for {symbol}: {e}")
                xirr = 0.0

            currency = ""
            orders_for_symbol = orders_by_symbol.get(symbol, [])
            if orders_for_symbol:
                currency = orders_for_symbol[0].currency or ""

            # Converti alla valuta di riferimento per i totali
            market_value_ref = convert_to_reference_currency(market_value, currency, reference_currency, db)
            cost_basis_ref = convert_to_reference_currency(data["total_cost"], currency, reference_currency, db)

            positions.append(
                {
                    "symbol": symbol,
                    "quantity": round(data["quantity"], 4),
                    "avg_price": round(avg_price, 2),
                    "current_price": round(current_price, 2),
                    "market_value": round(market_value, 2),
                    "cost_basis": round(data["total_cost"], 2),
                    "gain_loss": round(gain_loss, 2),
                    "gain_loss_pct": round(gain_loss_pct, 2),
                    "instrument_type": instrument_type,
                    "currency": currency,
                    "xirr": round(xirr * 100, 2) if xirr else 0.0,
                    "fetch_error": fetch_error,
                }
            )
            if include_history and price_history:
                position_histories[symbol] = price_history

            # Usa valori convertiti per i totali
            total_value += market_value_ref
            total_cost += cost_basis_ref
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0

    return positions, total_value, total_cost, total_gain_loss, total_gain_loss_pct, position_histories


def aggregate_portfolio_history(position_histories: dict, orders_by_symbol: dict):
    """
    Aggrega il valore del portafoglio nel tempo usando solo le date in cui
    tutti gli strumenti con posizione > 0 hanno un prezzo disponibile.

    Args:
        position_histories: Dict of symbol -> price history
        orders_by_symbol: Dict of symbol -> list of orders

    Returns:
        Tuple of (aggregated_history, price_map, valid_dates)
    """
    if not position_histories:
        return [], {}, []

    price_map = {}
    all_dates = set()
    for symbol, history in position_histories.items():
        m = {}
        for point in history:
            d = parse_date_input(point.get("date"))
            if not d:
                continue
            try:
                m[d] = float(point.get("price"))
                all_dates.add(d)
            except Exception:
                continue
        price_map[symbol] = m

    if not all_dates:
        return [], price_map, []

    orders_sorted = {sym: sorted(orders_by_symbol.get(sym, []), key=lambda o: o.date) for sym in price_map.keys()}
    order_idx = {sym: 0 for sym in price_map.keys()}
    qty_map = {sym: 0.0 for sym in price_map.keys()}

    aggregated = []
    valid_dates = []
    for current_date in sorted(all_dates):
        total_value = 0.0
        any_position = False
        valid = True

        # applica ordini fino a questa data (inclusa)
        for sym, ords in orders_sorted.items():
            idx = order_idx[sym]
            while idx < len(ords) and ords[idx].date <= current_date:
                delta = ords[idx].quantity if ords[idx].order_type == "buy" else -ords[idx].quantity
                qty_map[sym] += delta
                idx += 1
            order_idx[sym] = idx

        for sym, prices in price_map.items():
            qty = qty_map.get(sym, 0.0)
            if qty <= 0:
                continue
            any_position = True
            price = prices.get(current_date)
            if price is None:
                valid = False
                break
            total_value += qty * price

        if valid and any_position:
            aggregated.append({"date": current_date.strftime(DATE_FMT), "value": round(total_value, 2)})
            valid_dates.append(current_date)

    return aggregated, price_map, valid_dates


def compute_portfolio_performance(price_map: dict, orders_by_symbol: dict, common_dates: list):
    """
    Calcola una curva di performance (TWR-like) basata sui rendimenti dei singoli asset
    pesati per il valore del giorno precedente, escludendo l'effetto dei flussi (ordini).

    Args:
        price_map: Dict of symbol -> date -> price
        orders_by_symbol: Dict of symbol -> list of orders
        common_dates: List of dates to calculate performance for

    Returns:
        List of performance data points
    """
    if not price_map or not common_dates:
        return []

    # ordini per simbolo ordinati per data
    orders_sorted = {sym: sorted(orders_by_symbol.get(sym, []), key=lambda o: o.date) for sym in price_map.keys()}
    order_idx = {sym: 0 for sym in price_map.keys()}
    qty_map = {sym: 0.0 for sym in price_map.keys()}

    perf = []
    nav = 100.0
    prev_date = None

    for current_date in common_dates:
        if prev_date is not None:
            total_prev_value = 0.0
            for sym, prices in price_map.items():
                prev_price = prices.get(prev_date)
                if prev_price and qty_map.get(sym, 0.0) > 0:
                    total_prev_value += qty_map[sym] * prev_price

            if total_prev_value > 0:
                daily_return = 0.0
                for sym, prices in price_map.items():
                    prev_price = prices.get(prev_date)
                    curr_price = prices.get(current_date)
                    qty = qty_map.get(sym, 0.0)
                    if qty > 0 and prev_price and curr_price:
                        weight = (qty * prev_price) / total_prev_value
                        asset_ret = (curr_price / prev_price) - 1
                        daily_return += weight * asset_ret
                nav *= (1 + daily_return)
                perf.append({"date": current_date.strftime(DATE_FMT), "value": round(nav, 4)})
        else:
            perf.append({"date": current_date.strftime(DATE_FMT), "value": round(nav, 4)})

        # applica ordini alla fine del giorno per la giornata successiva
        for sym, ords in orders_sorted.items():
            idx = order_idx[sym]
            while idx < len(ords) and ords[idx].date == current_date:
                delta = ords[idx].quantity if ords[idx].order_type == "buy" else -ords[idx].quantity
                qty_map[sym] = qty_map.get(sym, 0.0) + delta
                idx += 1
            order_idx[sym] = idx

        prev_date = current_date

    return perf
