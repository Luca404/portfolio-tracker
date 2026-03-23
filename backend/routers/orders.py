from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import date as dt_date, datetime
from types import SimpleNamespace

import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from schemas import Order
from schemas.order import OptimizationRequest
from utils import (
    get_db,
    get_supabase,
    verify_token,
    format_date,
    format_datetime,
    validate_order_input,
    ensure_symbol_exists,
    aggregate_positions,
    get_risk_free_rate,
)

router = APIRouter(prefix="/orders", tags=["orders"])


def _order_from_row(row: dict) -> SimpleNamespace:
    date_val = row.get("date")
    if isinstance(date_val, str):
        try:
            date_val = dt_date.fromisoformat(date_val)
        except ValueError:
            date_val = dt_date.today()
    return SimpleNamespace(
        id=row.get("id"),
        portfolio_id=row.get("portfolio_id"),
        symbol=(row.get("symbol") or "").upper(),
        isin=row.get("isin") or "",
        ter=row.get("ter") or "",
        name=row.get("name") or "",
        exchange=row.get("exchange") or "",
        currency=row.get("currency") or "",
        quantity=float(row.get("quantity") or 0),
        price=float(row.get("price") or 0),
        commission=float(row.get("commission") or 0),
        instrument_type=(row.get("instrument_type") or "stock").lower(),
        order_type=(row.get("order_type") or "buy").lower(),
        date=date_val,
        created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else None,
    )


def _order_response(o: SimpleNamespace) -> dict:
    return {
        "id": o.id,
        "portfolio_id": o.portfolio_id,
        "symbol": o.symbol,
        "isin": o.isin,
        "ter": o.ter,
        "name": o.name,
        "exchange": o.exchange,
        "currency": o.currency,
        "quantity": o.quantity,
        "price": o.price,
        "commission": o.commission,
        "instrument_type": o.instrument_type,
        "order_type": o.order_type,
        "date": format_date(o.date),
        "created_at": format_datetime(o.created_at) if o.created_at else None,
    }


@router.post("")
def create_order(order: Order, user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    validate_order_input(order)
    try:
        match = ensure_symbol_exists(order.symbol, order.instrument_type)
    except HTTPException:
        raise

    sb = get_supabase()
    pf_result = sb.table("portfolios").select("id").eq("id", order.portfolio_id).eq("user_id", user_id).execute()
    if not pf_result.data:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    existing_orders = [_order_from_row(r) for r in sb.table("orders").select("*").eq("portfolio_id", order.portfolio_id).execute().data]
    current_positions = aggregate_positions(existing_orders)
    symbol_positions = current_positions.get(order.symbol.upper(), {"quantity": 0})

    if order.order_type == "sell" and order.quantity > symbol_positions["quantity"]:
        raise HTTPException(status_code=400, detail="Cannot sell more than current position")

    resolved_isin = order.isin or (match.get("isin") if match else "")
    resolved_ter = order.ter or (match.get("ter") if match else "")

    result = sb.table("orders").insert({
        "portfolio_id": order.portfolio_id,
        "symbol": order.symbol.upper(),
        "isin": resolved_isin,
        "ter": resolved_ter,
        "name": order.name or (match.get("name") if match else ""),
        "exchange": order.exchange or (match.get("exchange") if match else ""),
        "currency": order.currency or (match.get("currency") if match else ""),
        "quantity": order.quantity,
        "price": order.price,
        "commission": order.commission,
        "instrument_type": order.instrument_type.lower(),
        "order_type": order.order_type,
        "date": order.date.isoformat() if hasattr(order.date, "isoformat") else str(order.date),
    }).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create order")
    return _order_response(_order_from_row(result.data[0]))


@router.get("/{portfolio_id}")
def get_orders(portfolio_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    pf_result = sb.table("portfolios").select("id").eq("id", portfolio_id).eq("user_id", user_id).execute()
    if not pf_result.data:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    orders_data = sb.table("orders").select("*").eq("portfolio_id", portfolio_id).execute().data
    return {"orders": [_order_response(_order_from_row(r)) for r in orders_data]}


@router.delete("/{order_id}")
def delete_order(order_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    order_result = sb.table("orders").select("portfolio_id").eq("id", order_id).execute()
    if not order_result.data:
        raise HTTPException(status_code=404, detail="Order not found")
    portfolio_id = order_result.data[0]["portfolio_id"]
    pf_result = sb.table("portfolios").select("id").eq("id", portfolio_id).eq("user_id", user_id).execute()
    if not pf_result.data:
        raise HTTPException(status_code=404, detail="Order not found")
    sb.table("orders").delete().eq("id", order_id).execute()
    return {"message": "Order deleted"}


@router.put("/{order_id}")
def update_order(order_id: int, updated: Order, user_id: str = Depends(verify_token)):
    validate_order_input(updated)
    try:
        match = ensure_symbol_exists(updated.symbol, updated.instrument_type)
    except HTTPException:
        raise

    sb = get_supabase()
    order_result = sb.table("orders").select("*").eq("id", order_id).execute()
    if not order_result.data:
        raise HTTPException(status_code=404, detail="Order not found")
    portfolio_id = order_result.data[0]["portfolio_id"]

    pf_result = sb.table("portfolios").select("id").eq("id", portfolio_id).eq("user_id", user_id).execute()
    if not pf_result.data:
        raise HTTPException(status_code=404, detail="Order not found")

    other_orders = [_order_from_row(r) for r in sb.table("orders").select("*").eq("portfolio_id", portfolio_id).neq("id", order_id).execute().data]
    symbol_positions = aggregate_positions(other_orders).get(updated.symbol.upper(), {"quantity": 0})
    if updated.order_type == "sell" and updated.quantity > symbol_positions["quantity"]:
        raise HTTPException(status_code=400, detail="Cannot sell more than current position")

    resolved_isin = updated.isin or (match.get("isin") if match else "")
    resolved_ter = updated.ter or (match.get("ter") if match else "")

    result = sb.table("orders").update({
        "symbol": updated.symbol.upper(),
        "isin": resolved_isin,
        "ter": resolved_ter,
        "name": updated.name or (match.get("name") if match else ""),
        "exchange": updated.exchange or (match.get("exchange") if match else ""),
        "currency": updated.currency or (match.get("currency") if match else ""),
        "quantity": updated.quantity,
        "price": updated.price,
        "commission": updated.commission,
        "instrument_type": updated.instrument_type.lower(),
        "order_type": updated.order_type,
        "date": updated.date.isoformat() if hasattr(updated.date, "isoformat") else str(updated.date),
    }).eq("id", order_id).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to update order")
    return _order_response(_order_from_row(result.data[0]))


@router.post("/optimize")
def optimize_portfolio(request: OptimizationRequest, user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    sb = get_supabase()
    pf_result = sb.table("portfolios").select("*").eq("id", request.portfolio_id).eq("user_id", user_id).execute()
    if not pf_result.data:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    portfolio = pf_result.data[0]

    try:
        symbols = [s.upper() for s in request.symbols]
        data = yf.download(symbols, period="1y", progress=False)["Adj Close"]
        if data.empty:
            raise HTTPException(status_code=400, detail="No data available")

        reference_currency = portfolio.get("reference_currency") or "EUR"
        risk_free_rate = get_risk_free_rate(reference_currency, db, portfolio.get("risk_free_source", "auto")) / 100.0

        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)

        if request.optimization_type == "max_sharpe":
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif request.optimization_type == "min_volatility":
            ef.min_volatility()
        else:
            ef.efficient_risk(target_volatility=0.15)

        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

        latest_prices = get_latest_prices(data)
        orders = [_order_from_row(r) for r in sb.table("orders").select("*").eq("portfolio_id", request.portfolio_id).execute().data]
        positions_map = aggregate_positions(orders)
        total_value = sum(
            pos["quantity"] * latest_prices.get(symbol, 0)
            for symbol, pos in positions_map.items()
            if pos["quantity"] > 0 and symbol in latest_prices
        )

        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_value)
        allocation, leftover = da.greedy_portfolio()

        return {
            "weights": {k: round(v, 4) for k, v in cleaned_weights.items() if v > 0.001},
            "allocation": allocation,
            "leftover": round(leftover, 2),
            "expected_return": round(perf[0] * 100, 2),
            "volatility": round(perf[1] * 100, 2),
            "sharpe_ratio": round(perf[2], 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
