from typing import List

import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import UserModel, PortfolioModel, OrderModel
from schemas import Order
from schemas.order import OptimizationRequest
from utils import (
    get_db,
    verify_token,
    commit_with_retry,
    format_date,
    format_datetime,
    validate_order_input,
    ensure_symbol_exists,
    aggregate_positions,
    get_risk_free_rate,
)

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post("")
def create_order(order: Order, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    validate_order_input(order)
    try:
        match = ensure_symbol_exists(order.symbol, order.instrument_type)
    except HTTPException:
        # bubble up validation errors, but allow missing key error through HTTPException
        raise
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == order.portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    existing_orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
    current_positions = aggregate_positions(existing_orders)
    symbol_positions = current_positions.get(order.symbol.upper(), {"quantity": 0, "total_cost": 0})

    if order.order_type == "sell" and order.quantity > symbol_positions["quantity"]:
        raise HTTPException(status_code=400, detail="Cannot sell more than current position")

    resolved_isin = order.isin or (match.get("isin") if match else "")
    resolved_ter = order.ter or (match.get("ter") if match else "")

    db_order = OrderModel(
        portfolio_id=portfolio.id,
        symbol=order.symbol.upper(),
        isin=resolved_isin,
        ter=resolved_ter,
        name=order.name or (match.get("name") if match else ""),
        exchange=order.exchange or (match.get("exchange") if match else ""),
        currency=order.currency or (match.get("currency") if match else ""),
        quantity=order.quantity,
        price=order.price,
        commission=order.commission,
        instrument_type=order.instrument_type.lower(),
        order_type=order.order_type,
        date=order.date,
    )
    db.add(db_order)
    commit_with_retry(db)
    db.refresh(db_order)

    return {
        "id": db_order.id,
        "portfolio_id": db_order.portfolio_id,
        "symbol": db_order.symbol,
        "isin": db_order.isin,
        "ter": db_order.ter,
        "name": db_order.name,
        "exchange": db_order.exchange,
        "currency": db_order.currency,
        "quantity": db_order.quantity,
        "price": db_order.price,
        "commission": db_order.commission,
        "instrument_type": db_order.instrument_type,
        "order_type": db_order.order_type,
        "date": format_date(db_order.date),
        "created_at": format_datetime(db_order.created_at),
    }


@router.get("/{portfolio_id}")
def get_orders(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
    return {
        "orders": [
            {
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
                "created_at": format_datetime(o.created_at),
            }
            for o in orders
        ]
    }


@router.delete("/{order_id}")
def delete_order(order_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    order = db.execute(
        select(OrderModel).join(PortfolioModel).where(OrderModel.id == order_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    db.delete(order)
    db.commit()
    return {"message": "Order deleted"}


@router.put("/{order_id}")
def update_order(order_id: int, updated: Order, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    validate_order_input(updated)
    try:
        match = ensure_symbol_exists(updated.symbol, updated.instrument_type)
    except HTTPException:
        raise

    order = db.execute(
        select(OrderModel).join(PortfolioModel).where(OrderModel.id == order_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    # Recompute positions excluding this order to validate sells
    existing_orders = (
        db.execute(
            select(OrderModel).where(OrderModel.portfolio_id == order.portfolio_id, OrderModel.id != order.id)
        ).scalars().all()
    )
    positions_before = aggregate_positions(existing_orders)
    symbol_positions = positions_before.get(updated.symbol.upper(), {"quantity": 0, "total_cost": 0})
    if updated.order_type == "sell" and updated.quantity > symbol_positions["quantity"]:
        raise HTTPException(status_code=400, detail="Cannot sell more than current position")

    resolved_isin = updated.isin or (match.get("isin") if match else "")
    resolved_ter = updated.ter or (match.get("ter") if match else "")

    order.symbol = updated.symbol.upper()
    order.isin = resolved_isin
    order.ter = resolved_ter
    order.name = updated.name or (match.get("name") if match else "")
    order.exchange = updated.exchange or (match.get("exchange") if match else "")
    order.currency = updated.currency or (match.get("currency") if match else "")
    order.quantity = updated.quantity
    order.price = updated.price
    order.commission = updated.commission
    order.instrument_type = updated.instrument_type.lower()
    order.order_type = updated.order_type
    order.date = updated.date

    commit_with_retry(db)
    db.refresh(order)

    return {
        "id": order.id,
        "portfolio_id": order.portfolio_id,
        "symbol": order.symbol,
        "isin": order.isin,
        "ter": order.ter,
        "name": order.name,
        "exchange": order.exchange,
        "currency": order.currency,
        "quantity": order.quantity,
        "price": order.price,
        "commission": order.commission,
        "instrument_type": order.instrument_type,
        "order_type": order.order_type,
        "date": format_date(order.date),
        "created_at": format_datetime(order.created_at),
    }


@router.post("/optimize")
def optimize_portfolio(request: OptimizationRequest, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    """Portfolio optimization using PyPortfolioOpt."""
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == request.portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    try:
        symbols = [s.upper() for s in request.symbols]
        data = yf.download(symbols, period="1y", progress=False)["Adj Close"]

        if data.empty:
            raise HTTPException(status_code=400, detail="No data available")

        # Get dynamic risk-free rate based on portfolio settings
        reference_currency = portfolio.reference_currency or "EUR"
        risk_free_rate_pct = get_risk_free_rate(reference_currency, db, portfolio.risk_free_source)
        risk_free_rate = risk_free_rate_pct / 100.0  # Convert from percentage to decimal

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
        orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
