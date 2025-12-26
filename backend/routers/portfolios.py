from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import UserModel, PortfolioModel, OrderModel
from schemas import Portfolio
from utils import get_db, verify_token, format_datetime

# Import functions from main.py (they will be moved to services later)
# These imports will work because main.py is loaded before the routers are included
import main as main_module

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


@router.post("")
def create_portfolio(portfolio: Portfolio, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    db_portfolio = PortfolioModel(
        user_id=user.id,
        name=portfolio.name,
        description=portfolio.description,
        initial_capital=portfolio.initial_capital,
        reference_currency=portfolio.reference_currency or "EUR",
        risk_free_source=portfolio.risk_free_source or "auto",
        market_benchmark=portfolio.market_benchmark or "auto",
    )
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return {
        "id": db_portfolio.id,
        "user_email": user.email,
        "name": db_portfolio.name,
        "description": db_portfolio.description,
        "initial_capital": db_portfolio.initial_capital,
        "reference_currency": db_portfolio.reference_currency,
        "risk_free_source": db_portfolio.risk_free_source,
        "market_benchmark": db_portfolio.market_benchmark,
        "created_at": format_datetime(db_portfolio.created_at),
    }


@router.get("/count")
def get_portfolios_count(user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    """Lightweight endpoint to get just the portfolio count"""
    count = db.execute(
        select(PortfolioModel).where(PortfolioModel.user_id == user.id)
    ).scalars().all()
    return {"count": len(count)}


@router.get("")
def get_portfolios(user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    results = db.execute(select(PortfolioModel).where(PortfolioModel.user_id == user.id)).scalars().all()
    response = []
    for p in results:
        orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == p.id)).scalars().all()
        positions_map = main_module.aggregate_positions(orders)
        orders_by_symbol = {}
        symbol_type_map = {}
        symbol_isin_map = {}
        for o in orders:
            orders_by_symbol.setdefault(o.symbol.upper(), []).append(o)
            symbol_type_map[o.symbol.upper()] = (o.instrument_type or "stock").lower()
            symbol_isin_map[o.symbol.upper()] = (o.isin or "").upper()
        reference_currency = p.reference_currency or "EUR"
        _, total_value, total_cost, total_gain_loss, total_gain_loss_pct, _ = main_module.compute_portfolio_value(
            positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db, reference_currency=reference_currency
        )

        # Calculate asset composition (ETF vs STOCK)
        asset_composition = {"etf": 0, "stock": 0}
        for symbol, position in positions_map.items():
            if position['quantity'] > 0:
                asset_type = symbol_type_map.get(symbol, "stock").lower()
                if asset_type == "etf":
                    asset_composition["etf"] += position['quantity']
                else:
                    asset_composition["stock"] += position['quantity']

        response.append(
            {
                "id": p.id,
                "user_email": user.email,
                "name": p.name,
                "description": p.description,
                "initial_capital": p.initial_capital,
                "reference_currency": reference_currency,
                "risk_free_source": p.risk_free_source,
                "market_benchmark": p.market_benchmark,
                "created_at": format_datetime(p.created_at),
                "total_value": total_value,
                "total_cost": total_cost,
                "total_gain_loss": total_gain_loss,
                "total_gain_loss_pct": total_gain_loss_pct,
                "asset_composition": asset_composition,
            }
        )
    return {"portfolios": response}


@router.get("/{portfolio_id}")
def get_portfolio(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
    orders_by_symbol = {}
    symbol_type_map = {}
    symbol_isin_map = {}
    for o in orders:
        orders_by_symbol.setdefault(o.symbol.upper(), []).append(o)
        symbol_type_map[o.symbol.upper()] = (o.instrument_type or "stock").lower()
        symbol_isin_map[o.symbol.upper()] = (o.isin or "").upper()
    positions_map = main_module.aggregate_positions(orders)
    reference_currency = portfolio.reference_currency or "EUR"
    positions, total_value, total_cost, total_gain_loss, total_gain_loss_pct, position_histories = main_module.compute_portfolio_value(
        positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db, include_history=True, reference_currency=reference_currency
    )

    portfolio_history, price_map, common_dates = main_module.aggregate_portfolio_history(position_histories, orders_by_symbol)
    performance_history = main_module.compute_portfolio_performance(price_map, orders_by_symbol, common_dates)

    # Calculate portfolio XIRR (annualized return)
    # Include all cashflows (buys + sells) to get true money-weighted return
    portfolio_xirr = 0.0
    try:
        # Group by date to get net daily cashflow (handles same-day buy+sell)
        from collections import defaultdict
        daily_cashflows = defaultdict(float)

        for o in orders:
            # Convert to reference currency
            order_currency = o.currency or reference_currency
            if order_currency == reference_currency:
                rate = 1.0
            else:
                rate = main_module.convert_to_reference_currency(1.0, order_currency, reference_currency, db)

            if o.order_type == "buy":
                cf_amount = -(o.quantity * o.price + o.commission) * rate
            else:  # sell
                cf_amount = (o.quantity * o.price - o.commission) * rate

            daily_cashflows[o.date] += cf_amount

        cashflows = [(dt, amount) for dt, amount in daily_cashflows.items()]

        # Add current portfolio value as final cashflow
        if total_value > 0:
            from datetime import date as dt_date
            cashflows.append((dt_date.today(), total_value))

        if cashflows and len(cashflows) > 1:
            cashflows.sort(key=lambda x: x[0])

            # Total invested = sum of all negative flows
            # Total withdrawn = sum of positive flows (excluding final value)
            total_invested = sum(cf for _, cf in cashflows if cf < 0)
            total_withdrawn = sum(cf for _, cf in cashflows[:-1] if cf > 0)
            net_invested = total_invested + total_withdrawn

            # Calculate XIRR
            portfolio_xirr_raw = main_module.calc_xirr(cashflows)
            portfolio_xirr = portfolio_xirr_raw * 100

            # Simple return = (current_value + withdrawn) / invested - 1
            total_return_value = total_value + total_withdrawn
            simple_return = (total_return_value / abs(total_invested) - 1) * 100 if total_invested != 0 else 0

            print(f"[XIRR] Portfolio {portfolio_id}: invested={total_invested:.2f}, withdrawn={total_withdrawn:.2f}, current={total_value:.2f}, total_return={total_return_value:.2f}, simple_return={simple_return:.2f}%, XIRR={portfolio_xirr:.2f}%")
    except Exception as e:
        print(f"[XIRR] Error calculating portfolio XIRR: {e}")
        import traceback
        traceback.print_exc()
        portfolio_xirr = 0.0

    return {
        "portfolio": {
            "id": portfolio.id,
            "user_email": user.email,
            "name": portfolio.name,
            "description": portfolio.description,
            "initial_capital": portfolio.initial_capital,
            "reference_currency": reference_currency,
            "risk_free_source": portfolio.risk_free_source or "auto",
            "market_benchmark": portfolio.market_benchmark or "auto",
            "created_at": format_datetime(portfolio.created_at),
        },
        "positions": positions,
        "summary": {
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain_loss": round(total_gain_loss, 2),
            "total_gain_loss_pct": round(total_gain_loss_pct, 2),
            "portfolio_xirr": round(portfolio_xirr, 2),
            "reference_currency": reference_currency,
        },
        "history": {
            "portfolio": portfolio_history,
            "performance": performance_history,
            "positions": position_histories,
        },
        "last_updated": format_datetime(main_module.datetime.now(main_module.timezone.utc)),
    }


@router.put("/{portfolio_id}")
def update_portfolio(portfolio_id: int, portfolio_update: Portfolio, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Aggiorna campi
    if portfolio_update.name:
        portfolio.name = portfolio_update.name
    if portfolio_update.description is not None:
        portfolio.description = portfolio_update.description
    if portfolio_update.reference_currency:
        portfolio.reference_currency = portfolio_update.reference_currency
    if portfolio_update.risk_free_source is not None:
        portfolio.risk_free_source = portfolio_update.risk_free_source
    if portfolio_update.market_benchmark is not None:
        portfolio.market_benchmark = portfolio_update.market_benchmark

    db.commit()
    db.refresh(portfolio)
    return {
        "id": portfolio.id,
        "user_email": user.email,
        "name": portfolio.name,
        "description": portfolio.description,
        "reference_currency": portfolio.reference_currency,
        "risk_free_source": portfolio.risk_free_source,
        "market_benchmark": portfolio.market_benchmark,
        "created_at": format_datetime(portfolio.created_at),
    }


@router.delete("/{portfolio_id}")
def delete_portfolio(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    db.delete(portfolio)
    db.commit()
    return {"message": "Portfolio deleted"}
