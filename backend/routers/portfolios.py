from fastapi import APIRouter, Depends, HTTPException
from datetime import date as dt_date, datetime, timezone
from collections import defaultdict
from sqlalchemy import select
from sqlalchemy.orm import Session

import numpy as np
import pandas as pd
import yfinance as yf

from models import UserModel, PortfolioModel, OrderModel
from schemas import Portfolio
from utils import (
    get_db,
    verify_token,
    format_datetime,
    aggregate_positions,
    compute_portfolio_value,
    aggregate_portfolio_history,
    compute_portfolio_performance,
    calc_xirr,
    convert_to_reference_currency,
    parse_date_input,
    DATE_FMT,
    get_stock_splits,
    apply_splits_to_orders,
)

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


def apply_stock_splits_to_orders(orders):
    """
    Helper function per applicare gli stock splits agli ordini.
    Recupera automaticamente gli splits e li applica.

    Args:
        orders: Lista di OrderModel

    Returns:
        Lista di ordini con splits applicati (copie con valori aggiustati)
    """
    symbol_splits = {}
    stock_symbols = set()
    first_order_dates = {}

    for o in orders:
        symbol = o.symbol.upper()
        instrument_type = (o.instrument_type or "stock").lower()

        # Teniamo traccia solo degli stock, non degli ETF
        if instrument_type == "stock":
            stock_symbols.add(symbol)
            if symbol not in first_order_dates or o.date < first_order_dates[symbol]:
                first_order_dates[symbol] = o.date

    # Recupera gli splits per ogni stock
    for symbol in stock_symbols:
        splits = get_stock_splits(symbol, first_order_dates[symbol])
        if splits:
            symbol_splits[symbol] = splits

    # Applica gli splits agli ordini
    if symbol_splits:
        print(f"[SPLITS] Found splits for {len(symbol_splits)} symbols, applying adjustments...")
        return apply_splits_to_orders(list(orders), symbol_splits)

    return list(orders)


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

        # Applica gli stock splits agli ordini
        orders = apply_stock_splits_to_orders(orders)

        positions_map = aggregate_positions(orders)
        orders_by_symbol = {}
        symbol_type_map = {}
        symbol_isin_map = {}
        for o in orders:
            orders_by_symbol.setdefault(o.symbol.upper(), []).append(o)
            symbol_type_map[o.symbol.upper()] = (o.instrument_type or "stock").lower()
            symbol_isin_map[o.symbol.upper()] = (o.isin or "").upper()
        reference_currency = p.reference_currency or "EUR"
        _, total_value, total_cost, total_gain_loss, total_gain_loss_pct, _ = compute_portfolio_value(
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

    # Applica gli stock splits agli ordini
    orders = apply_stock_splits_to_orders(orders)

    orders_by_symbol = {}
    symbol_type_map = {}
    symbol_isin_map = {}
    for o in orders:
        orders_by_symbol.setdefault(o.symbol.upper(), []).append(o)
        symbol_type_map[o.symbol.upper()] = (o.instrument_type or "stock").lower()
        symbol_isin_map[o.symbol.upper()] = (o.isin or "").upper()
    positions_map = aggregate_positions(orders)
    reference_currency = portfolio.reference_currency or "EUR"
    positions, total_value, total_cost, total_gain_loss, total_gain_loss_pct, position_histories = compute_portfolio_value(
        positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db, include_history=True, reference_currency=reference_currency
    )

    portfolio_history, price_map, common_dates = aggregate_portfolio_history(position_histories, orders_by_symbol)
    performance_history = compute_portfolio_performance(price_map, orders_by_symbol, common_dates)

    # Calculate portfolio XIRR (annualized return)
    # Include all cashflows (buys + sells) to get true money-weighted return
    portfolio_xirr = 0.0
    try:
        # Group by date to get net daily cashflow (handles same-day buy+sell)
        daily_cashflows = defaultdict(float)

        for o in orders:
            # Convert to reference currency
            order_currency = o.currency or reference_currency
            if order_currency == reference_currency:
                rate = 1.0
            else:
                rate = convert_to_reference_currency(1.0, order_currency, reference_currency, db)

            if o.order_type == "buy":
                cf_amount = -(o.quantity * o.price + o.commission) * rate
            else:  # sell
                cf_amount = (o.quantity * o.price - o.commission) * rate

            daily_cashflows[o.date] += cf_amount

        cashflows = [(dt, amount) for dt, amount in daily_cashflows.items()]

        # Add current portfolio value as final cashflow
        if total_value > 0:
            cashflows.append((dt_date.today(), total_value))

        if cashflows and len(cashflows) > 1:
            cashflows.sort(key=lambda x: x[0])

            # Total invested = sum of all negative flows
            # Total withdrawn = sum of positive flows (excluding final value)
            total_invested = sum(cf for _, cf in cashflows if cf < 0)
            total_withdrawn = sum(cf for _, cf in cashflows[:-1] if cf > 0)
            net_invested = total_invested + total_withdrawn

            # Calculate XIRR
            portfolio_xirr_raw = calc_xirr(cashflows)
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
        "last_updated": format_datetime(datetime.now(timezone.utc)),
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


@router.get("/history/{portfolio_id}/{symbol}")
def get_position_history(
    portfolio_id: int, symbol: str, period: str = "1y", user: UserModel = Depends(verify_token), db: Session = Depends(get_db)
):
    """Get historical price data for a symbol in a portfolio."""
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        history = []
        for idx, row in hist.iterrows():
            history.append({"date": idx.strftime(DATE_FMT), "close": round(row["Close"], 2), "volume": int(row["Volume"])})

        return {"symbol": symbol, "history": history}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/{portfolio_id}")
def analyze_portfolio(
    portfolio_id: int,
    monte_carlo_years: int = 1,
    user: UserModel = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Advanced portfolio analysis including:
    - Correlation matrix
    - Monte Carlo simulation (configurable projection years)
    - Risk metrics (VaR, CVaR, Sharpe, Sortino, etc.)
    - Drawdown analysis

    Query params:
    - monte_carlo_years: Number of years for Monte Carlo projection (default: 1, max: 30)
    """
    # Validate monte_carlo_years
    if monte_carlo_years < 1 or monte_carlo_years > 30:
        raise HTTPException(status_code=400, detail="monte_carlo_years must be between 1 and 30")
    portfolio = db.execute(
        select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user.id)
    ).scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    orders = db.execute(select(OrderModel).where(OrderModel.portfolio_id == portfolio.id)).scalars().all()
    if not orders:
        return {
            "message": "No orders in portfolio",
            "correlation": None,
            "montecarlo": None,
            "risk_metrics": None,
            "drawdown": None
        }

    # Applica gli stock splits agli ordini
    orders = apply_stock_splits_to_orders(orders)

    # Get positions and historical data from cache
    positions_map = aggregate_positions(orders)
    orders_by_symbol = {}
    symbol_type_map = {}
    symbol_isin_map = {}
    for o in orders:
        orders_by_symbol.setdefault(o.symbol.upper(), []).append(o)
        symbol_type_map[o.symbol.upper()] = (o.instrument_type or "stock").lower()
        symbol_isin_map[o.symbol.upper()] = (o.isin or "").upper()

    reference_currency = portfolio.reference_currency or "EUR"
    positions, total_value, total_cost, _, _, position_histories = compute_portfolio_value(
        positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db,
        include_history=True, reference_currency=reference_currency
    )

    if len(positions) < 2:
        return {
            "message": "Need at least 2 assets for analysis",
            "correlation": None,
            "montecarlo": None,
            "risk_metrics": None,
            "drawdown": None
        }

    try:
        # Use cached historical data instead of downloading again
        if not position_histories:
            return {
                "message": "Insufficient historical data",
                "correlation": None,
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Build a unified DataFrame from cached price histories
        # Only include symbols that are currently in the portfolio
        current_symbols = {p["symbol"] for p in positions}
        filtered_position_histories = {sym: hist for sym, hist in position_histories.items() if sym in current_symbols}

        # Find common dates across all symbols
        all_dates_sets = []
        for symbol, history in filtered_position_histories.items():
            dates = set()
            for point in history:
                d = parse_date_input(point.get("date"))
                if d:
                    dates.add(d)
            all_dates_sets.append(dates)

        if not all_dates_sets:
            return {
                "message": "No historical data available",
                "correlation": None,
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Get common dates (intersection)
        common_dates = set.intersection(*all_dates_sets) if len(all_dates_sets) > 1 else all_dates_sets[0]

        # Find earliest order date to ensure drawdown starts from portfolio inception
        earliest_order_date = min(parse_date_input(o.date) for o in orders)
        print(f"[DRAWDOWN] Earliest order date: {earliest_order_date.strftime(DATE_FMT)}")

        # Filter common_dates to only include dates from earliest order onwards
        common_dates = {d for d in common_dates if d >= earliest_order_date}

        if len(common_dates) < 30:
            return {
                "message": "Insufficient overlapping historical data (need at least 30 days)",
                "correlation": None,
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Sort dates
        sorted_dates = sorted(list(common_dates))
        print(f"[DRAWDOWN] Analysis period: {sorted_dates[0].strftime(DATE_FMT)} to {sorted_dates[-1].strftime(DATE_FMT)} ({len(sorted_dates)} days)")

        # Build price DataFrame (only for current positions)
        price_data = {}
        for symbol, history in filtered_position_histories.items():
            prices_map = {}
            for point in history:
                d = parse_date_input(point.get("date"))
                if d and d in common_dates:
                    try:
                        prices_map[d] = float(point.get("price", 0))
                    except:
                        pass
            price_data[symbol] = prices_map

        # Create DataFrame
        data_dict = {}
        for symbol in price_data:
            data_dict[symbol] = [price_data[symbol].get(d, np.nan) for d in sorted_dates]

        # Convert sorted_dates to pandas datetime for proper alignment
        data = pd.DataFrame(data_dict, index=pd.to_datetime(sorted_dates))
        data = data.dropna()

        if len(data) < 30:
            return {
                "message": "Insufficient clean historical data after removing gaps",
                "correlation": None,
                "montecarlo": None,
                "risk_metrics": None,
                "drawdown": None
            }

        # Calculate returns
        returns = data.pct_change(fill_method=None).dropna()

        # 1. Correlation Matrix
        corr_matrix = returns.corr().values.tolist()

        # Calculate portfolio weights
        weights = np.array([p["market_value"] / total_value if total_value > 0 else 0 for p in positions])

        # Ensure weights match data columns
        position_symbols = [p["symbol"] for p in positions]
        ordered_weights = []
        for sym in data.columns:
            if sym in position_symbols:
                idx = position_symbols.index(sym)
                ordered_weights.append(weights[idx])
            else:
                ordered_weights.append(0.0)

        weights = np.array(ordered_weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # 2. Risk Metrics
        sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0
        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino = (portfolio_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

        # VaR and CVaR - daily values in percentage
        var_95 = np.percentile(portfolio_returns, 5) * 100  # Convert to percentage
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100

        # Annualized return for Calmar calculation
        annualized_return = portfolio_returns.mean() * 252 * 100  # in percentage

        risk_metrics = {
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "var_95": round(float(var_95), 4),  # Changed from value_at_risk_95
            "cvar_95": round(float(cvar_95), 4),  # Changed from conditional_var_95
            "volatility": round(float(portfolio_returns.std() * np.sqrt(252)), 4),
            "max_drawdown": 0.0,  # Will be updated after drawdown calculation
            "calmar_ratio": 0.0,  # Will be calculated after max_drawdown
            "beta": None,  # Will try to calculate vs benchmark
            "beta_vs": None  # Benchmark name
        }

        # 3. Monte Carlo Simulation
        num_simulations = 1000
        num_days = 252 * monte_carlo_years  # Scale by number of years
        simulations = []
        for _ in range(num_simulations):
            sim_returns = np.random.choice(portfolio_returns, size=num_days, replace=True)
            sim_cumulative = (1 + sim_returns).cumprod()
            simulations.append(sim_cumulative.tolist())

        # Calculate percentiles for each day across all simulations
        simulations_array = np.array(simulations)  # shape: (1000, num_days)

        # Compute percentiles across simulations for each day
        p5_array = np.percentile(simulations_array, 5, axis=0)
        p25_array = np.percentile(simulations_array, 25, axis=0)
        p50_array = np.percentile(simulations_array, 50, axis=0)
        p75_array = np.percentile(simulations_array, 75, axis=0)
        p95_array = np.percentile(simulations_array, 95, axis=0)

        # Generate date labels for the simulation period
        from datetime import timedelta
        today = dt_date.today()
        simulation_dates = [(today + timedelta(days=i)).strftime(DATE_FMT) for i in range(num_days)]

        montecarlo = {
            "simulations": simulations[:10],  # Return only first 10 for bandwidth
            "dates": simulation_dates,
            "current_value": float(total_value),
            "years": monte_carlo_years,  # Include years in response
            "percentiles": {
                "p5": [round(float(v * total_value), 2) for v in p5_array],
                "p25": [round(float(v * total_value), 2) for v in p25_array],
                "p50": [round(float(v * total_value), 2) for v in p50_array],
                "p75": [round(float(v * total_value), 2) for v in p75_array],
                "p95": [round(float(v * total_value), 2) for v in p95_array],
            }
        }

        # 4. Drawdown Analysis
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min())
        risk_metrics["max_drawdown"] = round(max_dd, 4)

        # Calculate Calmar Ratio (annualized return / abs(max_drawdown))
        if max_dd != 0:
            calmar = annualized_return / abs(max_dd * 100)  # max_dd is negative decimal, convert to positive percentage
            risk_metrics["calmar_ratio"] = round(float(calmar), 4)

        drawdown_data = [
            {"date": idx.strftime(DATE_FMT), "drawdown": round(float(val), 4)}
            for idx, val in drawdown.items()
        ]

        # 5. Calculate Beta vs benchmark (if available)
        try:
            # Get benchmark symbol - handle AUTO case
            benchmark_symbol = portfolio.market_benchmark
            print(f"[BETA] Original benchmark from portfolio: {benchmark_symbol}")

            if not benchmark_symbol or (isinstance(benchmark_symbol, str) and benchmark_symbol.upper() == "AUTO"):
                # Use default based on currency
                if reference_currency == "USD":
                    benchmark_symbol = "SPY"  # S&P 500 ETF
                else:
                    benchmark_symbol = "VWCE.DE"  # World ETF
                print(f"[BETA] Using default benchmark for {reference_currency}: {benchmark_symbol}")

            risk_metrics["beta_vs"] = benchmark_symbol

            # Try to get benchmark data
            import utils.pricing as pricing_utils
            benchmark_data = pricing_utils.get_stock_price_and_history_cached(benchmark_symbol, db, days=len(data))
            if benchmark_data and benchmark_data.get("history"):
                # Build benchmark returns
                benchmark_prices = {}
                for point in benchmark_data["history"]:
                    d = parse_date_input(point.get("date"))
                    if d:
                        benchmark_prices[d] = float(point.get("price", 0))

                # Align benchmark with portfolio dates
                benchmark_series = [benchmark_prices.get(d, np.nan) for d in sorted_dates]
                benchmark_df = pd.Series(benchmark_series, index=pd.to_datetime(sorted_dates))
                benchmark_df = benchmark_df.dropna()

                # Calculate benchmark returns
                benchmark_returns = benchmark_df.pct_change(fill_method=None).dropna()

                # Align dates
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 30:
                    pf_aligned = portfolio_returns.loc[common_index]
                    bm_aligned = benchmark_returns.loc[common_index]

                    # Beta = Cov(portfolio, benchmark) / Var(benchmark)
                    covariance = np.cov(pf_aligned, bm_aligned)[0, 1]
                    variance = np.var(bm_aligned)
                    if variance > 0:
                        beta = covariance / variance
                        risk_metrics["beta"] = round(float(beta), 4)
        except Exception as e:
            print(f"Beta calculation failed: {str(e)}")
            # Keep beta as None

        # 6. Asset Contribution (formerly Performance Attribution)
        assets_attribution = []
        print(f"\n[ATTRIBUTION] Calculating contribution for {len(position_symbols)} assets")
        for symbol in data.columns:
            if symbol in position_symbols:
                idx = position_symbols.index(symbol)
                position = positions[idx]

                # Calculate contribution to portfolio return
                asset_returns = returns[symbol]
                weight = weights[idx]
                contribution = (asset_returns * weight).sum() * 100  # Percentage contribution
                total_return = asset_returns.sum() * 100  # Total return percentage

                print(f"[ATTRIBUTION] {symbol}: weight={weight*100:.2f}%, "
                      f"market_value={position['market_value']:.2f}, "
                      f"gain_loss={position.get('gain_loss', 0):.2f}, "
                      f"total_return={total_return:.2f}%, "
                      f"contribution={contribution:.2f}%")

                assets_attribution.append({
                    "symbol": symbol,
                    "weight": round(float(weight * 100), 2),  # Percentage
                    "current_value": round(float(position["market_value"]), 2),
                    "annualized_return": round(float(asset_returns.mean() * 252 * 100), 2),  # Annualized %
                    "total_return_pct": round(float(total_return), 2),  # Total return %
                    "contribution_to_portfolio": round(float(contribution), 2),  # Total contribution %
                    "volatility": round(float(asset_returns.std() * np.sqrt(252)), 4),
                    "gain_loss": round(float(position.get("gain_loss", 0)), 2)
                })

        # Sort by contribution (descending)
        assets_attribution.sort(key=lambda x: x["contribution_to_portfolio"], reverse=True)

        # Calculate portfolio total return
        portfolio_total_return = (portfolio_returns.sum() * 100) if len(portfolio_returns) > 0 else 0

        # Get worst contributors (sorted ascending by contribution)
        worst_sorted = sorted(assets_attribution, key=lambda x: x["contribution_to_portfolio"])

        performance_attribution = {
            "portfolio_total_return": round(float(portfolio_total_return), 2),
            "top_contributors": assets_attribution[:3] if len(assets_attribution) >= 3 else assets_attribution,
            "worst_contributors": worst_sorted[:3] if len(worst_sorted) > 0 else [],
            "assets": assets_attribution
        }

        # 6. Historical prices for all assets (normalized to 100 at start)
        # Use the same sorted_dates from analysis
        historical_prices = {
            "dates": [d.strftime(DATE_FMT) for d in sorted_dates],
            "assets": {}
        }

        for symbol in data.columns:
            # Get prices for this symbol
            prices = data[symbol].values
            # Normalize to 100 at start date
            if prices[0] > 0:
                normalized = (prices / prices[0] * 100).tolist()
                historical_prices["assets"][symbol] = [round(float(p), 2) for p in normalized]

        return {
            "message": "Analysis complete",
            "correlation": {
                "matrix": corr_matrix,
                "symbols": list(data.columns)
            },
            "montecarlo": montecarlo,
            "risk_metrics": risk_metrics,
            "drawdown": drawdown_data,  # All drawdown history
            "performance_attribution": performance_attribution,
            "historical_prices": historical_prices  # Historical price chart data
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/compare/{portfolio_id}")
def compare_dca_vs_lumpsum(
    portfolio_id: int,
    user: UserModel = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Compare DCA (Dollar Cost Averaging) strategy vs Lump Sum investment.

    DCA: The actual strategy used (buying assets over time as orders were placed)
    Lump Sum: What would have happened if all current asset quantities were bought
              on the date of the first order in the portfolio
    """
    # Get portfolio
    pf = db.scalar(select(PortfolioModel).where(
        PortfolioModel.id == portfolio_id,
        PortfolioModel.user_id == user.id
    ))
    if not pf:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Get all orders
    orders = list(db.scalars(
        select(OrderModel)
        .where(OrderModel.portfolio_id == portfolio_id)
        .order_by(OrderModel.date)
    ).all())

    if not orders:
        raise HTTPException(status_code=400, detail="No orders in portfolio")

    # Apply stock splits
    orders = apply_stock_splits_to_orders(orders)

    # Get first order date
    first_order_date = min(o.date for o in orders)

    # Calculate current positions using same logic as dashboard
    from utils import aggregate_positions

    positions_map_temp = aggregate_positions(orders)

    # Rebuild positions_map with instrument_type
    positions_map = {}
    for symbol, data in positions_map_temp.items():
        # Find instrument type from orders
        instrument_type = "stock"
        for o in orders:
            if o.symbol.upper() == symbol:
                instrument_type = o.instrument_type or "stock"
                break

        positions_map[symbol] = {
            "symbol": symbol,
            "quantity": data["quantity"],
            "total_cost": data["total_cost"],
            "instrument_type": instrument_type
        }

    if not positions_map:
        raise HTTPException(status_code=400, detail="No current positions")

    # Total invested is the sum of total_cost from all current positions
    # IMPORTANT: Need to convert to reference currency!
    total_invested_dca = 0.0
    for symbol, pos in positions_map.items():
        # Get currency from first order
        symbol_orders = [o for o in orders if o.symbol.upper() == symbol]
        order_currency = symbol_orders[0].currency if symbol_orders else pf.reference_currency

        # Convert cost to reference currency
        cost_in_ref_currency = convert_to_reference_currency(
            pos["total_cost"],
            order_currency,
            pf.reference_currency,
            db
        )
        total_invested_dca += cost_in_ref_currency
        pos["total_cost"] = cost_in_ref_currency  # Update for later use

    # Get historical prices for all symbols from first_order_date to today
    from utils import get_stock_price_and_history_cached, get_etf_price_and_history

    # Calculate days from first order to today
    days_since_first_order = (datetime.now().date() - first_order_date).days

    symbols_history = {}
    lumpsum_cost = 0.0

    for symbol, pos_data in positions_map.items():
        try:
            instrument_type = pos_data["instrument_type"]

            if instrument_type == "etf":
                # For ETFs, we need to get the ISIN first
                symbol_orders = [o for o in orders if o.symbol.upper() == symbol]
                if not symbol_orders:
                    continue

                isin = symbol_orders[0].isin or ""
                if not isin:
                    continue

                price_data = get_etf_price_and_history(isin=isin, db=db)
            else:
                price_data = get_stock_price_and_history_cached(
                    symbol=symbol,
                    db=db,
                    days=days_since_first_order
                )

            if not price_data or "history" not in price_data or not price_data["history"]:
                continue

            history = price_data["history"]

            # For currency conversion, we'll use the latest rate for simplicity
            # (since historical FX rates might not align perfectly with asset prices)
            symbol_currency = price_data.get("currency", "USD")

            # Get FX rate if needed
            fx_rate = 1.0
            if symbol_currency != pf.reference_currency:
                try:
                    from utils import get_exchange_rate_history
                    fx_data = get_exchange_rate_history(symbol_currency, pf.reference_currency, db)
                    fx_rates = fx_data.get("rates", [])
                    if fx_rates:
                        fx_rate = fx_rates[-1].get("rate", 1.0)
                except Exception:
                    pass

            # Convert history to dates and apply FX
            converted_history = []
            for entry in history:
                price = entry.get("price", 0.0)
                converted_price = price * fx_rate

                # Parse date string to date object
                entry_date = parse_date_input(entry["date"])
                if entry_date:
                    converted_history.append({
                        "date": entry_date,
                        "price": converted_price
                    })

            if not converted_history:
                continue

            # Sort by date
            converted_history.sort(key=lambda x: x["date"])

            symbols_history[symbol] = {
                "history": converted_history,
                "quantity": pos_data["quantity"],
                "total_cost": pos_data["total_cost"]  # Already converted to ref currency
            }

        except Exception:
            continue

    if not symbols_history:
        raise HTTPException(status_code=400, detail="Could not fetch price history")

    # Calculate allocation percentages for Lump Sum strategy
    # Lump Sum will invest total_invested_dca at first_order_date in current allocation percentages
    allocation_percentages = {}
    for symbol, symbol_data in symbols_history.items():
        allocation_percentages[symbol] = symbol_data["total_cost"] / total_invested_dca if total_invested_dca > 0 else 0

    # Build timeline of values for both strategies
    # Collect all unique dates where ALL symbols have prices (to avoid partial data)
    all_dates = set()
    for symbol_data in symbols_history.values():
        for entry in symbol_data["history"]:
            all_dates.add(entry["date"])

    sorted_dates = sorted(all_dates)

    # Build price map for faster lookup
    price_maps = {}
    for symbol, symbol_data in symbols_history.items():
        price_maps[symbol] = {entry["date"]: entry["price"] for entry in symbol_data["history"]}

    # Calculate Lump Sum quantities based on first date prices
    # Invest total_invested_dca at first_order_date according to allocation percentages
    lumpsum_quantities = {}
    for symbol in symbols_history.keys():
        # Find price at first_order_date (or closest date after)
        first_date_price = None
        for date in sorted_dates:
            if date >= first_order_date and symbol in price_maps and date in price_maps[symbol]:
                first_date_price = price_maps[symbol][date]
                break

        if first_date_price and first_date_price > 0:
            # Calculate how much to invest in this symbol
            amount_to_invest = total_invested_dca * allocation_percentages[symbol]
            # Calculate quantity
            lumpsum_quantities[symbol] = amount_to_invest / first_date_price
        else:
            lumpsum_quantities[symbol] = 0

    # For each date, calculate portfolio value under both strategies
    dca_timeline = []
    lumpsum_timeline = []

    for current_date in sorted_dates:
        # Skip dates before first order
        if current_date < first_order_date:
            continue

        # Skip dates where not all symbols have prices
        all_have_prices = all(current_date in price_maps[symbol] for symbol in symbols_history.keys())
        if not all_have_prices:
            continue

        # DCA value: simulate buying current quantities gradually over time
        # Using the same cost calculation as aggregate_positions
        dca_value = 0.0
        dca_invested = 0.0

        for symbol, symbol_data in symbols_history.items():
            current_qty = symbol_data["quantity"]
            price_at_date = price_maps[symbol][current_date]

            # Get the cost basis for this symbol at this date from positions_map
            # This already accounts for sales correctly
            symbol_total_cost = positions_map[symbol]["total_cost"]

            # Calculate how many shares we had bought by this date
            qty_bought_by_date = 0.0
            for o in orders:
                if o.symbol.upper() == symbol and o.date <= current_date and o.order_type.lower() == "buy":
                    qty_bought_by_date += o.quantity

            # Use minimum between bought quantity and current quantity
            effective_qty = min(qty_bought_by_date, current_qty)

            if effective_qty > 0:
                dca_value += effective_qty * price_at_date
                # Proportional cost: if we've bought all current qty, use full cost
                # Otherwise use proportional cost
                if qty_bought_by_date >= current_qty:
                    dca_invested += symbol_total_cost
                else:
                    # Scale cost proportionally to how much we've bought
                    dca_invested += symbol_total_cost * (qty_bought_by_date / current_qty)

        # Lump sum value: quantities bought at first_order_date with total_invested_dca
        lumpsum_value = 0.0
        for symbol in symbols_history.keys():
            price_at_date = price_maps[symbol][current_date]
            lumpsum_value += lumpsum_quantities[symbol] * price_at_date

        dca_timeline.append({
            "date": current_date.strftime(DATE_FMT),
            "value": round(dca_value, 2),
            "invested": round(dca_invested, 2)
        })

        lumpsum_timeline.append({
            "date": current_date.strftime(DATE_FMT),
            "value": round(lumpsum_value, 2)
        })

    # Calculate final metrics using CURRENT prices (not last timeline date which might be old)
    # Get current prices and calculate actual current value
    final_dca_value = 0.0
    final_lumpsum_value = 0.0

    for symbol, symbol_data in symbols_history.items():
        try:
            instrument_type = positions_map[symbol]["instrument_type"]

            # Get current price
            if instrument_type == "etf":
                symbol_orders = [o for o in orders if o.symbol.upper() == symbol]
                isin = symbol_orders[0].isin if symbol_orders else ""
                price_data = get_etf_price_and_history(isin, db)
            else:
                price_data = get_stock_price_and_history_cached(symbol, db, days=7)

            current_price = price_data.get("last_price", 0.0)

            # Get currency and convert if needed
            symbol_orders = [o for o in orders if o.symbol.upper() == symbol]
            order_currency = symbol_orders[0].currency if symbol_orders else pf.reference_currency

            # Convert price to reference currency (using latest rate)
            current_price_ref = convert_to_reference_currency(
                current_price,
                order_currency,
                pf.reference_currency,
                db
            )

            # Calculate DCA value (current quantities)
            final_dca_value += symbol_data["quantity"] * current_price_ref

            # Calculate Lump Sum value (quantities bought at first date)
            final_lumpsum_value += lumpsum_quantities[symbol] * current_price_ref

        except Exception:
            # Fallback to last timeline value
            if dca_timeline:
                final_dca_value = dca_timeline[-1]["value"]
            if lumpsum_timeline:
                final_lumpsum_value = lumpsum_timeline[-1]["value"]
            break

    # Both strategies invest the same total amount
    dca_return_pct = ((final_dca_value - total_invested_dca) / total_invested_dca * 100) if total_invested_dca > 0 else 0
    lumpsum_return_pct = ((final_lumpsum_value - total_invested_dca) / total_invested_dca * 100) if total_invested_dca > 0 else 0

    return {
        "first_order_date": first_order_date.strftime(DATE_FMT),
        "dca": {
            "timeline": dca_timeline,
            "total_invested": round(total_invested_dca, 2),
            "final_value": round(final_dca_value, 2),
            "return_pct": round(dca_return_pct, 2),
            "gain_loss": round(final_dca_value - total_invested_dca, 2)
        },
        "lumpsum": {
            "timeline": lumpsum_timeline,
            "total_invested": round(total_invested_dca, 2),
            "final_value": round(final_lumpsum_value, 2),
            "return_pct": round(lumpsum_return_pct, 2),
            "gain_loss": round(final_lumpsum_value - total_invested_dca, 2)
        },
        "comparison": {
            "difference_value": round(final_dca_value - final_lumpsum_value, 2),
            "difference_pct": round(abs(dca_return_pct - lumpsum_return_pct), 2),
            "winner": "DCA" if final_dca_value > final_lumpsum_value else "Lump Sum"
        }
    }


@router.get("/compare-benchmark/{portfolio_id}")
def compare_portfolio_vs_benchmark(
    portfolio_id: int,
    benchmark: str = "SPY",
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(verify_token)
):
    """
    Compare portfolio performance against a benchmark using the same DCA strategy.

    Args:
        portfolio_id: Portfolio ID
        benchmark: Benchmark symbol (SPY, VWCE.DE, etc.) - defaults to SPY

    Returns:
        Comparison data showing portfolio vs benchmark with same DCA investment pattern
    """
    user_id = current_user.id

    # Get portfolio
    pf = db.query(PortfolioModel).filter_by(id=portfolio_id, user_id=user_id).first()
    if not pf:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Get all orders for this portfolio
    orders = db.query(OrderModel).filter_by(portfolio_id=portfolio_id).order_by(OrderModel.date).all()
    if not orders:
        raise HTTPException(status_code=400, detail="No orders found")

    # Get first order date
    first_order_date = min(o.date for o in orders)

    # First, get current positions to know which assets are still in portfolio
    from utils import aggregate_positions, convert_to_reference_currency

    positions_map = aggregate_positions(orders)

    # Get symbols that are currently in portfolio (quantity > 0)
    current_symbols = set(positions_map.keys())

    # Calculate total invested over time (in reference currency)
    # Only consider orders for assets that are STILL in the portfolio
    order_dates_amounts = []
    for order in orders:
        if order.order_type.lower() == "buy" and order.symbol.upper() in current_symbols:
            # Convert order amount to reference currency
            order_amount = order.quantity * order.price
            order_amount_ref = convert_to_reference_currency(
                order_amount,
                order.currency,
                pf.reference_currency,
                db
            )
            order_dates_amounts.append({
                "date": order.date,
                "amount": order_amount_ref,
                "symbol": order.symbol.upper()
            })

    if not order_dates_amounts:
        raise HTTPException(status_code=400, detail="No buy orders found for current positions")

    # Fetch benchmark data using cache
    from utils import get_stock_price_and_history_cached, get_etf_price_and_history
    from datetime import datetime
    import datetime as dt_module

    # Map common benchmark names to their symbols and types
    benchmark_info = {
        "SPY": {"symbol": "^GSPC", "type": "stock"},  # S&P 500 Index
        "SP500": {"symbol": "^GSPC", "type": "stock"},
        "VWCE": {"symbol": "VWCE.DE", "isin": "IE00BK5BQT80", "type": "etf"}  # FTSE All-World
    }

    bench_info = benchmark_info.get(benchmark.upper(), {"symbol": benchmark, "type": "stock"})
    benchmark_symbol = bench_info["symbol"]
    benchmark_type = bench_info["type"]

    # Calculate how many days of history we need
    first_order_date = min(o["date"] for o in order_dates_amounts)
    today = datetime.now().date()
    days_needed = (today - first_order_date).days + 30  # Add buffer for weekends/holidays

    try:
        # Use appropriate cached function based on benchmark type
        if benchmark_type == "etf" and "isin" in bench_info:
            # Use ETF function with ISIN
            benchmark_data = get_etf_price_and_history(bench_info["isin"], db)
        else:
            # Use stock function with symbol
            benchmark_data = get_stock_price_and_history_cached(benchmark_symbol, db, days=days_needed)

        if not benchmark_data or not benchmark_data.get("history"):
            raise HTTPException(status_code=500, detail=f"No data available for benchmark {benchmark_symbol}")

        benchmark_history = benchmark_data.get("history", [])
        last_price = benchmark_data.get("last_price", 0.0)

        if not last_price:
            raise HTTPException(status_code=500, detail=f"No price data available for benchmark {benchmark_symbol}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch benchmark data: {str(e)}")

    # Build price map for benchmark
    benchmark_price_map = {}
    for entry in benchmark_history:
        entry_date = parse_date_input(entry["date"])
        if entry_date:
            benchmark_price_map[entry_date] = entry["price"]

    # Simulate DCA on benchmark: invest same amounts on same dates
    benchmark_timeline = []
    benchmark_shares = 0.0
    benchmark_total_invested = 0.0

    # Get all unique dates from first order to today
    from datetime import datetime
    import datetime as dt_module

    all_dates = sorted(benchmark_price_map.keys())
    today = datetime.now().date()

    for current_date in all_dates:
        if current_date < first_order_date or current_date > today:
            continue

        # Check if there are any orders on this date
        orders_on_date = [o for o in order_dates_amounts if o["date"] == current_date]

        # Buy benchmark shares with the same amount
        for order_info in orders_on_date:
            amount = order_info["amount"]
            price_on_date = benchmark_price_map.get(current_date)

            if price_on_date and price_on_date > 0:
                shares_bought = amount / price_on_date
                benchmark_shares += shares_bought
                benchmark_total_invested += amount

        # Calculate value at this date
        price_at_date = benchmark_price_map.get(current_date, 0)
        benchmark_value = benchmark_shares * price_at_date

        if benchmark_shares > 0:  # Only add to timeline if we have shares
            benchmark_timeline.append({
                "date": current_date.strftime(DATE_FMT),
                "value": round(benchmark_value, 2),
                "invested": round(benchmark_total_invested, 2)
            })

    # Get portfolio performance with timeline (using existing logic)
    from utils import compute_portfolio_value

    # Aggregate positions to get current portfolio value (positions_map already calculated above)
    symbol_type_map = {}
    symbol_isin_map = {}
    orders_by_symbol = {}

    for o in orders:
        sym = o.symbol.upper()
        if sym not in symbol_type_map:
            symbol_type_map[sym] = o.instrument_type or "stock"
            symbol_isin_map[sym] = o.isin or ""
        if sym not in orders_by_symbol:
            orders_by_symbol[sym] = []
        orders_by_symbol[sym].append(o)

    positions, portfolio_current_value, portfolio_total_cost, _, _, position_histories = compute_portfolio_value(
        positions_map,
        orders_by_symbol,
        symbol_type_map,
        symbol_isin_map,
        db,
        include_history=True,
        reference_currency=pf.reference_currency
    )

    # Build portfolio timeline reflecting DCA: quantities increase over time with each purchase
    # Only consider assets still in portfolio (same logic as benchmark)
    portfolio_timeline = []

    if position_histories:
        # Get all dates from position histories, but only from first_order_date onwards
        all_dates = set()
        price_map_current = {}

        for symbol, history in position_histories.items():
            # Only include symbols that are still in the portfolio
            if symbol not in positions_map:
                continue

            price_map_current[symbol] = {}
            for point in history:
                d = parse_date_input(point.get("date"))
                # Only include dates from first order onwards
                if d and d >= first_order_date:
                    price_map_current[symbol][d] = float(point.get("price", 0))
                    all_dates.add(d)

        # Track cumulative quantities per symbol (building up over time)
        quantity_tracker = {sym: 0.0 for sym in positions_map.keys()}

        # Get orders for current symbols only, sorted by date
        current_orders = [o for o in orders if o.symbol.upper() in current_symbols]
        current_orders.sort(key=lambda x: x.date)
        order_index = 0

        # For each date, calculate portfolio value with quantities accumulated up to that date
        if all_dates:
            for current_date in sorted(all_dates):
                # Apply orders up to current_date
                while order_index < len(current_orders) and current_orders[order_index].date <= current_date:
                    order = current_orders[order_index]
                    symbol = order.symbol.upper()
                    if order.order_type.lower() == "buy":
                        quantity_tracker[symbol] += order.quantity
                    elif order.order_type.lower() == "sell":
                        quantity_tracker[symbol] -= order.quantity
                    order_index += 1

                # Calculate total value with accumulated quantities
                total_value = 0.0
                valid = True

                for symbol in positions_map.keys():
                    qty = quantity_tracker[symbol]
                    if qty > 0:  # Only if we have shares at this date
                        if symbol in price_map_current:
                            price = price_map_current[symbol].get(current_date)
                            if price is not None:
                                total_value += qty * price
                            else:
                                valid = False
                                break
                        else:
                            # Symbol has no price history
                            valid = False
                            break

                if valid and total_value > 0:  # Only add if value > 0 (we own something)
                    portfolio_timeline.append({
                        "date": current_date.strftime(DATE_FMT),
                        "value": round(total_value, 2)
                    })

    # Calculate final benchmark value
    final_benchmark_value = benchmark_shares * last_price

    # Use portfolio_total_cost for both to ensure fair comparison
    # This ensures both invested the same amount
    benchmark_return_pct = ((final_benchmark_value - portfolio_total_cost) / portfolio_total_cost * 100) if portfolio_total_cost > 0 else 0
    portfolio_return_pct = ((portfolio_current_value - portfolio_total_cost) / portfolio_total_cost * 100) if portfolio_total_cost > 0 else 0

    return {
        "benchmark_symbol": benchmark_symbol,
        "portfolio": {
            "timeline": portfolio_timeline,
            "total_invested": round(portfolio_total_cost, 2),
            "final_value": round(portfolio_current_value, 2),
            "return_pct": round(portfolio_return_pct, 2),
            "gain_loss": round(portfolio_current_value - portfolio_total_cost, 2)
        },
        "benchmark": {
            "timeline": benchmark_timeline,
            "total_invested": round(portfolio_total_cost, 2),  # Use same total invested
            "final_value": round(final_benchmark_value, 2),
            "return_pct": round(benchmark_return_pct, 2),
            "gain_loss": round(final_benchmark_value - portfolio_total_cost, 2)
        },
        "comparison": {
            "difference_value": round(portfolio_current_value - final_benchmark_value, 2),
            "difference_pct": round(abs(portfolio_return_pct - benchmark_return_pct), 2),
            "winner": "Portfolio" if portfolio_current_value > final_benchmark_value else "Benchmark"
        }
    }
