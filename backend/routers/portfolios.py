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
)

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
def analyze_portfolio(portfolio_id: int, user: UserModel = Depends(verify_token), db: Session = Depends(get_db)):
    """
    Advanced portfolio analysis including:
    - Correlation matrix
    - Monte Carlo simulation
    - Risk metrics (VaR, CVaR, Sharpe, Sortino, etc.)
    - Drawdown analysis
    """
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
        # Find common dates across all symbols
        all_dates_sets = []
        for symbol, history in position_histories.items():
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

        # Build price DataFrame
        price_data = {}
        for symbol, history in position_histories.items():
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
        num_days = 252
        simulations = []
        for _ in range(num_simulations):
            sim_returns = np.random.choice(portfolio_returns, size=num_days, replace=True)
            sim_cumulative = (1 + sim_returns).cumprod()
            simulations.append(sim_cumulative.tolist())

        # Calculate percentiles for each day across all simulations
        simulations_array = np.array(simulations)  # shape: (1000, 252)

        # Compute percentiles across simulations for each day
        p5_array = np.percentile(simulations_array, 5, axis=0)
        p25_array = np.percentile(simulations_array, 25, axis=0)
        p50_array = np.percentile(simulations_array, 50, axis=0)
        p75_array = np.percentile(simulations_array, 75, axis=0)
        p95_array = np.percentile(simulations_array, 95, axis=0)

        # Generate date labels for the simulation period (1 year forward)
        from datetime import timedelta
        today = dt_date.today()
        simulation_dates = [(today + timedelta(days=i)).strftime(DATE_FMT) for i in range(num_days)]

        montecarlo = {
            "simulations": simulations[:10],  # Return only first 10 for bandwidth
            "dates": simulation_dates,
            "current_value": float(total_value),
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

        # 6. Performance Attribution
        assets_attribution = []
        for symbol in data.columns:
            if symbol in position_symbols:
                idx = position_symbols.index(symbol)
                position = positions[idx]

                # Calculate contribution to portfolio return
                asset_returns = returns[symbol]
                weight = weights[idx]
                contribution = (asset_returns * weight).sum() * 100  # Percentage contribution
                total_return = asset_returns.sum() * 100  # Total return percentage

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
