from fastapi import APIRouter, Depends, HTTPException
from datetime import date as dt_date, datetime, timezone
from collections import defaultdict
from types import SimpleNamespace
from sqlalchemy.orm import Session

import numpy as np
import pandas as pd
import yfinance as yf

from schemas import Portfolio
from utils import (
    get_db,
    get_supabase,
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

router = APIRouter(prefix="/api/portfolios", tags=["portfolios"])


def _order_from_row(row: dict) -> SimpleNamespace:
    """Converte una row Supabase in un oggetto con accesso ad attributi (compatibile con le utility functions)."""
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
    )


def _apply_stock_splits_to_orders(orders):
    symbol_splits = {}
    stock_symbols = set()
    first_order_dates = {}

    for o in orders:
        symbol = o.symbol.upper()
        if (o.instrument_type or "stock").lower() == "stock":
            stock_symbols.add(symbol)
            if symbol not in first_order_dates or o.date < first_order_dates[symbol]:
                first_order_dates[symbol] = o.date

    for symbol in stock_symbols:
        splits = get_stock_splits(symbol, first_order_dates[symbol])
        if splits:
            symbol_splits[symbol] = splits

    if symbol_splits:
        print(f"[SPLITS] Found splits for {len(symbol_splits)} symbols, applying adjustments...")
        return apply_splits_to_orders(list(orders), symbol_splits)

    return list(orders)


def _get_portfolio(sb, portfolio_id: int, user_id: str) -> dict:
    result = sb.table("portfolios").select("*").eq("id", portfolio_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return result.data[0]


def _get_orders(sb, portfolio_id: int) -> list:
    result = sb.table("orders").select("*").eq("portfolio_id", portfolio_id).execute()
    return [_order_from_row(r) for r in result.data]


@router.post("")
def create_portfolio(portfolio: Portfolio, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    result = sb.table("portfolios").insert({
        "user_id": user_id,
        "name": portfolio.name,
        "description": portfolio.description or "",
        "initial_capital": portfolio.initial_capital or 0.0,
        "reference_currency": portfolio.reference_currency or "EUR",
        "risk_free_source": portfolio.risk_free_source or "auto",
        "market_benchmark": portfolio.market_benchmark or "auto",
    }).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create portfolio")
    p = result.data[0]
    return {
        "id": p["id"],
        "name": p["name"],
        "description": p["description"],
        "initial_capital": p["initial_capital"],
        "reference_currency": p["reference_currency"],
        "risk_free_source": p["risk_free_source"],
        "market_benchmark": p["market_benchmark"],
        "created_at": p.get("created_at"),
    }


@router.get("/count")
def get_portfolios_count(user_id: str = Depends(verify_token)):
    sb = get_supabase()
    result = sb.table("portfolios").select("id").eq("user_id", user_id).execute()
    return {"count": len(result.data)}


@router.get("")
def get_portfolios(user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    sb = get_supabase()
    portfolios = sb.table("portfolios").select("*").eq("user_id", user_id).execute().data
    response = []
    for p in portfolios:
        orders = _get_orders(sb, p["id"])
        orders = _apply_stock_splits_to_orders(orders)
        positions_map = aggregate_positions(orders)
        orders_by_symbol = {}
        symbol_type_map = {}
        symbol_isin_map = {}
        for o in orders:
            orders_by_symbol.setdefault(o.symbol, []).append(o)
            symbol_type_map[o.symbol] = o.instrument_type
            symbol_isin_map[o.symbol] = o.isin.upper()
        reference_currency = p.get("reference_currency") or "EUR"
        _, total_value, total_cost, total_gain_loss, total_gain_loss_pct, _ = compute_portfolio_value(
            positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db, reference_currency=reference_currency
        )
        asset_composition = {"etf": 0, "stock": 0}
        for symbol, position in positions_map.items():
            if position["quantity"] > 0:
                asset_type = symbol_type_map.get(symbol, "stock").lower()
                asset_composition[asset_type if asset_type in asset_composition else "stock"] += position["quantity"]
        response.append({
            "id": p["id"],
            "name": p["name"],
            "description": p.get("description", ""),
            "initial_capital": p.get("initial_capital", 0.0),
            "reference_currency": reference_currency,
            "risk_free_source": p.get("risk_free_source", "auto"),
            "market_benchmark": p.get("market_benchmark", "auto"),
            "created_at": p.get("created_at"),
            "total_value": total_value,
            "total_cost": total_cost,
            "total_gain_loss": total_gain_loss,
            "total_gain_loss_pct": total_gain_loss_pct,
            "asset_composition": asset_composition,
        })
    return {"portfolios": response}


@router.get("/{portfolio_id}")
def get_portfolio(portfolio_id: int, user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    sb = get_supabase()
    p = _get_portfolio(sb, portfolio_id, user_id)
    orders = _get_orders(sb, portfolio_id)
    orders = _apply_stock_splits_to_orders(orders)

    orders_by_symbol = {}
    symbol_type_map = {}
    symbol_isin_map = {}
    for o in orders:
        orders_by_symbol.setdefault(o.symbol, []).append(o)
        symbol_type_map[o.symbol] = o.instrument_type
        symbol_isin_map[o.symbol] = o.isin.upper()
    positions_map = aggregate_positions(orders)
    reference_currency = p.get("reference_currency") or "EUR"
    positions, total_value, total_cost, total_gain_loss, total_gain_loss_pct, position_histories = compute_portfolio_value(
        positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db, include_history=True, reference_currency=reference_currency
    )

    portfolio_history, price_map, common_dates = aggregate_portfolio_history(position_histories, orders_by_symbol)
    performance_history = compute_portfolio_performance(price_map, orders_by_symbol, common_dates)

    portfolio_xirr = 0.0
    try:
        daily_cashflows = defaultdict(float)
        for o in orders:
            order_currency = o.currency or reference_currency
            rate = 1.0 if order_currency == reference_currency else convert_to_reference_currency(1.0, order_currency, reference_currency, db)
            if o.order_type == "buy":
                cf_amount = -(o.quantity * o.price + o.commission) * rate
            else:
                cf_amount = (o.quantity * o.price - o.commission) * rate
            daily_cashflows[o.date] += cf_amount

        cashflows = [(dt, amount) for dt, amount in daily_cashflows.items()]
        if total_value > 0:
            cashflows.append((dt_date.today(), total_value))
        if cashflows and len(cashflows) > 1:
            cashflows.sort(key=lambda x: x[0])
            total_invested = sum(cf for _, cf in cashflows if cf < 0)
            total_withdrawn = sum(cf for _, cf in cashflows[:-1] if cf > 0)
            portfolio_xirr_raw = calc_xirr(cashflows)
            portfolio_xirr = portfolio_xirr_raw * 100
    except Exception as e:
        print(f"[XIRR] Error: {e}")

    return {
        "portfolio": {
            "id": p["id"],
            "name": p["name"],
            "description": p.get("description", ""),
            "initial_capital": p.get("initial_capital", 0.0),
            "reference_currency": reference_currency,
            "risk_free_source": p.get("risk_free_source", "auto"),
            "market_benchmark": p.get("market_benchmark", "auto"),
            "created_at": p.get("created_at"),
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
def update_portfolio(portfolio_id: int, portfolio_update: Portfolio, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    _get_portfolio(sb, portfolio_id, user_id)  # 404 check
    updates = {}
    if portfolio_update.name:
        updates["name"] = portfolio_update.name
    if portfolio_update.description is not None:
        updates["description"] = portfolio_update.description
    if portfolio_update.reference_currency:
        updates["reference_currency"] = portfolio_update.reference_currency
    if portfolio_update.risk_free_source is not None:
        updates["risk_free_source"] = portfolio_update.risk_free_source
    if portfolio_update.market_benchmark is not None:
        updates["market_benchmark"] = portfolio_update.market_benchmark

    result = sb.table("portfolios").update(updates).eq("id", portfolio_id).eq("user_id", user_id).execute()
    p = result.data[0]
    return {
        "id": p["id"],
        "name": p["name"],
        "description": p.get("description", ""),
        "reference_currency": p.get("reference_currency", "EUR"),
        "risk_free_source": p.get("risk_free_source", "auto"),
        "market_benchmark": p.get("market_benchmark", "auto"),
        "created_at": p.get("created_at"),
    }


@router.delete("/{portfolio_id}")
def delete_portfolio(portfolio_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    _get_portfolio(sb, portfolio_id, user_id)  # 404 check
    sb.table("orders").delete().eq("portfolio_id", portfolio_id).execute()
    sb.table("portfolios").delete().eq("id", portfolio_id).eq("user_id", user_id).execute()
    return {"message": "Portfolio deleted"}


@router.get("/history/{portfolio_id}/{symbol}")
def get_position_history(
    portfolio_id: int, symbol: str, period: str = "1y",
    user_id: str = Depends(verify_token), db: Session = Depends(get_db)
):
    sb = get_supabase()
    _get_portfolio(sb, portfolio_id, user_id)  # 404 check
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        history = [
            {"date": idx.strftime(DATE_FMT), "close": round(row["Close"], 2), "volume": int(row["Volume"])}
            for idx, row in hist.iterrows()
        ]
        return {"symbol": symbol, "history": history}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/{portfolio_id}")
def analyze_portfolio(
    portfolio_id: int,
    monte_carlo_years: int = 1,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    if monte_carlo_years < 1 or monte_carlo_years > 30:
        raise HTTPException(status_code=400, detail="monte_carlo_years must be between 1 and 30")

    sb = get_supabase()
    p = _get_portfolio(sb, portfolio_id, user_id)
    orders = _get_orders(sb, portfolio_id)

    if not orders:
        return {"message": "No orders in portfolio", "correlation": None, "montecarlo": None, "risk_metrics": None, "drawdown": None}

    orders = _apply_stock_splits_to_orders(orders)

    positions_map = aggregate_positions(orders)
    orders_by_symbol = {}
    symbol_type_map = {}
    symbol_isin_map = {}
    for o in orders:
        orders_by_symbol.setdefault(o.symbol, []).append(o)
        symbol_type_map[o.symbol] = o.instrument_type
        symbol_isin_map[o.symbol] = o.isin.upper()

    reference_currency = p.get("reference_currency") or "EUR"
    positions, total_value, total_cost, _, _, position_histories = compute_portfolio_value(
        positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db,
        include_history=True, reference_currency=reference_currency
    )

    if len(positions) < 2:
        return {"message": "Need at least 2 assets for analysis", "correlation": None, "montecarlo": None, "risk_metrics": None, "drawdown": None}

    try:
        if not position_histories:
            return {"message": "Insufficient historical data", "correlation": None, "montecarlo": None, "risk_metrics": None, "drawdown": None}

        current_symbols = {pos["symbol"] for pos in positions}
        filtered_position_histories = {sym: hist for sym, hist in position_histories.items() if sym in current_symbols}

        all_dates_sets = []
        for symbol, history in filtered_position_histories.items():
            dates = set()
            for point in history:
                d = parse_date_input(point.get("date"))
                if d:
                    dates.add(d)
            all_dates_sets.append(dates)

        if not all_dates_sets:
            return {"message": "No historical data available", "correlation": None, "montecarlo": None, "risk_metrics": None, "drawdown": None}

        common_dates = set.intersection(*all_dates_sets) if len(all_dates_sets) > 1 else all_dates_sets[0]
        earliest_order_date = min(parse_date_input(o.date) for o in orders)
        common_dates = {d for d in common_dates if d >= earliest_order_date}

        if len(common_dates) < 30:
            return {"message": "Insufficient overlapping historical data (need at least 30 days)", "correlation": None, "montecarlo": None, "risk_metrics": None, "drawdown": None}

        sorted_dates = sorted(list(common_dates))
        price_data = {}
        for symbol, history in filtered_position_histories.items():
            prices_map = {}
            for point in history:
                d = parse_date_input(point.get("date"))
                if d and d in common_dates:
                    try:
                        prices_map[d] = float(point.get("price", 0))
                    except Exception:
                        pass
            price_data[symbol] = prices_map

        data_dict = {symbol: [price_data[symbol].get(d, np.nan) for d in sorted_dates] for symbol in price_data}
        data = pd.DataFrame(data_dict, index=pd.to_datetime(sorted_dates)).dropna()

        if len(data) < 30:
            return {"message": "Insufficient clean historical data after removing gaps", "correlation": None, "montecarlo": None, "risk_metrics": None, "drawdown": None}

        returns = data.pct_change(fill_method=None).dropna()
        corr_matrix = returns.corr().values.tolist()

        weights = np.array([p_item["market_value"] / total_value if total_value > 0 else 0 for p_item in positions])
        position_symbols = [p_item["symbol"] for p_item in positions]
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

        portfolio_returns = (returns * weights).sum(axis=1)

        sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0
        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino = (portfolio_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        var_95 = np.percentile(portfolio_returns, 5) * 100
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
        annualized_return = portfolio_returns.mean() * 252 * 100

        risk_metrics = {
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "var_95": round(float(var_95), 4),
            "cvar_95": round(float(cvar_95), 4),
            "volatility": round(float(portfolio_returns.std() * np.sqrt(252)), 4),
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "beta": None,
            "beta_vs": None
        }

        # Monte Carlo
        num_simulations = 1000
        num_days = 252 * monte_carlo_years
        simulations = []
        for _ in range(num_simulations):
            sim_returns = np.random.choice(portfolio_returns, size=num_days, replace=True)
            simulations.append((1 + sim_returns).cumprod().tolist())
        simulations_array = np.array(simulations)
        from datetime import timedelta
        today = dt_date.today()
        simulation_dates = [(today + timedelta(days=i)).strftime(DATE_FMT) for i in range(num_days)]
        montecarlo = {
            "simulations": simulations[:10],
            "dates": simulation_dates,
            "current_value": float(total_value),
            "years": monte_carlo_years,
            "percentiles": {
                "p5": [round(float(v * total_value), 2) for v in np.percentile(simulations_array, 5, axis=0)],
                "p25": [round(float(v * total_value), 2) for v in np.percentile(simulations_array, 25, axis=0)],
                "p50": [round(float(v * total_value), 2) for v in np.percentile(simulations_array, 50, axis=0)],
                "p75": [round(float(v * total_value), 2) for v in np.percentile(simulations_array, 75, axis=0)],
                "p95": [round(float(v * total_value), 2) for v in np.percentile(simulations_array, 95, axis=0)],
            }
        }

        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min())
        risk_metrics["max_drawdown"] = round(max_dd, 4)
        if max_dd != 0:
            risk_metrics["calmar_ratio"] = round(float(annualized_return / abs(max_dd * 100)), 4)
        drawdown_data = [{"date": idx.strftime(DATE_FMT), "drawdown": round(float(val), 4)} for idx, val in drawdown.items()]

        # Beta
        try:
            benchmark_symbol = p.get("market_benchmark")
            if not benchmark_symbol or str(benchmark_symbol).upper() == "AUTO":
                benchmark_symbol = "SPY" if reference_currency == "USD" else "VWCE.DE"
            risk_metrics["beta_vs"] = benchmark_symbol
            import utils.pricing as pricing_utils
            benchmark_data = pricing_utils.get_stock_price_and_history_cached(benchmark_symbol, db, days=len(data))
            if benchmark_data and benchmark_data.get("history"):
                benchmark_prices = {}
                for point in benchmark_data["history"]:
                    d = parse_date_input(point.get("date"))
                    if d:
                        benchmark_prices[d] = float(point.get("price", 0))
                benchmark_series = [benchmark_prices.get(d, np.nan) for d in sorted_dates]
                benchmark_df = pd.Series(benchmark_series, index=pd.to_datetime(sorted_dates)).dropna()
                benchmark_returns = benchmark_df.pct_change(fill_method=None).dropna()
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 30:
                    pf_aligned = portfolio_returns.loc[common_index]
                    bm_aligned = benchmark_returns.loc[common_index]
                    covariance = np.cov(pf_aligned, bm_aligned)[0, 1]
                    variance = np.var(bm_aligned)
                    if variance > 0:
                        risk_metrics["beta"] = round(float(covariance / variance), 4)
        except Exception as e:
            print(f"Beta calculation failed: {e}")

        # Asset contribution
        assets_attribution = []
        for symbol in data.columns:
            if symbol in position_symbols:
                idx = position_symbols.index(symbol)
                position = positions[idx]
                asset_returns = returns[symbol]
                weight = weights[idx]
                contribution = (asset_returns * weight).sum() * 100
                total_return = asset_returns.sum() * 100
                assets_attribution.append({
                    "symbol": symbol,
                    "weight": round(float(weight * 100), 2),
                    "current_value": round(float(position["market_value"]), 2),
                    "annualized_return": round(float(asset_returns.mean() * 252 * 100), 2),
                    "total_return_pct": round(float(total_return), 2),
                    "contribution_to_portfolio": round(float(contribution), 2),
                    "volatility": round(float(asset_returns.std() * np.sqrt(252)), 4),
                    "gain_loss": round(float(position.get("gain_loss", 0)), 2)
                })
        assets_attribution.sort(key=lambda x: x["contribution_to_portfolio"], reverse=True)
        worst_sorted = sorted(assets_attribution, key=lambda x: x["contribution_to_portfolio"])
        performance_attribution = {
            "portfolio_total_return": round(float((portfolio_returns.sum() * 100) if len(portfolio_returns) > 0 else 0), 2),
            "top_contributors": assets_attribution[:3],
            "worst_contributors": worst_sorted[:3],
            "assets": assets_attribution
        }

        historical_prices = {"dates": [d.strftime(DATE_FMT) for d in sorted_dates], "assets": {}}
        for symbol in data.columns:
            prices = data[symbol].values
            if prices[0] > 0:
                historical_prices["assets"][symbol] = [round(float(x), 2) for x in (prices / prices[0] * 100).tolist()]

        return {
            "message": "Analysis complete",
            "correlation": {"matrix": corr_matrix, "symbols": list(data.columns)},
            "montecarlo": montecarlo,
            "risk_metrics": risk_metrics,
            "drawdown": drawdown_data,
            "performance_attribution": performance_attribution,
            "historical_prices": historical_prices
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/compare/{portfolio_id}")
def compare_dca_vs_lumpsum(portfolio_id: int, user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    sb = get_supabase()
    pf = _get_portfolio(sb, portfolio_id, user_id)
    orders = _get_orders(sb, portfolio_id)
    if not orders:
        raise HTTPException(status_code=400, detail="No orders in portfolio")

    orders = _apply_stock_splits_to_orders(orders)
    first_order_date = min(o.date for o in orders)

    positions_map_temp = aggregate_positions(orders)
    positions_map = {}
    for symbol, data_item in positions_map_temp.items():
        instrument_type = "stock"
        for o in orders:
            if o.symbol == symbol:
                instrument_type = o.instrument_type or "stock"
                break
        positions_map[symbol] = {"symbol": symbol, "quantity": data_item["quantity"], "total_cost": data_item["total_cost"], "instrument_type": instrument_type}

    if not positions_map:
        raise HTTPException(status_code=400, detail="No current positions")

    reference_currency = pf.get("reference_currency") or "EUR"
    total_invested_dca = 0.0
    for symbol, pos in positions_map.items():
        symbol_orders = [o for o in orders if o.symbol == symbol]
        order_currency = symbol_orders[0].currency if symbol_orders else reference_currency
        cost_in_ref = convert_to_reference_currency(pos["total_cost"], order_currency, reference_currency, db)
        total_invested_dca += cost_in_ref
        pos["total_cost"] = cost_in_ref

    from utils import get_stock_price_and_history_cached, get_etf_price_and_history, get_exchange_rate_history
    days_since_first_order = (datetime.now().date() - first_order_date).days

    symbols_history = {}
    for symbol, pos_data in positions_map.items():
        try:
            if pos_data["instrument_type"] == "etf":
                symbol_orders = [o for o in orders if o.symbol == symbol]
                isin = symbol_orders[0].isin if symbol_orders else ""
                if not isin:
                    continue
                price_data = get_etf_price_and_history(isin=isin, db=db)
            else:
                price_data = get_stock_price_and_history_cached(symbol=symbol, db=db, days=days_since_first_order)

            if not price_data or not price_data.get("history"):
                continue

            symbol_currency = price_data.get("currency", "USD")
            fx_rate = 1.0
            if symbol_currency != reference_currency:
                try:
                    fx_data = get_exchange_rate_history(symbol_currency, reference_currency, db)
                    fx_rates = fx_data.get("rates", [])
                    if fx_rates:
                        fx_rate = fx_rates[-1].get("rate", 1.0)
                except Exception:
                    pass

            converted_history = []
            for entry in price_data["history"]:
                entry_date = parse_date_input(entry["date"])
                if entry_date:
                    converted_history.append({"date": entry_date, "price": entry.get("price", 0.0) * fx_rate})
            if not converted_history:
                continue
            converted_history.sort(key=lambda x: x["date"])
            symbols_history[symbol] = {"history": converted_history, "quantity": pos_data["quantity"], "total_cost": pos_data["total_cost"]}
        except Exception:
            continue

    if not symbols_history:
        raise HTTPException(status_code=400, detail="Could not fetch price history")

    all_dates = set()
    for symbol_data in symbols_history.values():
        for entry in symbol_data["history"]:
            all_dates.add(entry["date"])
    sorted_dates = sorted(all_dates)
    price_maps = {symbol: {entry["date"]: entry["price"] for entry in symbol_data["history"]} for symbol, symbol_data in symbols_history.items()}

    allocation_percentages = {symbol: symbol_data["total_cost"] / total_invested_dca if total_invested_dca > 0 else 0 for symbol, symbol_data in symbols_history.items()}
    lumpsum_quantities = {}
    for symbol in symbols_history.keys():
        first_date_price = next((price_maps[symbol][d] for d in sorted_dates if d >= first_order_date and symbol in price_maps and d in price_maps[symbol]), None)
        if first_date_price and first_date_price > 0:
            lumpsum_quantities[symbol] = total_invested_dca * allocation_percentages[symbol] / first_date_price
        else:
            lumpsum_quantities[symbol] = 0

    dca_timeline = []
    lumpsum_timeline = []
    for current_date in sorted_dates:
        if current_date < first_order_date:
            continue
        if not all(current_date in price_maps[sym] for sym in symbols_history.keys()):
            continue
        dca_value = dca_invested = 0.0
        for symbol, symbol_data in symbols_history.items():
            current_qty = symbol_data["quantity"]
            price_at_date = price_maps[symbol][current_date]
            symbol_total_cost = positions_map[symbol]["total_cost"]
            qty_bought_by_date = sum(o.quantity for o in orders if o.symbol == symbol and o.date <= current_date and o.order_type == "buy")
            effective_qty = min(qty_bought_by_date, current_qty)
            if effective_qty > 0:
                dca_value += effective_qty * price_at_date
                if qty_bought_by_date >= current_qty:
                    dca_invested += symbol_total_cost
                else:
                    dca_invested += symbol_total_cost * (qty_bought_by_date / current_qty)
        lumpsum_value = sum(lumpsum_quantities[sym] * price_maps[sym][current_date] for sym in symbols_history.keys())
        dca_timeline.append({"date": current_date.strftime(DATE_FMT), "value": round(dca_value, 2), "invested": round(dca_invested, 2)})
        lumpsum_timeline.append({"date": current_date.strftime(DATE_FMT), "value": round(lumpsum_value, 2)})

    final_dca_value = final_lumpsum_value = 0.0
    for symbol, symbol_data in symbols_history.items():
        try:
            instrument_type = positions_map[symbol]["instrument_type"]
            if instrument_type == "etf":
                symbol_orders = [o for o in orders if o.symbol == symbol]
                isin = symbol_orders[0].isin if symbol_orders else ""
                price_data = get_etf_price_and_history(isin, db)
            else:
                price_data = get_stock_price_and_history_cached(symbol, db, days=7)
            current_price = price_data.get("last_price", 0.0)
            symbol_orders = [o for o in orders if o.symbol == symbol]
            order_currency = symbol_orders[0].currency if symbol_orders else reference_currency
            current_price_ref = convert_to_reference_currency(current_price, order_currency, reference_currency, db)
            final_dca_value += symbol_data["quantity"] * current_price_ref
            final_lumpsum_value += lumpsum_quantities[symbol] * current_price_ref
        except Exception:
            if dca_timeline:
                final_dca_value = dca_timeline[-1]["value"]
            if lumpsum_timeline:
                final_lumpsum_value = lumpsum_timeline[-1]["value"]
            break

    dca_return_pct = ((final_dca_value - total_invested_dca) / total_invested_dca * 100) if total_invested_dca > 0 else 0
    lumpsum_return_pct = ((final_lumpsum_value - total_invested_dca) / total_invested_dca * 100) if total_invested_dca > 0 else 0

    return {
        "first_order_date": first_order_date.strftime(DATE_FMT),
        "dca": {"timeline": dca_timeline, "total_invested": round(total_invested_dca, 2), "final_value": round(final_dca_value, 2), "return_pct": round(dca_return_pct, 2), "gain_loss": round(final_dca_value - total_invested_dca, 2)},
        "lumpsum": {"timeline": lumpsum_timeline, "total_invested": round(total_invested_dca, 2), "final_value": round(final_lumpsum_value, 2), "return_pct": round(lumpsum_return_pct, 2), "gain_loss": round(final_lumpsum_value - total_invested_dca, 2)},
        "comparison": {"difference_value": round(final_dca_value - final_lumpsum_value, 2), "difference_pct": round(abs(dca_return_pct - lumpsum_return_pct), 2), "winner": "DCA" if final_dca_value > final_lumpsum_value else "Lump Sum"}
    }


@router.get("/compare-benchmark/{portfolio_id}")
def compare_portfolio_vs_benchmark(
    portfolio_id: int,
    benchmark: str = "SPY",
    db: Session = Depends(get_db),
    user_id: str = Depends(verify_token)
):
    sb = get_supabase()
    pf = _get_portfolio(sb, portfolio_id, user_id)
    orders = _get_orders(sb, portfolio_id)
    if not orders:
        raise HTTPException(status_code=400, detail="No orders found")

    orders = _apply_stock_splits_to_orders(orders)
    first_order_date = min(o.date for o in orders)
    reference_currency = pf.get("reference_currency") or "EUR"

    positions_map = aggregate_positions(orders)
    current_symbols = set(positions_map.keys())

    order_dates_amounts = []
    for order in orders:
        if order.order_type == "buy" and order.symbol in current_symbols:
            order_amount_ref = convert_to_reference_currency(order.quantity * order.price, order.currency, reference_currency, db)
            order_dates_amounts.append({"date": order.date, "amount": order_amount_ref, "symbol": order.symbol})

    if not order_dates_amounts:
        raise HTTPException(status_code=400, detail="No buy orders found for current positions")

    from utils import get_stock_price_and_history_cached, get_etf_price_and_history
    benchmark_info = {
        "SPY": {"symbol": "^GSPC", "type": "stock"},
        "SP500": {"symbol": "^GSPC", "type": "stock"},
        "VWCE": {"symbol": "VWCE.DE", "isin": "IE00BK5BQT80", "type": "etf"}
    }
    bench_info = benchmark_info.get(benchmark.upper(), {"symbol": benchmark, "type": "stock"})
    benchmark_symbol = bench_info["symbol"]
    days_needed = (datetime.now().date() - first_order_date).days + 30

    try:
        if bench_info["type"] == "etf" and "isin" in bench_info:
            benchmark_data = get_etf_price_and_history(bench_info["isin"], db)
        else:
            benchmark_data = get_stock_price_and_history_cached(benchmark_symbol, db, days=days_needed)
        if not benchmark_data or not benchmark_data.get("history"):
            raise HTTPException(status_code=500, detail=f"No data for benchmark {benchmark_symbol}")
        benchmark_history = benchmark_data["history"]
        last_price = benchmark_data.get("last_price", 0.0)
        if not last_price:
            raise HTTPException(status_code=500, detail=f"No price data for benchmark {benchmark_symbol}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch benchmark: {str(e)}")

    benchmark_price_map = {}
    for entry in benchmark_history:
        entry_date = parse_date_input(entry["date"])
        if entry_date:
            benchmark_price_map[entry_date] = entry["price"]

    benchmark_shares = benchmark_total_invested = 0.0
    benchmark_timeline = []
    today = datetime.now().date()
    for current_date in sorted(benchmark_price_map.keys()):
        if current_date < first_order_date or current_date > today:
            continue
        for order_info in [o for o in order_dates_amounts if o["date"] == current_date]:
            price_on_date = benchmark_price_map.get(current_date)
            if price_on_date and price_on_date > 0:
                benchmark_shares += order_info["amount"] / price_on_date
                benchmark_total_invested += order_info["amount"]
        price_at_date = benchmark_price_map.get(current_date, 0)
        if benchmark_shares > 0:
            benchmark_timeline.append({"date": current_date.strftime(DATE_FMT), "value": round(benchmark_shares * price_at_date, 2), "invested": round(benchmark_total_invested, 2)})

    symbol_type_map = {}
    symbol_isin_map = {}
    orders_by_symbol = {}
    for o in orders:
        if o.symbol not in symbol_type_map:
            symbol_type_map[o.symbol] = o.instrument_type
            symbol_isin_map[o.symbol] = o.isin
        orders_by_symbol.setdefault(o.symbol, []).append(o)

    positions, portfolio_current_value, portfolio_total_cost, _, _, position_histories = compute_portfolio_value(
        positions_map, orders_by_symbol, symbol_type_map, symbol_isin_map, db, include_history=True, reference_currency=reference_currency
    )

    portfolio_timeline = []
    if position_histories:
        all_dates_set = set()
        price_map_current = {}
        for symbol, history in position_histories.items():
            if symbol not in positions_map:
                continue
            price_map_current[symbol] = {}
            for point in history:
                d = parse_date_input(point.get("date"))
                if d and d >= first_order_date:
                    price_map_current[symbol][d] = float(point.get("price", 0))
                    all_dates_set.add(d)

        quantity_tracker = {sym: 0.0 for sym in positions_map.keys()}
        current_orders = sorted([o for o in orders if o.symbol in current_symbols], key=lambda x: x.date)
        order_index = 0
        for current_date in sorted(all_dates_set):
            while order_index < len(current_orders) and current_orders[order_index].date <= current_date:
                order = current_orders[order_index]
                if order.order_type == "buy":
                    quantity_tracker[order.symbol] += order.quantity
                elif order.order_type == "sell":
                    quantity_tracker[order.symbol] -= order.quantity
                order_index += 1
            total_val = 0.0
            valid = True
            for symbol in positions_map.keys():
                qty = quantity_tracker[symbol]
                if qty > 0:
                    price = price_map_current.get(symbol, {}).get(current_date)
                    if price is not None:
                        total_val += qty * price
                    else:
                        valid = False
                        break
            if valid and total_val > 0:
                portfolio_timeline.append({"date": current_date.strftime(DATE_FMT), "value": round(total_val, 2)})

    final_benchmark_value = benchmark_shares * last_price
    benchmark_return_pct = ((final_benchmark_value - portfolio_total_cost) / portfolio_total_cost * 100) if portfolio_total_cost > 0 else 0
    portfolio_return_pct = ((portfolio_current_value - portfolio_total_cost) / portfolio_total_cost * 100) if portfolio_total_cost > 0 else 0

    return {
        "benchmark_symbol": benchmark_symbol,
        "portfolio": {"timeline": portfolio_timeline, "total_invested": round(portfolio_total_cost, 2), "final_value": round(portfolio_current_value, 2), "return_pct": round(portfolio_return_pct, 2), "gain_loss": round(portfolio_current_value - portfolio_total_cost, 2)},
        "benchmark": {"timeline": benchmark_timeline, "total_invested": round(portfolio_total_cost, 2), "final_value": round(final_benchmark_value, 2), "return_pct": round(benchmark_return_pct, 2), "gain_loss": round(final_benchmark_value - portfolio_total_cost, 2)},
        "comparison": {"difference_value": round(portfolio_current_value - final_benchmark_value, 2), "difference_pct": round(abs(portfolio_return_pct - benchmark_return_pct), 2), "winner": "Portfolio" if portfolio_current_value > final_benchmark_value else "Benchmark"}
    }
