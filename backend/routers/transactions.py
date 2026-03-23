from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from schemas import TransactionCreate, TransactionUpdate, TransactionResponse, TransactionStats
from utils import get_supabase, verify_token

router = APIRouter(prefix="/api/transactions", tags=["transactions"])


@router.post("")
def create_transaction(transaction: TransactionCreate, user_id: str = Depends(verify_token)):
    sb = get_supabase()

    account_id = transaction.account_id
    if account_id is None:
        fav = sb.table("accounts").select("id").eq("user_id", user_id).eq("is_favorite", True).limit(1).execute().data
        if fav:
            account_id = fav[0]["id"]
        else:
            first = sb.table("accounts").select("id").eq("user_id", user_id).limit(1).execute().data
            if not first:
                raise HTTPException(status_code=400, detail="Nessun conto disponibile.")
            account_id = first[0]["id"]

    date_val = transaction.date
    if hasattr(date_val, "isoformat"):
        date_val = date_val.isoformat()

    insert_data = {
        "user_id": user_id,
        "account_id": account_id,
        "type": transaction.type if isinstance(transaction.type, str) else transaction.type.value,
        "category": transaction.category,
        "subcategory": transaction.subcategory,
        "amount": transaction.amount,
        "description": transaction.description,
        "date": date_val,
        "ticker": transaction.ticker,
        "quantity": transaction.quantity,
        "price": transaction.price,
    }
    result = sb.table("transactions").insert(insert_data).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create transaction")

    new_transaction = result.data[0]

    # Se è un investimento, crea anche un order
    if new_transaction.get("type") == "investment" and transaction.ticker:
        portfolio = sb.table("portfolios").select("id").eq("user_id", user_id).limit(1).execute().data
        if not portfolio:
            portfolio = sb.table("portfolios").insert({"user_id": user_id, "name": "Portfolio Principale"}).execute().data
        if portfolio:
            portfolio_id = portfolio[0]["id"]
            sb.table("orders").insert({
                "portfolio_id": portfolio_id,
                "symbol": transaction.ticker,
                "quantity": transaction.quantity or 0,
                "price": transaction.price or 0,
                "date": date_val,
                "order_type": "buy",
            }).execute()

    return new_transaction


@router.get("")
def get_transactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    type: Optional[str] = None,
    user_id: str = Depends(verify_token),
):
    sb = get_supabase()
    query = sb.table("transactions").select("*").eq("user_id", user_id)
    if start_date:
        query = query.gte("date", start_date)
    if end_date:
        query = query.lte("date", end_date)
    if category:
        query = query.eq("category", category)
    if type:
        query = query.eq("type", type)
    query = query.order("date", desc=True)
    return query.execute().data


@router.get("/stats")
def get_transaction_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: str = Depends(verify_token),
):
    sb = get_supabase()
    query = sb.table("transactions").select("*").eq("user_id", user_id)
    if start_date:
        query = query.gte("date", start_date)
    if end_date:
        query = query.lte("date", end_date)
    transactions = query.execute().data

    total_expenses = sum(float(t["amount"]) for t in transactions if t.get("type") == "expense")
    total_income = sum(float(t["amount"]) for t in transactions if t.get("type") == "income")
    total_investments = sum(float(t["amount"]) for t in transactions if t.get("type") == "investment")
    balance = total_income - total_expenses - total_investments

    expenses_by_category = {}
    for t in transactions:
        if t.get("type") == "expense":
            cat = t.get("category", "")
            expenses_by_category[cat] = expenses_by_category.get(cat, 0) + float(t["amount"])

    monthly_data = {}
    for t in transactions:
        date_str = t.get("date", "")
        month_key = date_str[:7] if date_str else ""
        if month_key:
            if month_key not in monthly_data:
                monthly_data[month_key] = {"month": month_key, "expenses": 0, "income": 0}
            if t.get("type") == "expense":
                monthly_data[month_key]["expenses"] += float(t["amount"])
            elif t.get("type") == "income":
                monthly_data[month_key]["income"] += float(t["amount"])

    return {
        "total_expenses": total_expenses,
        "total_income": total_income,
        "total_investments": total_investments,
        "balance": balance,
        "expenses_by_category": expenses_by_category,
        "monthly_trend": sorted(monthly_data.values(), key=lambda x: x["month"]),
    }


@router.get("/{transaction_id}")
def get_transaction(transaction_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    result = sb.table("transactions").select("*").eq("id", transaction_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Transazione non trovata")
    return result.data[0]


@router.put("/{transaction_id}")
def update_transaction(transaction_id: int, transaction_update: TransactionUpdate, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    existing = sb.table("transactions").select("id").eq("id", transaction_id).eq("user_id", user_id).execute().data
    if not existing:
        raise HTTPException(status_code=404, detail="Transazione non trovata")

    update_data = transaction_update.dict(exclude_unset=True)
    if "date" in update_data and hasattr(update_data["date"], "isoformat"):
        update_data["date"] = update_data["date"].isoformat()
    if "type" in update_data and not isinstance(update_data["type"], str):
        update_data["type"] = update_data["type"].value

    result = sb.table("transactions").update(update_data).eq("id", transaction_id).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to update transaction")
    return result.data[0]


@router.delete("/{transaction_id}")
def delete_transaction(transaction_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    existing = sb.table("transactions").select("id").eq("id", transaction_id).eq("user_id", user_id).execute().data
    if not existing:
        raise HTTPException(status_code=404, detail="Transazione non trovata")
    sb.table("transactions").delete().eq("id", transaction_id).execute()
    return {"message": "Transazione eliminata con successo"}
