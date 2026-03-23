from fastapi import APIRouter, Depends, HTTPException

from schemas import AccountCreate, AccountUpdate, Account
from utils import get_supabase, verify_token

router = APIRouter(prefix="/api/accounts", tags=["accounts"])

DEFAULT_ACCOUNTS = [
    {"name": "Conto Corrente", "icon": "🏦", "initial_balance": 0.0, "is_favorite": True},
    {"name": "Contanti", "icon": "💵", "initial_balance": 0.0, "is_favorite": False},
]


def _calc_balance(account: dict, transactions: list) -> float:
    balance = float(account.get("initial_balance") or 0)
    for t in transactions:
        t_type = t.get("type", "")
        amount = float(t.get("amount") or 0)
        if t_type == "income":
            balance += amount
        elif t_type in ("expense", "investment"):
            balance -= amount
    return balance


def _account_response(account: dict, current_balance: float) -> dict:
    return {
        "id": account["id"],
        "user_id": account["user_id"],
        "name": account["name"],
        "icon": account.get("icon", "💳"),
        "initial_balance": float(account.get("initial_balance") or 0),
        "is_favorite": account.get("is_favorite", False),
        "current_balance": current_balance,
        "created_at": account.get("created_at"),
        "updated_at": account.get("updated_at"),
    }


@router.get("/")
def get_accounts(user_id: str = Depends(verify_token)):
    sb = get_supabase()
    accounts = sb.table("accounts").select("*").eq("user_id", user_id).execute().data

    if not accounts:
        rows = [{"user_id": user_id, **a} for a in DEFAULT_ACCOUNTS]
        accounts = sb.table("accounts").insert(rows).execute().data

    result = []
    for account in accounts:
        transactions = sb.table("transactions").select("type,amount").eq("account_id", account["id"]).execute().data
        result.append(_account_response(account, _calc_balance(account, transactions)))
    return result


@router.post("/")
def create_account(account: AccountCreate, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    result = sb.table("accounts").insert({"user_id": user_id, **account.model_dump()}).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create account")
    acc = result.data[0]
    return _account_response(acc, float(acc.get("initial_balance") or 0))


@router.get("/{account_id}")
def get_account(account_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    result = sb.table("accounts").select("*").eq("id", account_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Account not found")
    account = result.data[0]
    transactions = sb.table("transactions").select("type,amount").eq("account_id", account_id).execute().data
    return _account_response(account, _calc_balance(account, transactions))


@router.put("/{account_id}")
def update_account(account_id: int, account_update: AccountUpdate, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    result = sb.table("accounts").select("*").eq("id", account_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Account not found")
    account = result.data[0]

    update_data = account_update.model_dump(exclude_unset=True)

    if "current_balance" in update_data:
        desired_balance = float(update_data.pop("current_balance"))
        transactions = sb.table("transactions").select("type,amount").eq("account_id", account_id).execute().data
        transactions_sum = sum(
            float(t["amount"]) * (1 if t["type"] == "income" else -1)
            for t in transactions if t["type"] in ("income", "expense", "investment")
        )
        update_data["initial_balance"] = desired_balance - transactions_sum

    if update_data.get("is_favorite") is True:
        sb.table("accounts").update({"is_favorite": False}).eq("user_id", user_id).neq("id", account_id).execute()

    updated = sb.table("accounts").update(update_data).eq("id", account_id).execute()
    acc = updated.data[0] if updated.data else {**account, **update_data}
    transactions = sb.table("transactions").select("type,amount").eq("account_id", account_id).execute().data
    return _account_response(acc, _calc_balance(acc, transactions))


@router.delete("/{account_id}")
def delete_account(account_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    result = sb.table("accounts").select("id").eq("id", account_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Account not found")

    transaction_count = len(sb.table("transactions").select("id").eq("account_id", account_id).execute().data)
    if transaction_count > 0:
        raise HTTPException(status_code=400, detail=f"Cannot delete account: {transaction_count} transaction(s) associated")

    sb.table("accounts").delete().eq("id", account_id).execute()
    return {"message": "Account deleted successfully"}
