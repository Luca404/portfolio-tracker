from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, extract
from sqlalchemy.orm import Session

from models import TransactionModel, TransactionType, UserModel, OrderModel
from schemas import TransactionCreate, TransactionUpdate, TransactionResponse, TransactionStats
from utils import get_db, verify_token

router = APIRouter(prefix="/api/transactions", tags=["transactions"])


@router.post("", response_model=TransactionResponse)
def create_transaction(
    transaction: TransactionCreate,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Crea una nuova transazione. Se è un investimento, crea anche un order in pfTrackr."""

    # Crea la transazione
    new_transaction = TransactionModel(
        user_id=user.id,
        type=transaction.type,
        category=transaction.category,
        amount=transaction.amount,
        description=transaction.description,
        date=transaction.date,
        ticker=transaction.ticker,
        quantity=transaction.quantity,
        price=transaction.price,
    )
    db.add(new_transaction)

    # Se è un investimento, crea anche un order nella tabella orders di pfTrackr
    if transaction.type == TransactionType.INVESTMENT and transaction.ticker:
        # Trova o crea il portfolio di default dell'utente
        from models import PortfolioModel
        portfolio = db.execute(
            select(PortfolioModel).where(PortfolioModel.user_id == user.id)
        ).scalars().first()

        if not portfolio:
            # Crea un portfolio di default
            portfolio = PortfolioModel(
                user_id=user.id,
                name="Portfolio Principale",
            )
            db.add(portfolio)
            db.flush()  # Per ottenere l'ID

        # Crea l'order
        new_order = OrderModel(
            portfolio_id=portfolio.id,
            symbol=transaction.ticker,
            quantity=transaction.quantity or 0,
            price=transaction.price or 0,
            date=transaction.date,
            type="buy",  # Sempre buy per ora
        )
        db.add(new_order)

    db.commit()
    db.refresh(new_transaction)

    return new_transaction


@router.get("", response_model=list[TransactionResponse])
def get_transactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    type: Optional[str] = None,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Ottieni tutte le transazioni dell'utente con filtri opzionali."""

    query = select(TransactionModel).where(TransactionModel.user_id == user.id)

    if start_date:
        query = query.where(TransactionModel.date >= datetime.fromisoformat(start_date))
    if end_date:
        query = query.where(TransactionModel.date <= datetime.fromisoformat(end_date))
    if category:
        query = query.where(TransactionModel.category == category)
    if type:
        query = query.where(TransactionModel.type == type)

    query = query.order_by(TransactionModel.date.desc())

    transactions = db.execute(query).scalars().all()
    return transactions


@router.get("/stats", response_model=TransactionStats)
def get_transaction_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Ottieni statistiche sulle transazioni."""

    query = select(TransactionModel).where(TransactionModel.user_id == user.id)

    if start_date:
        query = query.where(TransactionModel.date >= datetime.fromisoformat(start_date))
    if end_date:
        query = query.where(TransactionModel.date <= datetime.fromisoformat(end_date))

    transactions = db.execute(query).scalars().all()

    total_expenses = sum(t.amount for t in transactions if t.type == TransactionType.EXPENSE)
    total_income = sum(t.amount for t in transactions if t.type == TransactionType.INCOME)
    total_investments = sum(t.amount for t in transactions if t.type == TransactionType.INVESTMENT)

    balance = total_income - total_expenses - total_investments

    # Spese per categoria
    expenses_by_category = {}
    for t in transactions:
        if t.type == TransactionType.EXPENSE:
            if t.category not in expenses_by_category:
                expenses_by_category[t.category] = 0
            expenses_by_category[t.category] += t.amount

    # Trend mensile
    monthly_data = {}
    for t in transactions:
        month_key = t.date.strftime("%Y-%m")
        if month_key not in monthly_data:
            monthly_data[month_key] = {"month": month_key, "expenses": 0, "income": 0}

        if t.type == TransactionType.EXPENSE:
            monthly_data[month_key]["expenses"] += t.amount
        elif t.type == TransactionType.INCOME:
            monthly_data[month_key]["income"] += t.amount

    monthly_trend = sorted(monthly_data.values(), key=lambda x: x["month"])

    return TransactionStats(
        total_expenses=total_expenses,
        total_income=total_income,
        total_investments=total_investments,
        balance=balance,
        expenses_by_category=expenses_by_category,
        monthly_trend=monthly_trend,
    )


@router.get("/{transaction_id}", response_model=TransactionResponse)
def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Ottieni una singola transazione."""

    transaction = db.execute(
        select(TransactionModel).where(
            TransactionModel.id == transaction_id,
            TransactionModel.user_id == user.id,
        )
    ).scalar_one_or_none()

    if not transaction:
        raise HTTPException(status_code=404, detail="Transazione non trovata")

    return transaction


@router.put("/{transaction_id}", response_model=TransactionResponse)
def update_transaction(
    transaction_id: int,
    transaction_update: TransactionUpdate,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Aggiorna una transazione esistente."""

    transaction = db.execute(
        select(TransactionModel).where(
            TransactionModel.id == transaction_id,
            TransactionModel.user_id == user.id,
        )
    ).scalar_one_or_none()

    if not transaction:
        raise HTTPException(status_code=404, detail="Transazione non trovata")

    # Aggiorna solo i campi forniti
    update_data = transaction_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(transaction, key, value)

    db.commit()
    db.refresh(transaction)

    return transaction


@router.delete("/{transaction_id}")
def delete_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Elimina una transazione."""

    transaction = db.execute(
        select(TransactionModel).where(
            TransactionModel.id == transaction_id,
            TransactionModel.user_id == user.id,
        )
    ).scalar_one_or_none()

    if not transaction:
        raise HTTPException(status_code=404, detail="Transazione non trovata")

    db.delete(transaction)
    db.commit()

    return {"message": "Transazione eliminata con successo"}
