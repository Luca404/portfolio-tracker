from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from models import AccountModel, UserModel
from schemas import AccountCreate, AccountUpdate, Account
from utils.database import get_db
from utils import verify_token

router = APIRouter(prefix="/api/accounts", tags=["accounts"])


@router.get("/", response_model=List[Account])
def get_accounts(
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token)
):
    """Get all accounts for the current user with calculated current balance"""
    from models import TransactionModel

    accounts = db.query(AccountModel).filter(
        AccountModel.user_id == user.id
    ).all()

    # Se l'utente non ha conti, crea quelli di default
    if len(accounts) == 0:
        # Crea Conto Corrente (preferito di default)
        account1 = AccountModel(
            user_id=user.id,
            name="Conto Corrente",
            icon="🏦",
            initial_balance=0.0,
            is_favorite=True
        )
        db.add(account1)

        # Crea Contanti
        account2 = AccountModel(
            user_id=user.id,
            name="Contanti",
            icon="💵",
            initial_balance=0.0,
            is_favorite=False
        )
        db.add(account2)

        db.commit()
        db.refresh(account1)
        db.refresh(account2)

        accounts = [account1, account2]

    # Calcola il saldo corrente per ogni conto
    result = []
    for account in accounts:
        # Ottieni tutte le transazioni per questo conto
        transactions = db.query(TransactionModel).filter(
            TransactionModel.account_id == account.id
        ).all()

        # Calcola il saldo corrente
        current_balance = account.initial_balance
        for transaction in transactions:
            if transaction.type == 'income':
                current_balance += transaction.amount
            elif transaction.type in ['expense', 'investment']:
                current_balance -= transaction.amount
            # 'transfer' non modifica il saldo del singolo conto (verrà gestito in futuro)

        # Crea un dizionario con i dati dell'account + current_balance
        account_dict = {
            "id": account.id,
            "user_id": account.user_id,
            "name": account.name,
            "icon": account.icon,
            "initial_balance": account.initial_balance,
            "is_favorite": account.is_favorite,
            "current_balance": current_balance,
            "created_at": account.created_at,
            "updated_at": account.updated_at
        }
        result.append(account_dict)

    return result


@router.post("/", response_model=Account)
def create_account(
    account: AccountCreate,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token)
):
    """Create a new account"""
    db_account = AccountModel(
        **account.model_dump(),
        user_id=user.id
    )
    db.add(db_account)
    db.commit()
    db.refresh(db_account)

    # Restituisci con current_balance (uguale a initial_balance per un nuovo conto)
    return {
        "id": db_account.id,
        "user_id": db_account.user_id,
        "name": db_account.name,
        "icon": db_account.icon,
        "initial_balance": db_account.initial_balance,
        "is_favorite": db_account.is_favorite,
        "current_balance": db_account.initial_balance,
        "created_at": db_account.created_at,
        "updated_at": db_account.updated_at
    }


@router.get("/{account_id}", response_model=Account)
def get_account(
    account_id: int,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token)
):
    """Get a specific account with calculated current balance"""
    from models import TransactionModel

    account = db.query(AccountModel).filter(
        AccountModel.id == account_id,
        AccountModel.user_id == user.id
    ).first()

    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    # Calcola il saldo corrente
    transactions = db.query(TransactionModel).filter(
        TransactionModel.account_id == account.id
    ).all()

    current_balance = account.initial_balance
    for transaction in transactions:
        if transaction.type == 'income':
            current_balance += transaction.amount
        elif transaction.type in ['expense', 'investment']:
            current_balance -= transaction.amount

    return {
        "id": account.id,
        "user_id": account.user_id,
        "name": account.name,
        "icon": account.icon,
        "initial_balance": account.initial_balance,
        "is_favorite": account.is_favorite,
        "current_balance": current_balance,
        "created_at": account.created_at,
        "updated_at": account.updated_at
    }


@router.put("/{account_id}", response_model=Account)
def update_account(
    account_id: int,
    account_update: AccountUpdate,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token)
):
    """Update an account"""
    from models import TransactionModel

    db_account = db.query(AccountModel).filter(
        AccountModel.id == account_id,
        AccountModel.user_id == user.id
    ).first()

    if not db_account:
        raise HTTPException(status_code=404, detail="Account not found")

    update_data = account_update.model_dump(exclude_unset=True)

    # Se viene fornito current_balance, calcola il nuovo initial_balance
    if 'current_balance' in update_data:
        desired_current_balance = update_data.pop('current_balance')

        # Calcola la somma delle transazioni
        transactions = db.query(TransactionModel).filter(
            TransactionModel.account_id == account_id
        ).all()

        transactions_sum = 0.0
        for transaction in transactions:
            if transaction.type == 'income':
                transactions_sum += transaction.amount
            elif transaction.type in ['expense', 'investment']:
                transactions_sum -= transaction.amount

        # Calcola il nuovo initial_balance: initial_balance + transactions_sum = desired_current_balance
        # quindi: initial_balance = desired_current_balance - transactions_sum
        new_initial_balance = desired_current_balance - transactions_sum
        update_data['initial_balance'] = new_initial_balance

    # Se si sta impostando questo conto come preferito, rimuovi il preferito dagli altri
    if update_data.get('is_favorite') == True:
        db.query(AccountModel).filter(
            AccountModel.user_id == user.id,
            AccountModel.id != account_id
        ).update({"is_favorite": False})

    for field, value in update_data.items():
        setattr(db_account, field, value)

    db.commit()
    db.refresh(db_account)

    # Calcola il saldo corrente finale
    transactions = db.query(TransactionModel).filter(
        TransactionModel.account_id == db_account.id
    ).all()

    current_balance = db_account.initial_balance
    for transaction in transactions:
        if transaction.type == 'income':
            current_balance += transaction.amount
        elif transaction.type in ['expense', 'investment']:
            current_balance -= transaction.amount

    return {
        "id": db_account.id,
        "user_id": db_account.user_id,
        "name": db_account.name,
        "icon": db_account.icon,
        "initial_balance": db_account.initial_balance,
        "is_favorite": db_account.is_favorite,
        "current_balance": current_balance,
        "created_at": db_account.created_at,
        "updated_at": db_account.updated_at
    }


@router.delete("/{account_id}")
def delete_account(
    account_id: int,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token)
):
    """Delete an account"""
    db_account = db.query(AccountModel).filter(
        AccountModel.id == account_id,
        AccountModel.user_id == user.id
    ).first()

    if not db_account:
        raise HTTPException(status_code=404, detail="Account not found")

    # Controlla se ci sono transazioni associate a questo conto
    # Nota: questo richiede che TransactionModel abbia un campo account_id
    try:
        from models import TransactionModel
        transaction_count = db.query(TransactionModel).filter(
            TransactionModel.account_id == account_id
        ).count()

        if transaction_count > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete account: {transaction_count} transaction(s) are associated with this account"
            )
    except AttributeError:
        # Se TransactionModel non ha ancora il campo account_id, continua
        pass

    db.delete(db_account)
    db.commit()
    return {"message": "Account deleted successfully"}
