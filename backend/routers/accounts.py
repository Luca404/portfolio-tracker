from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from models import AccountModel, UserModel
from schemas import AccountCreate, AccountUpdate, Account
from utils.database import get_db
from routers.auth import get_current_user

router = APIRouter(prefix="/accounts", tags=["accounts"])


@router.get("/", response_model=List[Account])
def get_accounts(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """Get all accounts for the current user"""
    accounts = db.query(AccountModel).filter(
        AccountModel.user_id == current_user.id
    ).all()
    return accounts


@router.post("/", response_model=Account)
def create_account(
    account: AccountCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """Create a new account"""
    db_account = AccountModel(
        **account.model_dump(),
        user_id=current_user.id
    )
    db.add(db_account)
    db.commit()
    db.refresh(db_account)
    return db_account


@router.get("/{account_id}", response_model=Account)
def get_account(
    account_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """Get a specific account"""
    account = db.query(AccountModel).filter(
        AccountModel.id == account_id,
        AccountModel.user_id == current_user.id
    ).first()

    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    return account


@router.put("/{account_id}", response_model=Account)
def update_account(
    account_id: int,
    account_update: AccountUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """Update an account"""
    db_account = db.query(AccountModel).filter(
        AccountModel.id == account_id,
        AccountModel.user_id == current_user.id
    ).first()

    if not db_account:
        raise HTTPException(status_code=404, detail="Account not found")

    update_data = account_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_account, field, value)

    db.commit()
    db.refresh(db_account)
    return db_account


@router.delete("/{account_id}")
def delete_account(
    account_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """Delete an account"""
    db_account = db.query(AccountModel).filter(
        AccountModel.id == account_id,
        AccountModel.user_id == current_user.id
    ).first()

    if not db_account:
        raise HTTPException(status_code=404, detail="Account not found")

    db.delete(db_account)
    db.commit()
    return {"message": "Account deleted successfully"}
