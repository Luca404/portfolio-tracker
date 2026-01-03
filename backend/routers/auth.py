from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from models import UserModel
from schemas import UserRegister, UserLogin, Token
from utils import get_db, verify_password, get_password_hash, create_access_token, verify_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=Token)
def register(user: UserRegister, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(user.password)
    new_user = UserModel(email=user.email.lower(), username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email or username already registered")
    db.refresh(new_user)

    access_token = create_access_token(data={"sub": new_user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": new_user.id,
            "email": new_user.email,
            "name": new_user.username,
            "createdAt": new_user.created_at.isoformat() if new_user.created_at else None,
        },
    }


@router.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.execute(select(UserModel).where(UserModel.email == user.email.lower())).scalar_one_or_none()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": db_user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": db_user.id,
            "email": db_user.email,
            "name": db_user.username,
            "createdAt": db_user.created_at.isoformat() if db_user.created_at else None,
        },
    }


@router.get("/me")
def get_current_user(user: UserModel = Depends(verify_token)):
    return {"email": user.email, "username": user.username}
