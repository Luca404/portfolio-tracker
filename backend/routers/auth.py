from fastapi import APIRouter, Depends, HTTPException

from schemas import UserRegister, UserLogin, Token
from utils import verify_token, get_supabase

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=Token)
def register(user: UserRegister):
    sb = get_supabase()
    try:
        res = sb.auth.sign_up({"email": user.email, "password": user.password})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not res.user:
        raise HTTPException(status_code=400, detail="Registration failed")

    return {
        "access_token": res.session.access_token if res.session else "",
        "token_type": "bearer",
        "user": {
            "id": res.user.id,
            "email": res.user.email,
            "name": res.user.email,
            "createdAt": res.user.created_at.isoformat() if res.user.created_at else None,
        },
    }


@router.post("/login", response_model=Token)
def login(user: UserLogin):
    sb = get_supabase()
    try:
        res = sb.auth.sign_in_with_password({"email": user.email, "password": user.password})
    except Exception:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    return {
        "access_token": res.session.access_token,
        "token_type": "bearer",
        "user": {
            "id": res.user.id,
            "email": res.user.email,
            "name": res.user.email,
            "createdAt": res.user.created_at.isoformat() if res.user.created_at else None,
        },
    }


@router.get("/me")
def get_current_user(user_id: str = Depends(verify_token)):
    sb = get_supabase()
    try:
        res = sb.auth.admin.get_user_by_id(user_id)
        return {"id": res.user.id, "email": res.user.email}
    except Exception:
        return {"id": user_id}
