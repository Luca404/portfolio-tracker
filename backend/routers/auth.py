from fastapi import APIRouter, Depends, HTTPException

from schemas import UserRegister, UserLogin, Token
from schemas.user import RefreshRequest
from utils import verify_token, get_supabase
from utils.supabase_client import new_supabase_auth_client

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=Token)
def register(user: UserRegister):
    sb = new_supabase_auth_client()
    try:
        res = sb.auth.sign_up({"email": user.email, "password": user.password})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not res.user:
        raise HTTPException(status_code=400, detail="Registration failed")

    return {
        "access_token": res.session.access_token if res.session else "",
        "refresh_token": res.session.refresh_token if res.session else "",
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
    sb = new_supabase_auth_client()
    try:
        res = sb.auth.sign_in_with_password({"email": user.email, "password": user.password})
    except Exception:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    return {
        "access_token": res.session.access_token,
        "refresh_token": res.session.refresh_token,
        "token_type": "bearer",
        "user": {
            "id": res.user.id,
            "email": res.user.email,
            "name": res.user.email,
            "createdAt": res.user.created_at.isoformat() if res.user.created_at else None,
        },
    }


@router.post("/refresh")
def refresh_token(body: RefreshRequest):
    sb = new_supabase_auth_client()
    try:
        res = sb.auth.refresh_session(body.refresh_token)
        return {
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
        }
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")


@router.get("/me")
def get_current_user(user_id: str = Depends(verify_token)):
    sb = get_supabase()
    try:
        res = sb.auth.admin.get_user_by_id(user_id)
        return {"id": res.user.id, "email": res.user.email}
    except Exception:
        return {"id": user_id}
