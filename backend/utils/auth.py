import os

import requests
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verifica il JWT Supabase tramite HTTP diretto a /auth/v1/user.
    NON usa il client singleton per non contaminarne la sessione.
    """
    token = credentials.credentials
    try:
        res = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_SERVICE_KEY,
            },
            timeout=5,
        )
        if res.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = res.json().get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return str(user_id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
