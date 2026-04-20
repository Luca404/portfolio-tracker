import os
from supabase import create_client, Client

_client: Client | None = None


def get_supabase() -> Client:
    """Client singleton con service key — usato per query DB. Non chiamare auth.sign_in/sign_up su questo."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SECRET_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY must be set")
        _client = create_client(url, key)
    return _client


def new_supabase_auth_client() -> Client:
    """Crea un client fresco per operazioni di auth (login/register/refresh).
    NON usa il singleton per evitare di contaminarne la sessione con JWT utente che scadono."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY must be set")
    return create_client(url, key)
