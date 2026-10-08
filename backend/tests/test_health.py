"""Exercise the public keepalive without loading application data or credentials."""

from pathlib import Path
import socket
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from health import router


def test_health_is_public_and_does_not_access_external_services(monkeypatch):
    for name in ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_KEY"):
        monkeypatch.delenv(name, raising=False)

    def forbid_network(*args, **kwargs):
        raise AssertionError("Health must not call external services")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers["cache-control"] == "no-store"
