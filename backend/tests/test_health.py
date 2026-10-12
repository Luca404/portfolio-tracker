"""Exercise the public keepalive without loading application data or credentials."""

from pathlib import Path
import socket
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from health import router


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_health_is_public_and_does_not_access_external_services(monkeypatch, revision):
    for name in ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_KEY"):
        monkeypatch.delenv(name, raising=False)
    if revision is None:
        monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    else:
        monkeypatch.setenv("RENDER_GIT_COMMIT", revision)

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
        assert response.headers.get("x-backend-revision") == revision
