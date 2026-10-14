
from fastapi.testclient import TestClient

from deploy_check import app

client = TestClient(app)


def test_root_returns_project_status():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["project"] == "SACSIA"
    assert data["status"] == "online"


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_documentation_is_available():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "SACSIA Deployment Check"
