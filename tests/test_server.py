from fastapi.testclient import TestClient

from compass.scorer import UniformScorer
from compass.server import create_app

client = TestClient(create_app(UniformScorer()))
REQUEST = {
    "model": "compass-latest",
    "state": "Please cancel my order before it ships.",
    "questions": {"decision": {"type": "choice", "instructions": "Which intent?", "criteria": {"cancel": "Cancel an order", "track": "Where is my order"}}},
}


def test_systemone_answers_in_the_typesafe_wire_format():
    r = client.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "compass-uniform-0.0.0"
    assert data["answers"]["decision"]["type"] == "choice"
    assert data["answers"]["decision"]["probabilities"] == {"cancel": 0.5, "track": 0.5}
    assert data["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_error_statuses():
    assert client.post("/v1/systemone", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 400
    bad = client.post("/v1/systemone", json={**REQUEST, "questions": {}})
    assert bad.status_code == 422 and bad.json()["error"]["type"] == "invalid_request"
    assert client.post("/v1/systemone", json={**REQUEST, "model": "someone-else"}).status_code == 404


def test_models_and_health():
    assert client.get("/v1/models").json()["data"][0]["id"] == "compass-uniform-0.0.0"
    assert client.get("/healthz").json() == {"ok": True, "model": "compass-uniform-0.0.0"}
