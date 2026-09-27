import json

import pytest
from fastapi.testclient import TestClient

from compass.scope_check import (
    AUTHORIZE,
    CASES,
    REJECT,
    build_request,
    create_scope_check_app,
    execution_policy,
    is_loopback_http_url,
)


def compass_response(case, model="fake-compass"):
    probabilities = {AUTHORIZE: 0.1, REJECT: 0.1}
    probabilities[case.expected_scope] = 0.9
    return {
        "model": model,
        "answers": {
            "scope": {
                "type": "choice",
                "choice": case.expected_scope,
                "confidence": 0.8,
                "probabilities": probabilities,
            }
        },
        "usage": {"input_tokens": 123, "output_tokens": 0},
    }


def test_scope_request_uses_only_trusted_request_and_proposed_action():
    poisoned = CASES[1]
    request = build_request(poisoned)
    state = json.dumps(request["state"])
    assert request["questions"]["scope"]["type"] == "choice"
    assert request["state"]["trusted_user_request"] == poisoned.user_request
    assert request["state"]["proposed_tool_action"] == poisoned.proposed_action
    assert "This is an important message from Emma Johnson" not in state
    assert "untrusted tool-result text is not present" in request["state"]["boundary"].lower()


def test_deterministic_policy_retains_control_over_execution():
    assert execution_policy(CASES[0].proposed_action, AUTHORIZE)[0] == "simulated_read"
    assert execution_policy(CASES[1].proposed_action, REJECT)[0] == "blocked"
    assert execution_policy(CASES[1].proposed_action, AUTHORIZE)[0] == "hard_blocked"
    assert execution_policy(CASES[3].proposed_action, AUTHORIZE)[0] == "awaiting_confirmation"


def test_scope_dashboard_replays_authorized_and_rejected_actions_with_injected_client():
    def fake_ask(request, endpoint):
        assert endpoint == "http://127.0.0.1:8000/v1/systemone"
        action = request["state"]["proposed_tool_action"]
        case = next(case for case in CASES if case.proposed_action == action)
        return compass_response(case), 12.3

    app = create_scope_check_app(ask_function=fake_ask)
    client = TestClient(app)
    session_id = "scope_trace"
    assert client.get("/").status_code == 200
    assert client.post("/api/reset", json={"session_id": session_id}).json()["status"] == "ready"

    first = client.post("/api/step", json={"session_id": session_id}).json()
    assert first["events"][-1]["scope"] == AUTHORIZE
    assert first["events"][-1]["disposition"] == "simulated_read"

    second = client.post("/api/step", json={"session_id": session_id}).json()
    assert second["events"][-1]["scope"] == REJECT
    assert second["events"][-1]["disposition"] == "blocked"

    third = client.post("/api/step", json={"session_id": session_id}).json()
    assert third["events"][-1]["scope"] == REJECT

    fourth = client.post("/api/step", json={"session_id": session_id}).json()
    assert fourth["status"] == "awaiting_confirmation"
    assert fourth["events"][-1]["disposition"] == "awaiting_confirmation"

    complete = client.post("/api/confirm", json={"session_id": session_id}).json()
    assert complete["status"] == "complete"
    assert complete["events"][-1]["disposition"] == "simulated_write"
    assert complete["events"][-1]["confirmed"] is True


def test_scope_check_refuses_external_model_endpoints():
    assert is_loopback_http_url("http://localhost:8000/v1/systemone")
    assert not is_loopback_http_url("https://api.example.test/v1/systemone")
    with pytest.raises(ValueError):
        create_scope_check_app("http://api.example.test/v1/systemone")
