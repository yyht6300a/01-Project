import json

from fastapi.testclient import TestClient

from gate_server.app import create_app
from gate_server.config import Config


def test_stub_end_to_end(tmp_path):
    app = create_app(Config(engine="stub", log_dir=str(tmp_path)))
    client = TestClient(app)
    body = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": json.dumps({
                "user_task": "t",
                "history": [],
                "proposed_tool_call": {"function": "send_money", "args": {}},
                "tool_metadata": {"read_only": False, "touches_remote": True},
            })},
        ],
    }
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    data = r.json()
    content = json.loads(data["choices"][0]["message"]["content"])
    assert content["authorization"] == "UNAUTHORIZED"
    assert content["risk_level"] == "HIGH"
    assert isinstance(content["risk_factors"], list)
    assert isinstance(content["rationale"], str)
    assert data["choices"][0]["finish_reason"] == "stop"


def test_health(tmp_path):
    app = create_app(Config(engine="stub", log_dir=str(tmp_path)))
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["engine"] == "stub"


def test_trace_contract_end_to_end_stub(tmp_path):
    app = create_app(Config(engine="stub", log_dir=str(tmp_path)))
    client = TestClient(app)
    body = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": json.dumps({
                "task": "t",
                "tool_name": "send_money",
                "tool_args": {},
                "tool_metadata": {"read_only": False, "touches_remote": True},
                "trace": [],
            })},
        ],
    }
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    content = json.loads(r.json()["choices"][0]["message"]["content"])
    assert content["authorization"] == "UNAUTHORIZED"
    assert content["risk_level"] == "HIGH"

