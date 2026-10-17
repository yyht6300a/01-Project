import json

from gate_server.schemas import Authorization, ClassifierInput, GateOutput, RiskLevel


def test_gate_output_content_contains_four_fields():
    g = GateOutput(
        authorization=Authorization.UNAUTHORIZED,
        risk_level=RiskLevel.HIGH,
        risk_factors=["external_side_effect"],
        rationale="reason",
    )
    content = g.to_content()
    data = json.loads(content)
    assert data["authorization"] == "UNAUTHORIZED"
    assert data["risk_level"] == "HIGH"
    assert data["risk_factors"] == ["external_side_effect"]
    assert data["rationale"] == "reason"


def test_classifier_input_parses_trace_contract_payload():
    inp = ClassifierInput.model_validate_json(json.dumps({
        "task": "Pay the bill",
        "tool_name": "send_money",
        "tool_args": {"recipient": "US133", "amount": 0.01},
        "tool_metadata": {"read_only": False, "touches_remote": True},
        "trace": [{"function": "read_file", "args": {"file_path": "bill.txt"}}],
    }))
    assert inp.task == "Pay the bill"
    assert inp.tool_name == "send_money"
    assert inp.proposed_tool_call is None
    assert len(inp.trace) == 1

