from gate_server.schemas import Authorization, ClassifierInput, RiskLevel, ToolMetadata
from gate_server.stubs import StubEngine


def _make(read_only: bool, touches_remote: bool) -> ClassifierInput:
    return ClassifierInput(
        task="t",
        tool_name="f",
        tool_args={},
        tool_metadata=ToolMetadata(read_only=read_only, touches_remote=touches_remote),
    )


def test_read_only():
    g = StubEngine().classify(_make(True, False))
    assert g.authorization is Authorization.AUTHORIZED
    assert g.risk_level is RiskLevel.LOW


def test_remote_write():
    g = StubEngine().classify(_make(False, True))
    assert g.authorization is Authorization.UNAUTHORIZED
    assert g.risk_level is RiskLevel.HIGH


def test_local_write():
    g = StubEngine().classify(_make(False, False))
    assert g.authorization is Authorization.UNCERTAIN
    assert g.risk_level is RiskLevel.LOW

