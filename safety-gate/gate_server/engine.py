"""引擎抽象 + 工厂：stub / hf 切换点。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import ClassifierInput, GateOutput


class Engine(ABC):
    @abstractmethod
    def classify(self, inp: ClassifierInput) -> GateOutput:
        ...


def make_engine(cfg) -> Engine:
    if cfg.engine == "stub":
        from .stubs import StubEngine

        return StubEngine()
    if cfg.engine == "hf":
        from .hf_engine import HFEngine

        return HFEngine(model_path=cfg.model_path, local_only=cfg.local_only,
                        device=cfg.device, base_model=cfg.base_model, contract=cfg.contract)
    raise ValueError(f"unknown engine: {cfg.engine}")

