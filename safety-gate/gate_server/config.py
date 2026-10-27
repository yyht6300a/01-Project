"""服务配置：读 config.json + 环境变量。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    engine: str = "stub"  # stub | hf
    host: str = "127.0.0.1"
    port: int = 8001
    model_path: str = ""
    base_model: str = ""
    local_only: bool = False
    quantize: str = "4bit"  # 4bit | 8bit | none
    device: str = "auto"
    contract: str = "history"  # history | trace
    log_dir: str = "logs"


def load_config(path: str | None = None) -> Config:
    path = path or os.environ.get("GATE_SERVER_CONFIG") or str(Path(__file__).resolve().parent.parent / "config.json")
    data: dict = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    cfg = Config()
    for key in ("engine", "host", "model_path", "base_model", "local_only", "quantize", "device", "contract", "log_dir"):
        if key in data:
            setattr(cfg, key, data[key])
    cfg.port = int(os.environ.get("PORT", data.get("port", cfg.port)))
    cfg.host = os.environ.get("HOST", cfg.host)
    cfg.engine = os.environ.get("GATE_ENGINE", cfg.engine)
    return cfg

