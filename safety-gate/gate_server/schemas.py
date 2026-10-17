"""契约模型：请求/响应 + 安全门输出（4 字段）。"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Authorization(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    UNAUTHORIZED = "UNAUTHORIZED"
    UNCERTAIN = "UNCERTAIN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    HIGH = "HIGH"


class ToolMetadata(BaseModel):
    read_only: bool = False
    touches_remote: bool = False
    category: str = "other"


class ToolCall(BaseModel):
    function: str
    args: dict[str, Any] = Field(default_factory=dict)


class ClassifierInput(BaseModel):
    """插件发来的结构化输入（与模型组训练输入 task/tool_name/tool_args/tool_metadata/trace 一致）。"""

    task: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_metadata: ToolMetadata = Field(default_factory=ToolMetadata)
    trace: list[ToolCall] = Field(default_factory=list)


class GateOutput(BaseModel):
    """安全门模型产出的 4 字段结果。"""

    authorization: Authorization
    risk_level: RiskLevel
    risk_factors: list[str] = Field(default_factory=list)
    rationale: str = ""

    def to_content(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 128
    response_format: dict[str, Any] | None = None

    def user_content(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

