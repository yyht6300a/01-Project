"""HFEngine：加载 train.py 产出的 checkpoint 做真实推理。"""

from __future__ import annotations

import json
from pathlib import Path

from .engine import Engine
from .prompt import MODEL_SYSTEM_PROMPT, SYSTEM_PROMPT, build_model_input, build_user_text
from .schemas import Authorization, ClassifierInput, GateOutput, RiskLevel


class HFEngine(Engine):
    def __init__(self, model_path: str, local_only: bool = False, device: str = "auto",
                 base_model: str = "", contract: str = "history"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.contract = contract
        p = Path(model_path)
        if (p / "adapter_config.json").exists():
            from peft import PeftModel

            if base_model:
                base = base_model
            else:
                with open(p / "adapter_config.json", encoding="utf-8") as f:
                    base = json.load(f)["base_model_name_or_path"]
            self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=local_only)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16, device_map=device, local_files_only=local_only
            )
            self.model = PeftModel.from_pretrained(base_model_obj, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device, local_files_only=local_only
            )
        self.model.eval()

    def classify(self, inp: ClassifierInput) -> GateOutput:
        import torch

        if self.contract == "trace":
            user_text = build_model_input(inp.model_dump(mode="json"))
            messages = [{"role": "system", "content": MODEL_SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
            max_new_tokens = 384
        else:
            user_text = build_user_text(inp.user_task, inp.history, inp.proposed_tool_call.model_dump(mode="json"))
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
            max_new_tokens = 128
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        data = json.loads(text)
        return GateOutput(
            authorization=Authorization(data["authorization"].strip().upper()),
            risk_level=RiskLevel(data["risk_level"].strip().upper()),
            risk_factors=data.get("risk_factors") or [],
            rationale=data.get("rationale") or "",
        )

