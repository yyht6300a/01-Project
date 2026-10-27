"""HFEngine：加载 train.py 产出的 checkpoint 做真实推理。"""

from __future__ import annotations

import json
from pathlib import Path

from .engine import Engine
from .prompt import MODEL_SYSTEM_PROMPT, SYSTEM_PROMPT, build_model_input, build_user_text
from .schemas import Authorization, ClassifierInput, GateOutput, RiskLevel


class HFEngine(Engine):
    def __init__(self, model_path: str, local_only: bool = False, device: str = "auto",
                 quantize: str = "4bit", base_model: str = "", contract: str = "history"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.contract = contract
        p = Path(model_path)
        quant_kwargs = {}
        if quantize in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quantize == "4bit"),
                load_in_8bit=(quantize == "8bit"),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        if (p / "adapter_config.json").exists():
            from peft import PeftModel

            if base_model:
                base = base_model
            else:
                with open(p / "adapter_config.json", encoding="utf-8") as f:
                    base = json.load(f)["base_model_name_or_path"]
            self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=local_only)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base, device_map=device, local_files_only=local_only,
                **({"torch_dtype": torch.bfloat16} if not quant_kwargs else {}),
                **quant_kwargs,
            )
            self.model = PeftModel.from_pretrained(base_model_obj, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map=device, local_files_only=local_only,
                **({"torch_dtype": torch.bfloat16} if not quant_kwargs else {}),
                **quant_kwargs,
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
            max_new_tokens = 64
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        data = self._parse_json(text)
        return GateOutput(
            authorization=Authorization(data["authorization"].strip().upper()),
            risk_level=RiskLevel(data["risk_level"].strip().upper()),
            risk_factors=data.get("risk_factors") or [],
            rationale=data.get("rationale") or "",
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                return json.loads(text[start:end + 1])
            raise

