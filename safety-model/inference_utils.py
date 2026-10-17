"""Model loading + prompt encoding + generation helpers shared by train / eval / probe."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompts
from safety_data import serialize_input


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def pick_dtype(device: torch.device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_model_and_tokenizer(base_model: str, adapter: str | None = None, device: str = "auto",
                             dtype=None, trainable: bool = False):
    dev = pick_device(device)
    dtype = dtype or pick_dtype(dev)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    model.to(dev)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer, dev


def _flat_ids(obj) -> list[int]:
    """Normalize tokenizer outputs (BatchEncoding / tensor / nested list) to list[int]."""
    if not isinstance(obj, list):
        if hasattr(obj, "keys") and "input_ids" in obj:
            obj = obj["input_ids"]
        elif hasattr(obj, "tolist"):
            obj = obj.tolist()
    obj = list(obj)
    if obj and isinstance(obj[0], (list, tuple)):
        obj = list(obj[0])
    return [int(t) for t in obj]


def chat_prompt_ids(tokenizer, sample_input: dict) -> list[int]:
    """Frozen prompt: system instruction + compact JSON of the tool-call context."""
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": serialize_input(sample_input)},
    ]
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return _flat_ids(out)


def encode_sample(sample: dict, tokenizer, max_seq_len: int, answer_fn=None):
    """Return (prompt_ids, answer_ids) with the answer carrying the EOS token.

    When prompt + answer exceed max_seq_len the prompt is truncated from the
    LEFT (oldest context dropped first). Answers must fit in max_seq_len - 16.
    """
    prompt_ids = chat_prompt_ids(tokenizer, sample["input"])
    answer = (answer_fn or prompts.answer_json)(sample["label"])
    answer_ids = _flat_ids(tokenizer(answer, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    budget = max_seq_len - len(answer_ids)
    if budget < 16:
        raise ValueError(f"answer too long for max_seq_len={max_seq_len}")
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]
    return prompt_ids, answer_ids


@torch.no_grad()
def generate_text(model, tokenizer, prompt_ids: list[int], device: torch.device,
                  max_new_tokens: int = 48) -> str:
    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", None)
    if prev_cache is not None:
        model.config.use_cache = True
    try:
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    finally:
        if prev_cache is not None:
            model.config.use_cache = prev_cache
        if was_training:
            model.train()
    return text


def generate_risk(model, tokenizer, prompt_ids: list[int], device: torch.device,
                  max_new_tokens: int = 48):
    """Returns (raw_text, risk_or_None)."""
    text = generate_text(model, tokenizer, prompt_ids, device, max_new_tokens=max_new_tokens)
    return text, prompts.parse_risk(text)


@torch.no_grad()
def generate_batch_text(model, tokenizer, prompts_list: list[list[int]], device: torch.device,
                        max_new_tokens: int = 384, batch_size: int = 8, progress=None) -> list[str]:
    """Greedy batched generation with left padding; returns one decoded string per prompt."""
    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", None)
    if prev_cache is not None:
        model.config.use_cache = True
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    texts: list[str] = []
    try:
        for start in range(0, len(prompts_list), batch_size):
            chunk = prompts_list[start:start + batch_size]
            width = max(len(ids) for ids in chunk)
            input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, ids in enumerate(chunk):
                input_ids[i, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
                attn[i, width - len(ids):] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)
            out = model.generate(input_ids=input_ids, attention_mask=attn,
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad_id)
            texts.extend(tokenizer.batch_decode(out[:, width:], skip_special_tokens=True))
            if progress is not None:
                progress(min(start + batch_size, len(prompts_list)), len(prompts_list))
    finally:
        if prev_cache is not None:
            model.config.use_cache = prev_cache
        if was_training:
            model.train()
    return texts

