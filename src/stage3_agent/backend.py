from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json


class ScriptedBackend:
    name = "scripted"

    def __init__(self, task: dict[str, Any], arm: str):
        plan = list(task["script_plan"])
        if arm == "no_tools":
            plan = [action for action in plan if "tool_call" not in action]
        if arm == "no_memory":
            plan = [action for action in plan if "memory_call" not in action]
        succeeds = arm != "no_tools" and not (arm == "no_memory" and task["task_type"] == "cross_session_memory")
        if succeeds:
            final = {
                "answer": task["target_fields"],
                "evidence_ids": task["required_evidence_ids"],
                "text": "Synthetic campus result grounded in the cited fixture records.",
            }
        else:
            final = {"answer": {"unknown": True}, "evidence_ids": [], "text": "I do not know without the disabled capability."}
        plan.append({"final": final})
        self.responses = [canonical_json(action) for action in plan]
        self.position = 0

    def generate(self, prompt: str) -> str:
        del prompt
        if self.position >= len(self.responses):
            return canonical_json({"final": {"answer": {"unknown": True}, "evidence_ids": [], "text": "No scripted action remains."}})
        response = self.responses[self.position]
        self.position += 1
        return response


class LocalHFBackend:
    name = "local_hf"

    def __init__(self, model_dir: Path, seed: int, max_new_tokens: int):
        if not model_dir.is_dir():
            raise FileNotFoundError(f"local model directory does not exist: {model_dir}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("local_hf requires installation with the 'hf' extra") from exc
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype="auto", device_map="auto"
        )

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, temperature=None, top_p=None, max_new_tokens=self.max_new_tokens
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def backend_metadata(backend: Any, model_dir: Path | None) -> dict[str, Any]:
    result = {"backend": backend.name}
    if model_dir is not None:
        result["model_dir"] = str(model_dir.resolve())
        result["local_files_only"] = True
    return result
