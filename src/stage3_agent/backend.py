from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .canonical import canonical_json, state_hash


def is_complete_json_object(raw: str) -> bool:
    try:
        value, end = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and not raw[end:].strip()


def assistant_prefill(prompt: str) -> str:
    marker = "\nCURRENT_TASK="
    if marker not in prompt:
        raise ValueError("assembled prompt is missing CURRENT_TASK divider")
    task_text = prompt.split(marker, 1)[1]
    try:
        task, _ = json.JSONDecoder().raw_decode(task_text)
        next_action = task["next_action"]
        required = next_action["required_type"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("assembled prompt has no valid next_action") from exc
    if required == "tool_call":
        name = next_action["allowed_names"][0]
        prefix = '{"tool_call":{"name":' + canonical_json(name) + ',"arguments":{'
        if name == "lookup_schedule":
            return prefix + '"course":"'
        if name == "lookup_policy":
            return prefix + '"topic":"'
        weekday = re.search(
            r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            str(task.get("input", "")),
            flags=re.IGNORECASE,
        )
        return prefix + ('"day":"' if weekday else '"query":"')
    if required == "memory_call":
        operation = next_action["allowed_operations"][0]
        if operation == "search":
            return '{"memory_call":{"operation":"search","query":'
        query = next_action["reuse_search_query"]
        return (
            '{"memory_call":{"operation":"invalidate","query":'
            + canonical_json(query)
            + ',"reason":'
        )
    if required == "final":
        return '{"final":{"answer":'
    raise ValueError(f"unsupported required action type: {required}")


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
            from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
        except ImportError as exc:
            raise RuntimeError("local_hf requires installation with the 'hf' extra") from exc
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.stopping_criteria_class = StoppingCriteria
        self.stopping_criteria_list_class = StoppingCriteriaList
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype="auto", device_map="auto"
        )

    def generate(self, prompt: str) -> str:
        divider = "\nCURRENT_TASK="
        if divider not in prompt:
            raise ValueError("assembled prompt is missing CURRENT_TASK divider")
        system_content, task_content = prompt.split(divider, 1)
        prefill = assistant_prefill(prompt)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "CURRENT_TASK=" + task_content},
            {"role": "assistant", "content": prefill},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        tokenizer = self.tokenizer
        prompt_tokens = inputs["input_ids"].shape[1]
        prefix = prefill
        base_class = self.stopping_criteria_class

        class StopAfterCompleteJson(base_class):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                del scores, kwargs
                generated = input_ids[0, prompt_tokens:]
                continuation = tokenizer.decode(generated, skip_special_tokens=True)
                return is_complete_json_object(prefix + continuation)

        stopping = self.stopping_criteria_list_class([StopAfterCompleteJson()])
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, temperature=None, top_p=None,
                max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return prefill + continuation


def backend_metadata(backend: Any, model_dir: Path | None) -> dict[str, Any]:
    result = {"backend": backend.name}
    if model_dir is not None:
        result["model_dir"] = str(model_dir.resolve())
        result["local_files_only"] = True
        if hasattr(backend, "tokenizer"):
            result["tokenizer_class"] = type(backend.tokenizer).__name__
            result["chat_template_hash"] = state_hash(backend.tokenizer.chat_template or "")
            result["assistant_prefill"] = "state_derived_json_prefix_v3"
            result["generation_stop"] = "first_complete_json_object"
    return result
