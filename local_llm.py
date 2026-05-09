# local_llm.py — CAAMS Local LLM Inference (Apache 2.0)
#
# Loads Qwen2.5-1.5B-Instruct directly from HuggingFace transformers.
# NO Ollama server, NO third-party API, NO commercial dependencies.
# Model license: Apache 2.0  (Qwen/Qwen2.5-1.5B-Instruct on HuggingFace)
#
# Compatible drop-in for langchain_ollama.ChatOllama:
#   llm.invoke([SystemMessage(...), HumanMessage(...)]) -> LLMResponse(content=str)
#
# Usage:
#   from local_llm import get_local_llm
#   llm = get_local_llm()            # lazy-loaded singleton
#   resp = llm.invoke([...messages])
#   print(resp.content)              # JSON string
#
# License: Apache 2.0

from __future__ import annotations

import os
import json
import time
import torch
from dataclasses import dataclass
from typing import Optional

MODELS_DIR = "./models"
MODEL_HF_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # Apache 2.0
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct")


# ─────────────────────────────────────────────────────────────────────────────
# Response wrapper — mirrors langchain Message.content interface
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    content: str


# ─────────────────────────────────────────────────────────────────────────────
# Local Qwen2.5-1.5B-Instruct
# ─────────────────────────────────────────────────────────────────────────────
class LocalQwenLLM:
    """
    Wraps Qwen2.5-1.5B-Instruct (Apache 2.0) for local CPU/GPU inference.
    No server, no Ollama, no API key.
    Auto-downloads weights from HuggingFace on first run (~3 GB).
    Subsequent runs use local HuggingFace cache.

    Optimizations for edge / laptop:
    - CPU: float32, greedy decode (deterministic, fast)
    - GPU: float16, device_map=auto
    - max_new_tokens capped at 200 to keep latency reasonable on CPU
    """

    def __init__(self, max_new_tokens: int = 200):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else MODEL_HF_ID
        print(f"[LocalQwen] Loading {model_id}  (Apache 2.0, ~3 GB first download)")
        print(f"[LocalQwen] This may take 2-5 min on first run; cached afterwards.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print(f"[LocalQwen] Ready on {self.device.upper()} | dtype={dtype}")

    # ── LangChain-compatible invoke ───────────────────────────────────────────
    def invoke(self, messages: list) -> LLMResponse:
        """
        Accepts LangChain SystemMessage/HumanMessage objects OR plain dicts.
        Always returns LLMResponse(content=str) — JSON string expected by callers.
        """
        msg_dicts = []
        for m in messages:
            if hasattr(m, "type"):                         # LangChain objects
                role = "system" if m.type == "system" else "user"
                msg_dicts.append({"role": role, "content": m.content})
            elif hasattr(m, "role"):                       # plain dict-like
                msg_dicts.append({"role": m.role, "content": m.content})
            else:
                msg_dicts.append(m)

        text = self.tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = round((time.perf_counter() - t0) * 1000, 0)

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response  = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"[LocalQwen] Generated in {elapsed:.0f}ms | tokens={len(generated)}")
        return LLMResponse(content=response)

    # ── bind() stub — used by orchestrator.py: llm.bind(num_predict=N) ───────
    def bind(self, **kwargs) -> "LocalQwenLLM":
        if "num_predict" in kwargs:
            self.max_new_tokens = int(kwargs["num_predict"])
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — shared across all agents in the process
# ─────────────────────────────────────────────────────────────────────────────
_singleton: Optional[LocalQwenLLM] = None


def get_local_llm() -> Optional[LocalQwenLLM]:
    """
    Lazy-loads the Qwen model on first call; returns the same instance after.
    Returns None gracefully if transformers is not installed or load fails.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    try:
        _singleton = LocalQwenLLM(max_new_tokens=200)
        return _singleton
    except Exception as e:
        print(f"[LocalQwen] Load failed → deterministic fallback active. Error: {e}")
        return None


def parse_json_response(resp: LLMResponse) -> dict:
    """
    Safely parses JSON from model output.
    Strips markdown fences if model wraps in ```json ... ```.
    Returns {} on parse failure so callers can detect and use fallback.
    """
    if resp is None:
        return {}
    raw = resp.content.strip()
    # Strip markdown code fences
    if "```" in raw:
        parts = raw.split("```")
        raw   = parts[1] if len(parts) >= 2 else raw
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    try:
        return json.loads(raw.strip())
    except Exception:
        # Try finding a JSON object inside the text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except Exception:
                pass
        return {}