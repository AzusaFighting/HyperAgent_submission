from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]


class DeepSeekClient:
    def __init__(self, config: dict[str, Any], dry_run: bool = False):
        self.model = config.get("model", "deepseek-chat")
        self.api_base = config.get("api_base", "https://api.deepseek.com").rstrip("/")
        self.timeout = int(config.get("timeout", 90))
        self.temperature = float(config.get("temperature", 0))
        self.dry_run = dry_run
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key and not dry_run:
            raise RuntimeError("DEEPSEEK_API_KEY is required unless --dry-run is used.")

    def complete_json(self, system: str, user: str) -> LLMResponse:
        if self.dry_run:
            text = json.dumps({"goal_effects": [], "mentioned_schemas": [], "candidate_apps_or_tools": [], "rationale": "dry-run"})
            return LLMResponse(text=text, raw={"dry_run": True})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                raw = response.json()
                return LLMResponse(text=raw["choices"][0]["message"]["content"], raw=raw)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error


def parse_json_object(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}, None
    except Exception as exc:  # noqa: BLE001
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}, str(exc)
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}, None
        except Exception as exc2:  # noqa: BLE001
            return {}, str(exc2)
