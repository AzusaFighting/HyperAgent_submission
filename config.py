from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "appworld_root": "PATH/TO/APPWORLD",
    "hypergraph_path": "PATH/TO/schema_hypergraph.json",
    "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_base": "https://api.deepseek.com",
        "timeout": 90,
        "temperature": 0,
    },
    "retrieval": {"schema_anchor_top_k": 12, "tool_anchor_top_k": 12, "raw_tool_doc_top_k": 40},
    "subgraph": {
        "backward_depth": 2,
        "forward_depth": 1,
        "producers_per_input": 8,
        "forward_top_k": 20,
        "app_neighbor_depth": 1,
        "max_tools": 80,
    },
    "planning": {"max_context_tools": 60, "max_raw_tools": 40},
    "judge": {"enabled": True},
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return DEFAULT_CONFIG
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text) or {}
    except Exception:
        loaded = json.loads(text)
    return deep_update(DEFAULT_CONFIG, loaded)


def write_config(config: dict[str, Any], path: Path) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
