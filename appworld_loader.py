from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    instruction: str


@dataclass(frozen=True)
class ToolDoc:
    name: str
    app: str
    description: str
    parameters: list[dict[str, Any]]
    text: str


def canonical_tool_name(name: str) -> str:
    return name.replace("__", ".")


def load_tasks(appworld_root: Path, split: str, max_tasks: int | None = None, task_ids: list[str] | None = None) -> list[TaskRecord]:
    if task_ids:
        ids = task_ids
    else:
        split_path = appworld_root / "data" / "datasets" / f"{split}.txt"
        ids = [x.strip() for x in split_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    tasks: list[TaskRecord] = []
    for task_id in ids:
        specs_path = appworld_root / "data" / "tasks" / task_id / "specs.json"
        if not specs_path.exists():
            continue
        specs = json.loads(specs_path.read_text(encoding="utf-8"))
        tasks.append(TaskRecord(task_id=task_id, instruction=specs.get("instruction", "")))
        if max_tasks and len(tasks) >= max_tasks:
            break
    return tasks


def load_tool_docs(appworld_root: Path, include_supervisor: bool = False) -> list[ToolDoc]:
    docs: dict[str, ToolDoc] = {}
    for path in sorted((appworld_root / "data" / "api_docs" / "function_calling").glob("*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            continue
        for item in items:
            fn = item.get("function", {})
            name = canonical_tool_name(fn.get("name", ""))
            if not name or (not include_supervisor and name.startswith("supervisor.")):
                continue
            app = name.split(".", 1)[0]
            params = []
            for pname, spec in sorted(fn.get("parameters", {}).get("properties", {}).items()):
                params.append(
                    {
                        "name": pname,
                        "type": spec.get("type", ""),
                        "description": spec.get("description", ""),
                    }
                )
            param_text = "\n".join(f"- {p['name']} ({p['type']}): {p['description']}" for p in params)
            description = fn.get("description", "")
            text = f"Tool: {name}\nApp: {app}\nDescription: {description}\nInputs:\n{param_text}"
            docs.setdefault(name, ToolDoc(name=name, app=app, description=description, parameters=params, text=text))
    return list(docs.values())

