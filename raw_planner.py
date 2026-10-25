from __future__ import annotations

from .appworld_loader import ToolDoc
from .dag_planner import DAG_SCHEMA
from .llm_client import DeepSeekClient
from .plan_parser import parse_plan


def format_raw_tool_docs(tool_docs: list[ToolDoc]) -> str:
    return "\n\n".join(
        f"- {tool.name}: {tool.description}\n  inputs: {[p['name'] for p in tool.parameters]}" for tool in tool_docs
    )


def generate_raw_llm_plan(task: str, tool_docs: list[ToolDoc], llm_client: DeepSeekClient) -> dict:
    system = "You generate complete acyclic DAG tool plans from raw tool documentation. Return JSON only."
    user = (
        f"Task:\n{task}\n\n"
        "Available tool docs retrieved by task/document similarity. Do not use graph anchors or graph expansion.\n"
        f"{format_raw_tool_docs(tool_docs)}\n\n"
        f"{DAG_SCHEMA}"
    )
    response = llm_client.complete_json(system, user)
    return parse_plan(response.text)
