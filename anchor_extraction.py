from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .appworld_loader import ToolDoc
from .llm_client import DeepSeekClient, parse_json_object
from .schema_matching import EmbeddingIndex
from .tool_hypergraph import ToolHypergraph


@dataclass
class AnchorResult:
    raw_llm_output: str
    parsed_schema_anchors: list[str]
    parsed_goal_effects: list[str]
    parsed_app_or_tool_anchors: list[str]
    embedding_schema_anchors: list[dict[str, Any]]
    embedding_tool_anchors: list[dict[str, Any]]
    final_schema_anchors: list[str]
    final_target_tools: list[str]
    final_app_or_tool_anchors: list[str]
    parse_error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return self.__dict__


def _tool_catalog_snippet(tool_docs: list[ToolDoc], limit: int = 80) -> str:
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tool_docs[:limit])


def extract_anchors_with_llm(
    task: str,
    tool_docs: list[ToolDoc],
    graph: ToolHypergraph,
    llm_client: DeepSeekClient,
) -> tuple[list[str], list[str], list[str], str, str | None]:
    apps = sorted(graph.candidate_apps())
    system = "You extract task anchors for tool planning. Return strict JSON only. Do not create a plan."
    user = (
        "Task:\n"
        f"{task}\n\n"
        "Available apps:\n"
        f"{', '.join(apps)}\n\n"
        "Tool catalog sample:\n"
        f"{_tool_catalog_snippet(tool_docs)}\n\n"
        "Return JSON exactly with keys: goal_effects, mentioned_schemas, candidate_apps_or_tools, rationale.\n"
        "goal_effects should contain candidate terminal tools or effects like gmail.send_email or phone.message_sent.\n"
        "mentioned_schemas should contain schema slot names or concrete schema ids if clear.\n"
        "candidate_apps_or_tools should contain app names or tool names."
    )
    response = llm_client.complete_json(system, user)
    parsed, error = parse_json_object(response.text)
    return (
        [str(x).replace("__", ".") for x in parsed.get("mentioned_schemas", []) if isinstance(x, str)],
        [str(x).replace("__", ".") for x in parsed.get("goal_effects", []) if isinstance(x, str)],
        [str(x).replace("__", ".") for x in parsed.get("candidate_apps_or_tools", []) if isinstance(x, str)],
        response.text,
        error,
    )


def retrieve_schema_anchors_by_embedding(task: str, schema_index: EmbeddingIndex, top_k: int) -> list[dict[str, Any]]:
    return schema_index.query(task, top_k)


def retrieve_tool_anchors_by_embedding(task: str, tool_index: EmbeddingIndex, top_k: int) -> list[dict[str, Any]]:
    return tool_index.query(task, top_k)


def build_anchor_result(
    task: str,
    tool_docs: list[ToolDoc],
    graph: ToolHypergraph,
    llm_client: DeepSeekClient,
    schema_index: EmbeddingIndex,
    tool_index: EmbeddingIndex,
    schema_top_k: int,
    tool_top_k: int,
) -> AnchorResult:
    parsed_schema, parsed_effects, parsed_apps, raw, error = extract_anchors_with_llm(task, tool_docs, graph, llm_client)
    emb_schema = retrieve_schema_anchors_by_embedding(task, schema_index, schema_top_k)
    emb_tool = retrieve_tool_anchors_by_embedding(task, tool_index, tool_top_k)

    schema_ids = set(graph.schemas)
    tool_names = set(graph.tools)
    app_names = graph.candidate_apps()

    final_schema: list[str] = []
    for x in parsed_schema:
        if x in schema_ids and x not in final_schema:
            final_schema.append(x)
    for x in emb_schema:
        if x["id"] not in final_schema:
            final_schema.append(x["id"])

    final_tools: list[str] = []
    for x in parsed_effects:
        if x in tool_names and x not in final_tools:
            final_tools.append(x)
        else:
            for tool in tool_names:
                if x == tool or x.endswith(tool) or x in tool:
                    if tool not in final_tools:
                        final_tools.append(tool)
    for x in emb_tool:
        if x["id"] in tool_names and x["id"] not in final_tools:
            final_tools.append(x["id"])

    final_apps: list[str] = []
    for x in parsed_apps:
        candidate = x.split(".", 1)[0]
        if candidate in app_names and candidate not in final_apps:
            final_apps.append(candidate)
        if x in tool_names and x not in final_tools:
            final_tools.append(x)

    return AnchorResult(
        raw_llm_output=raw,
        parsed_schema_anchors=parsed_schema,
        parsed_goal_effects=parsed_effects,
        parsed_app_or_tool_anchors=parsed_apps,
        embedding_schema_anchors=emb_schema,
        embedding_tool_anchors=emb_tool,
        final_schema_anchors=final_schema,
        final_target_tools=final_tools,
        final_app_or_tool_anchors=final_apps,
        parse_error=error,
    )
