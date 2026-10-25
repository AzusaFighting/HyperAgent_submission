from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SchemaNode:
    id: str
    kind: str
    api: str
    name: str
    description: str
    text: str


@dataclass
class ToolNode:
    id: str
    name: str
    app: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


class ToolHypergraph:
    def __init__(self, path: Path):
        self.path = path
        self.raw = json.loads(path.read_text(encoding="utf-8"))
        self.nodes_by_id: dict[str, dict[str, Any]] = {node["id"]: node for node in self.raw["nodes"]}
        self.tools: dict[str, ToolNode] = {}
        self.schemas: dict[str, SchemaNode] = {}
        self.producers_by_input: dict[str, list[str]] = {}
        self.consumers_by_output: dict[str, list[str]] = {}
        self.tool_reason: dict[str, set[str]] = {}
        self._load_nodes()
        self._load_edges()

    def _schema_text(self, node: dict[str, Any]) -> tuple[str, str]:
        if node.get("kind") == "effect":
            effect = node.get("effect", {})
            name = effect.get("name", node["id"])
            desc = effect.get("description", "")
            return name, f"Effect: {name}. {desc}"
        schema = node.get("schema", {})
        name = schema.get("name", node["id"].split(":")[-1])
        desc = schema.get("description", "")
        return name, f"{node.get('kind')}: {node.get('api')}.{name}. {desc}"

    def _load_nodes(self) -> None:
        for node in self.raw["nodes"]:
            if node.get("kind") == "tool":
                name = node["name"]
                self.tools[name] = ToolNode(
                    id=node["id"],
                    name=name,
                    app=name.split(".", 1)[0],
                    description=node.get("description", ""),
                )
            elif node.get("kind") in {"input_schema", "output_schema", "effect"}:
                name, text = self._schema_text(node)
                self.schemas[node["id"]] = SchemaNode(
                    id=node["id"],
                    kind=node.get("kind", ""),
                    api=node.get("api", ""),
                    name=name,
                    description=text,
                    text=text,
                )

    def _load_edges(self) -> None:
        for edge in self.raw["hyperedges"]:
            if edge.get("kind") == "tool_invocation":
                tool_name = edge["via"].removeprefix("tool:")
                if tool_name in self.tools:
                    self.tools[tool_name].inputs = list(edge.get("tail", []))
                    self.tools[tool_name].outputs = list(edge.get("head", []))
            elif edge.get("kind") == "schema_dependency":
                tail = edge.get("tail", [])
                head = edge.get("head", [])
                if tail and head:
                    out_node, in_node = tail[0], head[0]
                    producer = edge.get("source_api")
                    consumer = edge.get("target_api")
                    if producer:
                        self.producers_by_input.setdefault(in_node, []).append(producer)
                    if consumer:
                        self.consumers_by_output.setdefault(out_node, []).append(consumer)

    def tool_text(self, tool_name: str) -> str:
        tool = self.tools[tool_name]
        ins = [self.schemas[x].text for x in tool.inputs if x in self.schemas]
        outs = [self.schemas[x].text for x in tool.outputs if x in self.schemas]
        return f"Tool: {tool.name}\nDescription: {tool.description}\nInputs:\n" + "\n".join(ins) + "\nOutputs/effects:\n" + "\n".join(outs)

    def candidate_apps(self) -> set[str]:
        return {tool.app for tool in self.tools.values()}

    def add_reason(self, reasons: dict[str, set[str]], tool: str, reason: str) -> None:
        if tool in self.tools:
            reasons.setdefault(tool, set()).add(reason)
