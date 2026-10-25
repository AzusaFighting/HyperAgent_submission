from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .anchor_extraction import AnchorResult
from .schema_matching import EmbeddingIndex
from .tool_hypergraph import ToolHypergraph


@dataclass
class TaskToolSubgraph:
    schema_nodes: list[str]
    tool_nodes: list[str]
    reasons: dict[str, list[str]]
    backward_expansion_tools: list[str]
    forward_expansion_tools: list[str]
    app_filtered_tools: list[str]
    app_filtered_apps: list[str] = field(default_factory=list)

    def to_json(self, graph: ToolHypergraph) -> dict[str, Any]:
        edges = []
        for tool in self.tool_nodes:
            if tool in graph.tools:
                edges.append({"tool": tool, "inputs": graph.tools[tool].inputs, "outputs": graph.tools[tool].outputs})
        return {
            "schema_nodes": self.schema_nodes,
            "tool_nodes": self.tool_nodes,
            "tool_reasons": self.reasons,
            "backward_expansion_tools": self.backward_expansion_tools,
            "forward_expansion_tools": self.forward_expansion_tools,
            "app_filtered_tools": self.app_filtered_tools,
            "app_filtered_apps": self.app_filtered_apps,
            "edges": edges,
            "subgraph_stats": {
                "num_schema_nodes": len(self.schema_nodes),
                "num_tool_nodes": len(self.tool_nodes),
                "num_edges": len(edges),
            },
        }


def build_task_tool_subgraph(
    task: str,
    anchors: AnchorResult,
    graph: ToolHypergraph,
    tool_index: EmbeddingIndex,
    config: dict[str, Any],
) -> TaskToolSubgraph:
    cfg = config.get("subgraph", {})
    backward_depth = int(cfg.get("backward_depth", 2))
    producers_per_input = int(cfg.get("producers_per_input", 8))
    forward_top_k = int(cfg.get("forward_top_k", 20))
    expansion_mode = str(cfg.get("expansion_mode", "legacy"))
    app_filter_top_k_per_app = int(cfg.get("app_filter_top_k_per_app", 0))
    max_tools = int(cfg.get("max_tools", 80))
    if expansion_mode == "hybrid_topk":
        tool_scores = tool_index.hybrid_score_map(
            task,
            dense_weight=float(cfg.get("dense_weight", 0.7)),
            bm25_weight=float(cfg.get("bm25_weight", 0.3)),
        )
    elif expansion_mode == "semantic_topk":
        tool_scores = tool_index.score_map(task)
    else:
        tool_scores = {}
    reasons: dict[str, set[str]] = {}
    tools: set[str] = set()
    backward_tools: set[str] = set()
    forward_tools: set[str] = set()
    app_tools: set[str] = set()

    def add(tool: str, reason: str) -> None:
        if tool in graph.tools:
            tools.add(tool)
            reasons.setdefault(tool, set()).add(reason)

    for tool in anchors.final_target_tools:
        add(tool, "terminal_anchor")

    frontier = [tool for tool in anchors.final_target_tools if tool in graph.tools]
    seen = set(frontier)
    for _depth in range(backward_depth):
        next_frontier: list[str] = []
        for tool in frontier:
            for input_node in graph.tools[tool].inputs:
                producers = list(graph.producers_by_input.get(input_node, []))
                if expansion_mode in {"semantic_topk", "hybrid_topk"}:
                    producers.sort(key=lambda name: tool_scores.get(name, -1.0), reverse=True)
                for producer in producers[:producers_per_input]:
                    if producer not in seen:
                        next_frontier.append(producer)
                        seen.add(producer)
                    add(producer, "backward_expansion")
                    backward_tools.add(producer)
        frontier = next_frontier

    for hit in tool_index.query(task, forward_top_k):
        tool = hit["id"]
        add(tool, "forward_expansion")
        forward_tools.add(tool)

    for app in anchors.final_app_or_tool_anchors:
        app_candidates = [tool for tool, node in graph.tools.items() if node.app == app]
        if expansion_mode in {"semantic_topk", "hybrid_topk"} and app_filter_top_k_per_app > 0:
            app_candidates.sort(key=lambda name: tool_scores.get(name, -1.0), reverse=True)
            app_candidates = app_candidates[:app_filter_top_k_per_app]
        for tool in app_candidates:
            add(tool, "app_filter")
            app_tools.add(tool)

    if len(tools) > max_tools:
        ranked = [hit["id"] for hit in tool_index.query(task, max(max_tools * 2, forward_top_k))]
        keep = set(anchors.final_target_tools)
        for tool in ranked:
            if tool in tools:
                keep.add(tool)
            if len(keep) >= max_tools:
                break
        tools = {tool for tool in tools if tool in keep}

    schema_nodes: set[str] = set()
    for tool in tools:
        schema_nodes.update(graph.tools[tool].inputs)
        schema_nodes.update(graph.tools[tool].outputs)
    schema_nodes.update(x for x in anchors.final_schema_anchors if x in graph.schemas)

    return TaskToolSubgraph(
        schema_nodes=sorted(schema_nodes),
        tool_nodes=sorted(tools),
        reasons={k: sorted(v) for k, v in reasons.items() if k in tools},
        backward_expansion_tools=sorted(backward_tools & tools),
        forward_expansion_tools=sorted(forward_tools & tools),
        app_filtered_tools=sorted(app_tools & tools),
        app_filtered_apps=anchors.final_app_or_tool_anchors,
    )
