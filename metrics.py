from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .judge import SCORE_KEYS
from .tool_hypergraph import ToolHypergraph


def tool_in_subgraph_rate(plan: dict[str, Any], subgraph_tools: set[str]) -> float:
    tools = plan.get("tools_used", [])
    if not tools:
        return 0.0
    return sum(1 for t in tools if t in subgraph_tools) / len(tools)


def terminal_tool_coverage(plan: dict[str, Any], anchors: set[str]) -> float:
    terminals = plan.get("terminal_tools", [])
    if not terminals:
        return 0.0
    return sum(1 for t in terminals if t in anchors) / len(terminals)


def unsupported_edge_rate_light(plan: dict[str, Any], graph: ToolHypergraph) -> float:
    nodes = {n["id"]: n for n in plan.get("parsed_plan", {}).get("nodes", [])}
    edges = plan.get("parsed_plan", {}).get("edges", [])
    if not edges:
        return 0.0
    unsupported = 0
    for edge in edges:
        src_tool = nodes.get(edge["from"], {}).get("tool")
        dst_tool = nodes.get(edge["to"], {}).get("tool")
        if src_tool not in graph.tools or dst_tool not in graph.tools:
            unsupported += 1
            continue
        src_outputs = set(graph.tools[src_tool].outputs)
        dst_inputs = set(graph.tools[dst_tool].inputs)
        supported = any(dst in graph.producers_by_input and src_tool in graph.producers_by_input[dst] for dst in dst_inputs)
        supported = supported or bool(src_outputs & dst_inputs)
        if not supported:
            unsupported += 1
    return unsupported / len(edges)


def aggregate(records: list[dict[str, Any]], graph: ToolHypergraph) -> dict[str, Any]:
    methods = {"raw": "raw_llm_plan", "anchored": "anchored_dag_plan"}
    out: dict[str, Any] = {"methods": {}, "pairwise": {}, "failure_modes": {}}
    for method, key in methods.items():
        plans = [r[key] for r in records]
        judges = [r["llm_as_plan_judge"][f"{method}_plan_scores"] for r in records]
        out["methods"][method] = {
            "parse_rate": float(np.mean([p.get("parse_status") == "ok" for p in plans])) if plans else 0.0,
            "dag_acyclic_rate": float(np.mean([p.get("dag_status") == "acyclic" for p in plans])) if plans else 0.0,
            "avg_num_nodes": float(np.mean([p.get("num_nodes", 0) for p in plans])) if plans else 0.0,
            "avg_num_edges": float(np.mean([p.get("num_edges", 0) for p in plans])) if plans else 0.0,
            "unsupported_edge_rate_light": float(np.mean([unsupported_edge_rate_light(p, graph) for p in plans])) if plans else 0.0,
            "tool_in_subgraph_rate": float(
                np.mean(
                    [
                        tool_in_subgraph_rate(p, set(r["task_tool_subgraph"].get("tool_nodes", [])))
                        for p, r in zip(plans, records)
                    ]
                )
            )
            if plans
            else 0.0,
            "terminal_tool_in_anchors_rate": float(
                np.mean(
                    [
                        terminal_tool_coverage(p, set(r["anchor_extraction"].get("final_target_tools", [])))
                        for p, r in zip(plans, records)
                    ]
                )
            )
            if plans
            else 0.0,
        }
        for score_key in SCORE_KEYS:
            out["methods"][method][score_key] = float(np.mean([j.get(score_key, 1) for j in judges])) if judges else 0.0
        out["methods"][method]["overall_score"] = float(np.mean([out["methods"][method][k] for k in SCORE_KEYS]))
    prefs = Counter(r["llm_as_plan_judge"].get("pairwise_preference", "tie") for r in records)
    total = max(len(records), 1)
    out["pairwise"] = {
        "anchored_win_rate": prefs["anchored"] / total,
        "raw_win_rate": prefs["raw"] / total,
        "tie_rate": prefs["tie"] / total,
        "counts": dict(prefs),
    }
    modes = Counter()
    for r in records:
        modes.update(r["llm_as_plan_judge"].get("major_failure_modes_raw", []))
        modes.update(r["llm_as_plan_judge"].get("major_failure_modes_anchored", []))
    out["failure_modes"] = dict(modes)
    return out
