from __future__ import annotations

from typing import Any

from .llm_client import parse_json_object


def _has_cycle(nodes: list[str], edges: list[dict[str, str]]) -> bool:
    graph = {n: [] for n in nodes}
    for edge in edges:
        graph.setdefault(edge["from"], []).append(edge["to"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in graph.get(node, []):
            if dfs(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(dfs(n) for n in nodes)


def parse_plan(raw_text: str) -> dict[str, Any]:
    obj, error = parse_json_object(raw_text)
    if error:
        return {
            "raw_llm_output": raw_text,
            "parsed_plan": {"nodes": [], "edges": []},
            "parse_status": "failed",
            "parse_error": error,
            "dag_status": "unknown",
            "num_nodes": 0,
            "num_edges": 0,
            "tools_used": [],
            "terminal_tools": [],
        }
    nodes_in = obj.get("nodes", [])
    edges_in = obj.get("edges", [])
    nodes = []
    seen_ids = set()
    for i, node in enumerate(nodes_in if isinstance(nodes_in, list) else []):
        if not isinstance(node, dict):
            continue
        nid = str(node.get("id") or f"n{i+1}")
        if nid in seen_ids:
            nid = f"{nid}_{i+1}"
        seen_ids.add(nid)
        normalized = dict(node)
        normalized["id"] = nid
        nodes.append(normalized)
    node_ids = {n["id"] for n in nodes}
    edges = []
    edge_seen = set()
    for edge in edges_in if isinstance(edges_in, list) else []:
        if not isinstance(edge, dict):
            continue
        src, dst = str(edge.get("from", "")), str(edge.get("to", ""))
        if src in node_ids and dst in node_ids and src != dst and (src, dst) not in edge_seen:
            edges.append({"from": src, "to": dst, "support": str(edge.get("support", ""))})
            edge_seen.add((src, dst))
    cyclic = _has_cycle(list(node_ids), edges)
    tools = [str(n.get("tool")) for n in nodes if n.get("tool")]
    downstream = {e["from"] for e in edges}
    terminal = [str(n.get("tool")) for n in nodes if n.get("id") not in downstream and n.get("tool")]
    return {
        "raw_llm_output": raw_text,
        "parsed_plan": {"nodes": nodes, "edges": edges},
        "parse_status": "ok",
        "parse_error": None,
        "dag_status": "cyclic" if cyclic else "acyclic",
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "tools_used": tools,
        "terminal_tools": terminal,
    }
