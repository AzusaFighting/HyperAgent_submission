from __future__ import annotations

from .anchor_extraction import AnchorResult
from .subgraph_builder import TaskToolSubgraph
from .tool_hypergraph import ToolHypergraph


def _schema_label(graph: ToolHypergraph, sid: str) -> str:
    node = graph.schemas.get(sid)
    return f"{sid} ({node.text})" if node else sid


def format_tool_block(graph: ToolHypergraph, tool: str, reason: str | None = None) -> str:
    t = graph.tools[tool]
    inputs = "\n".join(f"    - {_schema_label(graph, x)}" for x in t.inputs)
    outputs = "\n".join(f"    - {_schema_label(graph, x)}" for x in t.outputs)
    reason_text = f"\n  selected_by: {reason}" if reason else ""
    return f"- {tool}: {t.description}{reason_text}\n  required_inputs:\n{inputs}\n  outputs_or_effects:\n{outputs}"


def format_plan_context(
    task: str,
    anchors: AnchorResult,
    subgraph: TaskToolSubgraph,
    graph: ToolHypergraph,
    max_tools: int = 60,
) -> str:
    terminal = [tool for tool in anchors.final_target_tools if tool in subgraph.tool_nodes][:15]
    backward = subgraph.backward_expansion_tools[:max_tools]
    forward = subgraph.forward_expansion_tools[:max_tools]
    blocked = []
    for tool in terminal:
        missing = graph.tools[tool].inputs
        blocked.append(f"- {tool}: missing_or_required_inputs={missing}")
    tool_order = terminal + [t for t in backward + forward + subgraph.app_filtered_tools if t not in terminal]
    tool_order = [t for i, t in enumerate(tool_order) if t in subgraph.tool_nodes and t not in tool_order[:i]][:max_tools]
    blocks = [format_tool_block(graph, tool, ", ".join(subgraph.reasons.get(tool, []))) for tool in tool_order]
    return (
        f"Task:\n{task}\n\n"
        "Current grounded schemas:\n- Initial state is the task text and extractable literal values only.\n\n"
        "Candidate terminal tools:\n"
        + "\n".join(terminal)
        + "\n\n"
        "Task-conditioned tool subgraph:\n"
        + "\n\n".join(blocks)
        + "\n\nKnown blocked candidate terminal tools:\n"
        + "\n".join(blocked)
    )

