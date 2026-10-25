from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def aggregate_metrics_md(metrics: dict[str, Any]) -> str:
    raw = metrics["methods"]["raw"]
    anchored = metrics["methods"]["anchored"]
    pair = metrics["pairwise"]
    def row(name: str, m: dict[str, Any]) -> str:
        return (
            f"| {name} | {m['parse_rate']:.3f} | {m['dag_acyclic_rate']:.3f} | "
            f"{m['task_relevance']:.2f} | {m['tool_appropriateness']:.2f} | {m['dependency_coherence']:.2f} | "
            f"{m['plan_completeness']:.2f} | {m['execution_plausibility']:.2f} | {m['minimality']:.2f} | "
            f"{m['terminal_effect_correctness']:.2f} | {m['dag_quality']:.2f} | {m['overall_score']:.2f} |"
        )
    def diag(name: str, m: dict[str, Any]) -> str:
        return (
            f"| {name} | {m['avg_num_nodes']:.2f} | {m['avg_num_edges']:.2f} | "
            f"{m['dag_acyclic_rate']:.3f} | {m['unsupported_edge_rate_light']:.3f} | "
            f"{m['tool_in_subgraph_rate']:.3f} | {m['terminal_tool_in_anchors_rate']:.3f} |"
        )
    return "\n".join(
        [
            "# Task-to-Tool Anchoring Aggregate Metrics",
            "",
            "## Plan quality by method",
            "",
            "| Method | Parse Rate | DAG Rate | Task Rel. | Tool Approp. | Dep. Coherence | Completeness | Exec. Plaus. | Minimality | Terminal Correct | DAG Quality | Overall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            row("Raw LLM Plan", raw),
            row("Anchored DAG Plan", anchored),
            "",
            "## Pairwise judge result",
            "",
            "| Anchored Win | Raw Win | Tie |",
            "|---:|---:|---:|",
            f"| {pair['anchored_win_rate']:.3f} | {pair['raw_win_rate']:.3f} | {pair['tie_rate']:.3f} |",
            "",
            "## Structural diagnostics",
            "",
            "| Method | Avg Nodes | Avg Edges | DAG Acyclic Rate | Unsupported Edge Rate Light | Tool In Subgraph | Terminal Tool In Anchors |",
            "|---|---:|---:|---:|---:|---:|---:|",
            diag("Raw LLM Plan", raw),
            diag("Anchored DAG Plan", anchored),
        ]
    )


def failure_analysis_md(metrics: dict[str, Any]) -> str:
    lines = ["# Failure Analysis", "", "| Failure mode | Count |", "|---|---:|"]
    for mode, count in sorted(metrics.get("failure_modes", {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {mode} | {count} |")
    return "\n".join(lines)


def sample_cases_md(records: list[dict[str, Any]], limit_each: int = 5) -> str:
    buckets = {"anchored": [], "raw": [], "tie": [], "failure": []}
    for record in records:
        pref = record["llm_as_plan_judge"].get("pairwise_preference", "tie")
        if pref in buckets:
            buckets[pref].append(record)
        if record["anchored_dag_plan"].get("parse_status") != "ok" or record["raw_llm_plan"].get("parse_status") != "ok":
            buckets["failure"].append(record)
    lines = ["# Sample Cases"]
    for bucket, title in [("anchored", "Anchored Wins"), ("raw", "Raw Wins"), ("tie", "Ties"), ("failure", "Parse/Cyclic Failures")]:
        lines += ["", f"## {title}"]
        for record in buckets[bucket][:limit_each]:
            lines += [
                "",
                f"### {record['task_id']}",
                "",
                f"Task: {record['task_instruction']}",
                "",
                f"Anchors: `{json.dumps(record['anchor_extraction']['final_target_tools'][:10], ensure_ascii=False)}`",
                "",
                f"Subgraph tools: {len(record['task_tool_subgraph']['tool_nodes'])}",
                "",
                f"Raw tools: `{record['raw_llm_plan'].get('tools_used', [])}`",
                "",
                f"Anchored tools: `{record['anchored_dag_plan'].get('tools_used', [])}`",
                "",
                f"Preference reason: {record['llm_as_plan_judge'].get('preference_reason', '')}",
            ]
    return "\n".join(lines)
