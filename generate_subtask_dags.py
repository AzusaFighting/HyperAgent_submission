from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .llm_client import DeepSeekClient, parse_json_object


SCORE_KEYS = [
    "task_decomposition_quality",
    "dependency_correctness",
    "coverage_of_initial_dag",
    "execution_plausibility",
    "minimality",
    "agent_blueprint_usefulness",
]


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        plan = row.get("anchored_dag_plan", {}).get("parsed_plan")
        if plan and row.get("task_instruction"):
            records.append(row)
        if len(records) >= limit:
            break
    return records


def generate_subtask_dag(task: str, initial_dag: dict[str, Any], llm: DeepSeekClient) -> dict[str, Any]:
    system = "You convert tool-level DAG blueprints into task-level subtask DAGs. Return strict JSON only."
    user = (
        "Given a user task t and its initial tool-level DAG, decompose t into a small set of executable subtasks "
        "{t1, t2, ..., tm}. Each subtask should be a semantic execution unit, not merely a tool name. "
        "Preserve important prerequisites, lookup/detail steps, filtering/ranking steps, and final response/action steps.\n\n"
        f"Task t:\n{task}\n\n"
        f"Initial tool-level DAG:\n{json.dumps(initial_dag, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with keys:\n"
        "- subtasks: list of objects with id, title, objective, grounded_tools, expected_inputs, expected_outputs, success_criteria\n"
        "- dependencies: list of objects with from, to, reason; direction means from must happen before to\n"
        "- execution_order_hint: list of subtask ids in one valid topological order\n"
        "- assumptions: list of important assumptions or uncertain points\n"
        "Use 3-8 subtasks. Keep dependencies acyclic."
    )
    response = llm.complete_json(system, user)
    parsed, error = parse_json_object(response.text)
    return {"raw_llm_output": response.text, "parsed": parsed, "parse_error": error}


def judge_subtask_dag(task: str, initial_dag: dict[str, Any], subtask_dag: dict[str, Any], llm: DeepSeekClient) -> dict[str, Any]:
    system = "You are a strict judge of task-level DAG plans for tool-use agents. Return strict JSON only."
    user = (
        f"Task:\n{task}\n\n"
        f"Initial tool-level DAG:\n{json.dumps(initial_dag, ensure_ascii=False, indent=2)}\n\n"
        f"Subtask DAG:\n{json.dumps(subtask_dag, ensure_ascii=False, indent=2)}\n\n"
        "Score the subtask DAG from 1-5 on: "
        + ", ".join(SCORE_KEYS)
        + ". Judge whether the plan can serve as a reliable blueprint for a ReAct-style agent. "
        "Penalize missing prerequisite/detail steps, unsupported dependencies, vague subtasks, over-compression, and non-executable final steps. "
        "Return JSON with keys scores, overall_score, verdict, strengths, weaknesses, suggested_fix. "
        "overall_score should be a single number from 1 to 5."
    )
    response = llm.complete_json(system, user)
    parsed, error = parse_json_object(response.text)
    scores = parsed.get("scores", {}) if isinstance(parsed.get("scores"), dict) else {}
    normalized_scores = {}
    for key in SCORE_KEYS:
        try:
            normalized_scores[key] = max(1, min(5, int(scores.get(key, 1))))
        except Exception:
            normalized_scores[key] = 1
    try:
        overall = float(parsed.get("overall_score", mean(normalized_scores.values())))
    except Exception:
        overall = mean(normalized_scores.values())
    return {
        "raw_llm_output": response.text,
        "parse_error": error,
        "scores": normalized_scores,
        "overall_score": round(max(1.0, min(5.0, overall)), 2),
        "verdict": parsed.get("verdict", ""),
        "strengths": parsed.get("strengths", []),
        "weaknesses": parsed.get("weaknesses", []),
        "suggested_fix": parsed.get("suggested_fix", ""),
    }


def markdown_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Subtask DAG Dev10",
        "",
        "| # | Task ID | Overall | Verdict |",
        "|---:|---|---:|---|",
    ]
    for idx, row in enumerate(rows, 1):
        judge = row["llm_as_judge"]
        lines.append(f"| {idx} | `{row['task_id']}` | {judge['overall_score']:.2f} | {str(judge.get('verdict', '')).replace('|', '/')} |")
    lines.extend(["", f"Average overall score: {mean(row['llm_as_judge']['overall_score'] for row in rows):.2f}/5", ""])
    for idx, row in enumerate(rows, 1):
        plan = row["subtask_dag"]["parsed"]
        judge = row["llm_as_judge"]
        lines.extend([
            f"## {idx}. {row['task_id']}",
            "",
            f"Task: {row['task_instruction']}",
            "",
            f"Overall judge score: {judge['overall_score']:.2f}/5",
            "",
            "Subtasks:",
        ])
        for subtask in plan.get("subtasks", []):
            tools = ", ".join(subtask.get("grounded_tools", []) or [])
            lines.append(f"- `{subtask.get('id')}` {subtask.get('title')}: {subtask.get('objective')} Tools: {tools}")
        lines.append("")
        lines.append("Dependencies:")
        for edge in plan.get("dependencies", []):
            lines.append(f"- `{edge.get('from')}` -> `{edge.get('to')}`: {edge.get('reason')}")
        lines.extend(["", f"Judge verdict: {judge.get('verdict', '')}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtask DAGs from initial anchored DAGs and judge them.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/task_tool_anchoring_hybrid_topk.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    llm = DeepSeekClient(config.get("llm", {}), dry_run=args.dry_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in load_records(args.records, args.limit):
        initial_dag = record["anchored_dag_plan"]["parsed_plan"]
        subtask_dag = generate_subtask_dag(record["task_instruction"], initial_dag, llm)
        judge = judge_subtask_dag(record["task_instruction"], initial_dag, subtask_dag["parsed"], llm)
        rows.append(
            {
                "task_id": record["task_id"],
                "task_instruction": record["task_instruction"],
                "initial_dag": initial_dag,
                "subtask_dag": subtask_dag,
                "llm_as_judge": judge,
            }
        )
        (args.output_dir / "subtask_dag_records.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )

    summary = {
        "num_tasks": len(rows),
        "average_overall_score": round(mean(row["llm_as_judge"]["overall_score"] for row in rows), 3) if rows else 0,
        "score_keys": SCORE_KEYS,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "subtask_dag_report.md").write_text(markdown_report(rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
