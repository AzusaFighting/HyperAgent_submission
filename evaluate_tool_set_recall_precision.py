from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.appworld_embedding_quality_eval import load_tasks as load_gold_tasks
from src.appworld_embedding_quality_eval import load_tool_docs, normalize_rows


def extract_dag_tools(plan: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for node in plan.get("parsed_plan", {}).get("nodes", []):
        if not isinstance(node, dict):
            continue
        tool = str(node.get("tool", "")).replace("__", ".").strip()
        if tool and tool not in tools:
            tools.append(tool)
    return tools


def pr(pred: list[str], gold: list[str]) -> dict[str, float]:
    pred_set = set(pred)
    gold_set = set(gold)
    hits = len(pred_set & gold_set)
    return {
        "num_pred": len(pred_set),
        "num_gold": len(gold_set),
        "hits": hits,
        "recall": hits / max(len(gold_set), 1),
        "precision": hits / max(len(pred_set), 1),
    }


def mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--split", default="dev")
    parser.add_argument("--anchored-records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/task_tool_anchoring/tool_set_eval_dev57"))
    parser.add_argument("--sbert-model", default="local_models/all-MiniLM-L6-v2")
    parser.add_argument("--semantic-fixed-k", type=int, default=10)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_tasks = {task.task_id: task for task in load_gold_tasks(args.appworld_root, args.split, max_tasks=None)}
    records = [
        json.loads(line)
        for line in args.anchored_records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [row for row in records if row["task_id"] in gold_tasks]

    tool_docs = load_tool_docs(args.appworld_root)
    tool_names = [tool.name for tool in tool_docs]
    model_path = Path(args.sbert_model)
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    model = SentenceTransformer(str(model_path), device="cuda")
    tool_embeddings = normalize_rows(model.encode([tool.text for tool in tool_docs], batch_size=64, convert_to_numpy=True))
    task_embeddings = normalize_rows(model.encode([row["task_instruction"] for row in records], batch_size=64, convert_to_numpy=True))
    scores = task_embeddings @ tool_embeddings.T

    per_task: list[dict[str, Any]] = []
    dag_metrics = []
    semantic_budget_metrics = []
    semantic_fixed_metrics = []
    for i, row in enumerate(records):
        task = gold_tasks[row["task_id"]]
        gold = task.gold_tools
        dag_tools = extract_dag_tools(row.get("anchored_dag_plan", {}))
        budget_k = max(len(dag_tools), 1)
        semantic_budget = [tool_names[int(j)] for j in np.argsort(-scores[i])[:budget_k]]
        semantic_fixed = [tool_names[int(j)] for j in np.argsort(-scores[i])[: args.semantic_fixed_k]]
        dag_m = pr(dag_tools, gold)
        sem_budget_m = pr(semantic_budget, gold)
        sem_fixed_m = pr(semantic_fixed, gold)
        dag_metrics.append(dag_m)
        semantic_budget_metrics.append(sem_budget_m)
        semantic_fixed_metrics.append(sem_fixed_m)
        per_task.append(
            {
                "task_id": row["task_id"],
                "task": row["task_instruction"],
                "ground_truth_tools": gold,
                "dag_tools": dag_tools,
                "semantic_retrieval_tools_budget_matched": semantic_budget,
                "semantic_retrieval_tools_top10": semantic_fixed,
                "metrics": {
                    "dag_tools": dag_m,
                    "semantic_budget_matched": sem_budget_m,
                    "semantic_top10": sem_fixed_m,
                },
            }
        )

    summary = {
        "split": args.split,
        "num_tasks": len(per_task),
        "anchored_records": str(args.anchored_records),
        "semantic_fixed_k": args.semantic_fixed_k,
        "metrics": {
            "dag_tools": {
                "macro_recall": mean(dag_metrics, "recall"),
                "macro_precision": mean(dag_metrics, "precision"),
                "avg_num_pred": mean(dag_metrics, "num_pred"),
                "avg_num_gold": mean(dag_metrics, "num_gold"),
            },
            "semantic_budget_matched_to_dag_size": {
                "macro_recall": mean(semantic_budget_metrics, "recall"),
                "macro_precision": mean(semantic_budget_metrics, "precision"),
                "avg_num_pred": mean(semantic_budget_metrics, "num_pred"),
                "avg_num_gold": mean(semantic_budget_metrics, "num_gold"),
            },
            "semantic_top10": {
                "macro_recall": mean(semantic_fixed_metrics, "recall"),
                "macro_precision": mean(semantic_fixed_metrics, "precision"),
                "avg_num_pred": mean(semantic_fixed_metrics, "num_pred"),
                "avg_num_gold": mean(semantic_fixed_metrics, "num_gold"),
            },
        },
    }
    (out_dir / "tool_set_records.json").write_text(json.dumps(per_task, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Dev Tool Set Recall / Precision",
        "",
        f"Tasks: {len(per_task)}",
        "",
        "| Method | Recall | Precision | Avg Pred Tools | Avg Gold Tools |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["metrics"].items():
        md.append(
            f"| {name} | {metrics['macro_recall']:.4f} | {metrics['macro_precision']:.4f} | "
            f"{metrics['avg_num_pred']:.2f} | {metrics['avg_num_gold']:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

