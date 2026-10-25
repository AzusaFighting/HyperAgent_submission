from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .anchor_extraction import build_anchor_result
from .appworld_loader import load_tasks, load_tool_docs
from .config import load_config, write_config
from .dag_planner import generate_anchored_dag_plan
from .llm_client import DeepSeekClient
from .plan_context import format_plan_context
from .reporting import append_jsonl, write_text
from .schema_matching import EmbeddingIndex
from .subgraph_builder import build_task_tool_subgraph
from .tool_hypergraph import ToolHypergraph


def load_seed_records(paths: list[Path]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("anchored_dag_plan"):
                records[row["task_id"]] = {
                    "task_id": row["task_id"],
                    "task_instruction": row["task_instruction"],
                    "anchor_extraction": row["anchor_extraction"],
                    "task_tool_subgraph": row["task_tool_subgraph"],
                    "anchored_dag_plan": row["anchored_dag_plan"],
                    "source": str(path),
                }
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate anchored DAG plans only, without raw baseline or judge.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--config", default="configs/task_tool_anchoring.yaml")
    parser.add_argument("--run-id", default="anchored_only")
    parser.add_argument("--seed-records", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    config = load_config(args.config)
    appworld_root = Path(config["appworld_root"])
    graph_path = Path(config["hypergraph_path"])
    if not graph_path.is_absolute():
        graph_path = repo_root / graph_path
    model_path = Path(config["sbert_model"])
    if not model_path.is_absolute():
        model_path = repo_root / model_path

    run_dir = repo_root / "reports" / "task_tool_anchoring" / args.run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "raw_llm_outputs").mkdir(parents=True, exist_ok=True)
    write_config(config, run_dir / "config.yaml")

    seed_paths = [Path(x.strip()) for x in args.seed_records.split(";") if x.strip()]
    seed_records = load_seed_records(seed_paths)

    graph = ToolHypergraph(graph_path)
    tool_docs = load_tool_docs(appworld_root)
    task_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()] or None
    tasks = load_tasks(appworld_root, args.split, max_tasks=args.max_tasks, task_ids=task_ids)
    llm = DeepSeekClient(config["llm"], dry_run=args.dry_run)

    graph_tool_names = sorted(graph.tools)
    graph_tool_texts = [graph.tool_text(name) for name in graph_tool_names]
    schema_names = sorted(graph.schemas)
    schema_texts = [graph.schemas[name].text for name in schema_names]
    tool_index = EmbeddingIndex(str(model_path), graph_tool_texts, graph_tool_names)
    schema_index = EmbeddingIndex(str(model_path), schema_texts, schema_names)

    reused = 0
    generated = 0
    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{len(tasks)}] {task.task_id}", flush=True)
        if task.task_id in seed_records:
            record = seed_records[task.task_id]
            reused += 1
        else:
            anchors = build_anchor_result(
                task.instruction,
                tool_docs,
                graph,
                llm,
                schema_index,
                tool_index,
                int(config["retrieval"]["schema_anchor_top_k"]),
                int(config["retrieval"]["tool_anchor_top_k"]),
            )
            subgraph = build_task_tool_subgraph(task.instruction, anchors, graph, tool_index, config)
            context = format_plan_context(
                task.instruction,
                anchors,
                subgraph,
                graph,
                max_tools=int(config["planning"]["max_context_tools"]),
            )
            anchored_plan = generate_anchored_dag_plan(task.instruction, context, llm)
            record = {
                "task_id": task.task_id,
                "task_instruction": task.instruction,
                "anchor_extraction": anchors.to_json(),
                "task_tool_subgraph": subgraph.to_json(graph),
                "anchored_dag_plan": anchored_plan,
                "source": "generated",
            }
            write_text(run_dir / "raw_llm_outputs" / f"{task.task_id}_anchor_extraction.txt", anchors.raw_llm_output)
            write_text(run_dir / "raw_llm_outputs" / f"{task.task_id}_anchored_plan.txt", anchored_plan["raw_llm_output"])
            generated += 1
        append_jsonl(run_dir / "anchored_task_plan_records.jsonl", record)
        append_jsonl(run_dir / "anchored_plan_records.jsonl", {"task_id": task.task_id, **record["anchored_dag_plan"]})
        append_jsonl(run_dir / "anchor_records.jsonl", {"task_id": task.task_id, **record["anchor_extraction"]})
        append_jsonl(run_dir / "subgraph_records.jsonl", {"task_id": task.task_id, **record["task_tool_subgraph"]})

    summary = {"run_dir": str(run_dir), "tasks": len(tasks), "reused": reused, "generated": generated}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

