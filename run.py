from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from .anchor_extraction import build_anchor_result
from .appworld_loader import load_tasks, load_tool_docs
from .config import load_config, write_config
from .dag_planner import generate_anchored_dag_plan
from .judge import judge_plans
from .llm_client import DeepSeekClient
from .metrics import aggregate
from .plan_context import format_plan_context
from .raw_planner import generate_raw_llm_plan
from .reporting import aggregate_metrics_md, append_jsonl, failure_analysis_md, sample_cases_md, write_text
from .schema_matching import EmbeddingIndex
from .subgraph_builder import build_task_tool_subgraph
from .tool_hypergraph import ToolHypergraph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--config", default="configs/task_tool_anchoring.yaml")
    parser.add_argument("--run-id", default="anchoring_dev")
    parser.add_argument("--seed", type=int, default=13)
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

    rng = random.Random(args.seed)
    graph = ToolHypergraph(graph_path)
    tool_docs = load_tool_docs(appworld_root)
    tool_doc_by_name = {t.name: t for t in tool_docs}
    task_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()] or None
    tasks = load_tasks(appworld_root, args.split, max_tasks=args.max_tasks, task_ids=task_ids)
    llm = DeepSeekClient(config["llm"], dry_run=args.dry_run)

    graph_tool_names = sorted(graph.tools)
    graph_tool_texts = [graph.tool_text(name) for name in graph_tool_names]
    schema_names = sorted(graph.schemas)
    schema_texts = [graph.schemas[name].text for name in schema_names]
    tool_index = EmbeddingIndex(str(model_path), graph_tool_texts, graph_tool_names)
    schema_index = EmbeddingIndex(str(model_path), schema_texts, schema_names)
    raw_doc_names = [t.name for t in tool_docs]
    raw_doc_texts = [t.text for t in tool_docs]
    raw_tool_index = EmbeddingIndex(str(model_path), raw_doc_texts, raw_doc_names)

    records = []
    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{len(tasks)}] {task.task_id}", flush=True)
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
        raw_hits = raw_tool_index.query(task.instruction, int(config["planning"]["max_raw_tools"]))
        raw_docs = [tool_doc_by_name[h["id"]] for h in raw_hits if h["id"] in tool_doc_by_name]
        raw_plan = generate_raw_llm_plan(task.instruction, raw_docs, llm)
        tool_summary = "\n".join(f"- {d.name}: {d.description}" for d in raw_docs[:40])
        judge = judge_plans(task.instruction, tool_summary, raw_plan, anchored_plan, llm, rng)

        record = {
            "task_id": task.task_id,
            "task_instruction": task.instruction,
            "anchor_extraction": anchors.to_json(),
            "task_tool_subgraph": subgraph.to_json(graph),
            "anchored_dag_plan": anchored_plan,
            "raw_llm_plan": raw_plan,
            "llm_as_plan_judge": judge,
        }
        records.append(record)
        append_jsonl(run_dir / "task_plan_records.jsonl", record)
        append_jsonl(run_dir / "anchor_records.jsonl", {"task_id": task.task_id, **anchors.to_json()})
        append_jsonl(run_dir / "subgraph_records.jsonl", {"task_id": task.task_id, **subgraph.to_json(graph)})
        append_jsonl(run_dir / "raw_plan_records.jsonl", {"task_id": task.task_id, **raw_plan})
        append_jsonl(run_dir / "anchored_plan_records.jsonl", {"task_id": task.task_id, **anchored_plan})
        append_jsonl(run_dir / "judge_records.jsonl", {"task_id": task.task_id, **judge})
        write_text(run_dir / "raw_llm_outputs" / f"{task.task_id}_raw_plan.txt", raw_plan["raw_llm_output"])
        write_text(run_dir / "raw_llm_outputs" / f"{task.task_id}_anchored_plan.txt", anchored_plan["raw_llm_output"])
        write_text(run_dir / "raw_llm_outputs" / f"{task.task_id}_anchor_extraction.txt", anchors.raw_llm_output)

    metrics = aggregate(records, graph)
    (run_dir / "aggregate_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text(run_dir / "aggregate_metrics.md", aggregate_metrics_md(metrics))
    write_text(run_dir / "failure_analysis.md", failure_analysis_md(metrics))
    write_text(run_dir / "sample_cases.md", sample_cases_md(records))
    print(json.dumps({"run_dir": str(run_dir), **metrics["pairwise"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
