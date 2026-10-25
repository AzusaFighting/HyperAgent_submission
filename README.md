# HyperAgent

This repository contains the implementation of HyperAgent's task-conditioned tool-schema hypergraph planning components.

The released source includes:

- tool-schema hypergraph loading and port-level dependency indexing;
- tool and schema anchoring from a task request;
- bounded backward expansion for constructing a task-conditioned Tool Context Graph;
- DAG-plan generation from the retrieved context;
- plan parsing, reporting, and optional blind plan comparison.

No AppWorld data, API documentation, model checkpoints, experiment traces, cached responses, or evaluation results are included in this archive.

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Provide the AppWorld installation, a generated tool-schema hypergraph JSON file, and a SentenceTransformer model location in a copy of `config.example.yaml`. The implementation reads the LLM credential from `DEEPSEEK_API_KEY`; `--dry-run` avoids API calls for a configuration check.

## Run

```bash
python -m hyperagent.run --config config.yaml --split dev --max-tasks 10
```

The expected hypergraph format has `nodes` and `hyperedges` fields. Tool invocation hyperedges specify their input-side nodes in `tail` and output/effect nodes in `head`; schema-dependency hyperedges specify the producer and consumer API fields.

## Package layout

- `hyperagent/tool_hypergraph.py`: hypergraph representation and dependency indexes.
- `hyperagent/anchor_extraction.py`: LLM and embedding-based anchor extraction.
- `hyperagent/subgraph_builder.py`: bounded backward retrieval of producer tools.
- `hyperagent/plan_context.py` and `hyperagent/dag_planner.py`: context serialization and Task-DAG generation.
- `hyperagent/run.py`: experiment entry point.

