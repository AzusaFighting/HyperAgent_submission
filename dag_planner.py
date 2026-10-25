from __future__ import annotations

from .llm_client import DeepSeekClient
from .plan_parser import parse_plan


DAG_SCHEMA = """
Return strict JSON:
{
  "nodes": [
    {
      "id": "n1",
      "goal": "...",
      "tool": "app.tool",
      "expected_inputs": [],
      "expected_outputs": [],
      "argument_sketch": {}
    }
  ],
  "edges": [
    {"from": "n1", "to": "n2", "support": "n1 output provides ... for n2"}
  ]
}
"""


def generate_anchored_dag_plan(task: str, plan_context: str, llm_client: DeepSeekClient) -> dict:
    system = (
        "You generate complete acyclic DAG tool plans using only the graph-conditioned candidate context. "
        "Return JSON only."
    )
    user = (
        f"{plan_context}\n\n"
        "Generate an anchored DAG plan. Use only tools from the task-conditioned tool subgraph when possible. "
        "Do not compress away intermediate prerequisite, collection-expansion, detail-fetching, authentication, "
        "lookup, or id-resolution steps. If a library/list/search tool returns ids or references and the task needs "
        "item-level fields, include the detail tool that consumes those ids. Edges should represent data, "
        "prerequisite, or logical support. Do not include validation or execution repair.\n"
        f"{DAG_SCHEMA}"
    )
    response = llm_client.complete_json(system, user)
    return parse_plan(response.text)
