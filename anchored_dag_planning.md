Generate a complete acyclic DAG tool plan from the graph-conditioned planning context.

Use the task-conditioned tool subgraph when possible. Do not compress away
intermediate prerequisite, collection-expansion, detail-fetching,
authentication, lookup, or id-resolution steps. If a library/list/search tool
returns ids or references and the task needs item-level fields, include the
detail tool that consumes those ids. Return strict JSON with `nodes` and
`edges`.
