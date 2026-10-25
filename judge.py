from __future__ import annotations

import json
import random
from typing import Any

from .llm_client import DeepSeekClient, parse_json_object


SCORE_KEYS = [
    "task_relevance",
    "tool_appropriateness",
    "dependency_coherence",
    "plan_completeness",
    "execution_plausibility",
    "minimality",
    "terminal_effect_correctness",
    "dag_quality",
]


def judge_plans(task: str, tool_summary: str, raw_plan: dict[str, Any], anchored_plan: dict[str, Any], llm_client: DeepSeekClient, rng: random.Random) -> dict[str, Any]:
    order = ["raw", "anchored"]
    rng.shuffle(order)
    plan_a = raw_plan if order[0] == "raw" else anchored_plan
    plan_b = raw_plan if order[1] == "raw" else anchored_plan
    system = "You are a blind judge comparing two DAG tool plans. Return strict JSON only."
    user = (
        f"Task instruction:\n{task}\n\n"
        f"Available tool summary:\n{tool_summary}\n\n"
        f"Plan A:\n{json.dumps(plan_a.get('parsed_plan', {}), ensure_ascii=False, indent=2)}\n\n"
        f"Plan B:\n{json.dumps(plan_b.get('parsed_plan', {}), ensure_ascii=False, indent=2)}\n\n"
        "Score each plan 1-5 on: task_relevance, tool_appropriateness, dependency_coherence, "
        "plan_completeness, execution_plausibility, minimality, terminal_effect_correctness, dag_quality. "
        "Report pairwise_preference as A, B, or tie. Use failure modes from: wrong_terminal_tool, "
        "missing_major_subgoal, unsupported_dependency, irrelevant_detour, flat_non_dag_plan, overly_generic_plan, "
        "tool_mismatch, incomplete_plan, cyclic_plan, parse_failure.\n"
        "Return JSON with keys plan_a_scores, plan_b_scores, pairwise_preference, preference_reason, "
        "major_failure_modes_plan_a, major_failure_modes_plan_b."
    )
    response = llm_client.complete_json(system, user)
    parsed, error = parse_json_object(response.text)
    pref = parsed.get("pairwise_preference", "tie")
    if pref not in {"A", "B", "tie"}:
        pref = "tie"
    def scores(name: str) -> dict[str, int]:
        raw = parsed.get(name, {})
        return {k: max(1, min(5, int(raw.get(k, 1)))) for k in SCORE_KEYS}
    raw_scores = scores("plan_a_scores" if order[0] == "raw" else "plan_b_scores")
    anchored_scores = scores("plan_a_scores" if order[0] == "anchored" else "plan_b_scores")
    if pref == "tie":
        mapped = "tie"
    elif (pref == "A" and order[0] == "anchored") or (pref == "B" and order[1] == "anchored"):
        mapped = "anchored"
    else:
        mapped = "raw"
    return {
        "raw_plan_scores": raw_scores,
        "anchored_plan_scores": anchored_scores,
        "pairwise_preference": mapped,
        "judge_raw_output": response.text,
        "judge_parse_error": error,
        "blind_order": {"A": order[0], "B": order[1]},
        "preference_reason": parsed.get("preference_reason", ""),
        "major_failure_modes_raw": parsed.get("major_failure_modes_plan_a" if order[0] == "raw" else "major_failure_modes_plan_b", []),
        "major_failure_modes_anchored": parsed.get("major_failure_modes_plan_a" if order[0] == "anchored" else "major_failure_modes_plan_b", []),
    }
