"""analyze_requirements tool — detect target behaviors and recommend data categories."""

import json
from typing import Annotated

from dataagent_benchmark.domain.behavior_taxonomy import BEHAVIOR_TAXONOMY
from dataagent_benchmark.domain.tool_context import Injected
from dataagent_benchmark.tools.preselection.utils import _detect_behaviors


def analyze_requirements(
    target_behaviors: Annotated[
        str, "Comma-separated behavior labels (auto-detected when empty)"
    ] = "",
    output_style: Annotated[
        str, "Preferred style: short_answer, step_by_step, or structured_json"
    ] = "step_by_step",
    allow_unknown: Annotated[bool, "Include unanswerable samples to reduce hallucination"] = False,
    # --- Injected by env ---
    task_description: Annotated[str, Injected("task_description")] = "",
    sample_budget: Annotated[int, Injected("sample_budget")] = 100_000,
) -> str:
    """Analyze task description to detect target behaviors.

    Recommends data categories with SFT-recipe ratios. Call this FIRST.
    """
    if target_behaviors.strip():
        behavior_keys = [b.strip() for b in target_behaviors.split(",")]
    else:
        behavior_keys = _detect_behaviors(task_description)

    valid_behaviors = []
    unknown_behaviors = []
    for key in behavior_keys:
        if key in BEHAVIOR_TAXONOMY:
            valid_behaviors.append(key)
        else:
            unknown_behaviors.append(key)

    if not valid_behaviors:
        valid_behaviors = ["domain_qa"]

    categories_needed: dict[str, float] = {}
    instruction_types: list[str] = []
    for key in valid_behaviors:
        beh = BEHAVIOR_TAXONOMY[key]
        cat = beh["primary_category"]
        categories_needed[cat] = categories_needed.get(cat, 0) + 1
        instruction_types.append(key)

    if "general" not in categories_needed:
        categories_needed["general"] = 0.5
    if allow_unknown:
        categories_needed["unanswerable"] = 0.5

    total_weight = sum(categories_needed.values())
    category_ratios = {
        cat: round(weight / total_weight * 100, 1) for cat, weight in categories_needed.items()
    }

    output_styles = set()
    for key in valid_behaviors:
        output_styles.update(BEHAVIOR_TAXONOMY[key]["suggested_output_styles"])

    recommended_style = output_style
    if recommended_style not in output_styles and output_styles:
        recommended_style = list(output_styles)[0]

    spec: dict = {
        "task_description": task_description,
        "detected_behaviors": [
            {
                "behavior": key,
                "description": BEHAVIOR_TAXONOMY[key]["description"],
                "primary_category": BEHAVIOR_TAXONOMY[key]["primary_category"],
            }
            for key in valid_behaviors
        ],
        "recommended_categories": category_ratios,
        "output_style": recommended_style,
        "all_compatible_styles": sorted(output_styles),
        "allow_unknown": allow_unknown,
        "instruction_types": instruction_types,
        "sample_budget": sample_budget,
    }
    if unknown_behaviors:
        spec["warnings"] = [
            f"Unknown behavior(s) ignored: {', '.join(unknown_behaviors)}. "
            f"Available: {', '.join(BEHAVIOR_TAXONOMY.keys())}"
        ]
    return json.dumps(spec, indent=2)
