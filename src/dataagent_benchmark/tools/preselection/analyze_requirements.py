"""analyze_requirements tool with local lightweight heuristics."""

from __future__ import annotations

import json
from typing import Annotated

from dataagent_benchmark.domain.tool_context import Injected

_BEHAVIORS = {
    "ocr": {"keywords": {"ocr", "document", "receipt"}, "category": "ocr_doc_chart"},
    "reasoning": {"keywords": {"reason", "math", "multi-step", "proof"}, "category": "domain"},
    "general_qa": {"keywords": {"general", "qa", "vqa", "question"}, "category": "general"},
}


def _detect_behaviors(text: str) -> list[str]:
    tokens = set(text.lower().replace("-", " ").split())
    hits: list[str] = []
    for name, item in _BEHAVIORS.items():
        if tokens & item["keywords"]:
            hits.append(name)
    return hits or ["general_qa"]


def analyze_requirements(
    target_behaviors: Annotated[str, "Comma-separated behavior labels (auto-detected when empty)"] = "",
    output_style: Annotated[str, "Preferred style: short_answer, step_by_step, or structured_json"] = "step_by_step",
    allow_unknown: Annotated[bool, "Include unanswerable samples to reduce hallucination"] = False,
    task_description: Annotated[str, Injected("task_description")] = "",
    sample_budget: Annotated[int, Injected("sample_budget")] = 100_000,
) -> str:
    """Analyze task description and return recommended data categories."""
    behaviors = [x.strip() for x in target_behaviors.split(",") if x.strip()] if target_behaviors else _detect_behaviors(task_description)
    categories: dict[str, float] = {}
    for behavior in behaviors:
        mapped = _BEHAVIORS.get(behavior, {"category": "general"})
        cat = mapped["category"]
        categories[cat] = categories.get(cat, 0) + 1.0

    if allow_unknown:
        categories["unanswerable"] = categories.get("unanswerable", 0.0) + 0.5

    total = sum(categories.values()) or 1.0
    ratios = {k: round(v / total * 100.0, 1) for k, v in categories.items()}
    return json.dumps(
        {
            "task_description": task_description,
            "detected_behaviors": behaviors,
            "recommended_categories": ratios,
            "output_style": output_style,
            "sample_budget": sample_budget,
        },
        indent=2,
    )
