"""text_quality_check tool — measure text quality metrics."""

import json
from typing import Annotated, Any

from dataagent_benchmark.domain.tool_context import PerDataset
from dataagent_benchmark.tools.preselection.utils import _REASONING_MARKERS


def text_quality_check(
    dataset_id: Annotated[str, "Dataset to check"],
    min_length: Annotated[int, "Minimum character length for text field"] = 10,
    max_length: Annotated[int, "Maximum character length for text field"] = 5000,
    require_reasoning_steps: Annotated[bool, "Require step-by-step reasoning in solutions"] = False,
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    text_column: Annotated[str, PerDataset("text_column")] = "",
) -> str:
    """Measure text quality: text lengths, reasoning steps, language heuristics."""
    ds = dataset
    total = len(ds)
    texts = ds[text_column]

    lengths = [len(v) if isinstance(v, str) else 0 for v in texts]
    avg_len = round(sum(lengths) / max(total, 1), 1)

    reasoning_count = 0
    detected_marker = None
    for marker in _REASONING_MARKERS:
        hits = sum(1 for v in texts if isinstance(v, str) and marker in v)
        if hits > total * 0.3:
            reasoning_count = hits
            detected_marker = marker
            break
    if reasoning_count == 0:
        multi_line = sum(1 for v in texts if isinstance(v, str) and v.count("\n") >= 2)
        if multi_line > total * 0.3:
            reasoning_count = multi_line
            detected_marker = "multi-line"
    has_reasoning = reasoning_count > total * 0.3

    pass_mask = [min_length <= lengths[i] <= max_length for i in range(total)]
    if require_reasoning_steps:
        for i in range(total):
            if pass_mask[i]:
                a = texts[i]
                has_steps = isinstance(a, str) and (
                    any(m in a for m in _REASONING_MARKERS) or a.count("\n") >= 2
                )
                pass_mask[i] = has_steps

    retained = sum(pass_mask)

    return json.dumps(
        {
            "dataset": dataset_id,
            "original_samples": total,
            "checks": {
                "min_length": min_length,
                "max_length": max_length,
                "require_reasoning_steps": require_reasoning_steps,
                "has_reasoning_steps": has_reasoning,
                "reasoning_marker": detected_marker,
                "avg_text_len": avg_len,
            },
            "pass_rate": round(retained / max(total, 1), 4),
            "retained": retained,
            "removed": total - retained,
        },
        indent=2,
    )
