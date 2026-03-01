"""inspect_samples tool — view real example rows with quality notes."""

import json
from typing import Annotated, Any

from dataagent_benchmark.domain.tool_context import PerDataset
from dataagent_benchmark.tools.preselection.utils import _generate_quality_note


def inspect_samples(
    dataset_id: Annotated[str, "Dataset to inspect"],
    num_samples: Annotated[int, "Number of example rows to return (max 5)"] = 5,
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    dataset_config: Annotated[Any, PerDataset("config")] = None,
) -> str:
    """View real example rows from a dataset with automated quality notes."""
    n = min(num_samples, 5)
    rows = [dict(dataset[i]) for i in range(min(n, len(dataset)))]

    class _Meta:
        pass

    meta = _Meta()
    meta.question_field = dataset_config.columns.text
    meta.answer_field = dataset_config.columns.text
    meta.answer_marker = None
    meta.answer_style = "short_answer"

    samples = []
    for row in rows:
        row["_quality_note"] = _generate_quality_note(row, meta)
        samples.append(row)

    issues = []
    good = 0
    for row in samples:
        note = row.get("_quality_note", "")
        if note.startswith("Issue") or note.startswith("Borderline"):
            issues.append(note)
        else:
            good += 1

    return json.dumps(
        {
            "dataset": dataset_id,
            "num_returned": len(samples),
            "samples": samples,
            "summary": {
                "good_examples": good,
                "issues_found": len(issues),
                "issue_descriptions": issues,
            },
        },
        indent=2,
    )
