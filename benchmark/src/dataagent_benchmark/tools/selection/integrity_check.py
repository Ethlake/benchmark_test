"""integrity_check tool — verify structural integrity of a dataset."""

import json
from typing import Annotated, Any

from dataagent_benchmark.domain.tool_context import PerDataset


def integrity_check(
    dataset_id: Annotated[str, "Dataset to check"],
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    text_column: Annotated[str, PerDataset("text_column")] = "",
) -> str:
    """Check structural integrity of a dataset.

    Verifies required fields exist and text content is non-empty.
    Run for EVERY dataset.
    """
    ds = dataset
    total = len(ds)
    actual_cols = ds.column_names

    format_rules: list[str] = [f"{text_column} is non-empty"]
    values = ds[text_column]
    pass_count = sum(1 for v in values if isinstance(v, str) and v.strip() != "")

    pass_rate = pass_count / max(total, 1)

    return json.dumps(
        {
            "dataset": dataset_id,
            "original_samples": total,
            "checks": {
                "available_fields": actual_cols,
                "format_rules": format_rules,
            },
            "pass_rate": round(pass_rate, 4),
            "retained": pass_count,
            "removed": total - pass_count,
        },
        indent=2,
    )
