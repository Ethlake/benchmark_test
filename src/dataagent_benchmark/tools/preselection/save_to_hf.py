"""save_to_hf tool — concatenate and push datasets to HuggingFace Hub."""

import json
from typing import Annotated

from dataagent_benchmark.domain.tool_context import AllDatasets, FromEnv
from dataagent_benchmark.tools.postselection.utils import (
    _build_combined_dataset,
    _parse_selection,
)


def save_to_hf(
    sample_counts_csv: Annotated[
        str, "Comma-separated sample counts per dataset (use 0 to include all)"
    ],
    repo_id: Annotated[str, "HuggingFace Hub repo ID (e.g. 'user/my-dataset')"],
    private: Annotated[bool, "Whether to create a private repo"] = True,
    # --- Injected by env ---
    dataset_names: Annotated[list[str], FromEnv("dataset_names")] = None,
    all_working_datasets: Annotated[dict, AllDatasets("working")] = None,
) -> str:
    """Concatenate working copies of selected datasets and push to HuggingFace Hub."""
    datasets_csv = ",".join(dataset_names)
    names, counts, parse_error = _parse_selection(datasets_csv, sample_counts_csv)
    if parse_error is not None:
        return json.dumps({"error": parse_error})

    combined, included, warnings = _build_combined_dataset(all_working_datasets, names, counts)
    if combined is None:
        return json.dumps({"error": "No valid datasets to upload.", "warnings": warnings})

    if not repo_id.strip():
        return json.dumps({"error": "repo_id cannot be empty for save_to_hf."})
    combined.push_to_hub(repo_id, private=private)

    result: dict = {
        "repo_id": repo_id,
        "total_rows": len(combined),
        "datasets_included": included,
        "private": private,
        "status": "uploaded",
    }
    if warnings:
        result["warnings"] = warnings
    return json.dumps(result, indent=2)
