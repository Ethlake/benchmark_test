"""save_to_disk tool — concatenate and save datasets locally."""

import json
from pathlib import Path
from typing import Annotated

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import AllDatasets, FromEnv
from dataagent_benchmark.tools.postselection.utils import (
    _build_combined_dataset,
    _parse_selection,
)


def save_to_disk(
    sample_counts_csv: Annotated[
        str, "Comma-separated sample counts per dataset (use 0 to include all)"
    ],
    file_format: Annotated[str, "Output format: 'parquet' or 'jsonl'"] = "parquet",
    # --- Injected by env ---
    dataset_names: Annotated[list[str], FromEnv("dataset_names")] = None,
    output_path: Annotated[str, FromEnv("output_path")] = "",
    all_working_datasets: Annotated[dict, AllDatasets("working")] = None,
) -> StepResult | str:
    """Concatenate working copies of selected datasets and save locally."""
    datasets_csv = ",".join(dataset_names)
    names, counts, parse_error = _parse_selection(datasets_csv, sample_counts_csv)
    if parse_error is not None:
        return json.dumps({"error": parse_error})

    combined, included, warnings = _build_combined_dataset(all_working_datasets, names, counts)
    if combined is None:
        return json.dumps({"error": "No valid datasets to save.", "warnings": warnings})

    fmt = file_format.strip().lower()
    if fmt not in ("parquet", "jsonl"):
        return json.dumps(
            {"error": f"Unknown file_format '{file_format}'. Use 'parquet' or 'jsonl'."}
        )

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        combined.to_parquet(str(path))
    else:
        combined.to_json(str(path), orient="records", lines=True)

    result: dict = {
        "path": str(path),
        "format": fmt,
        "total_rows": len(combined),
        "datasets_included": included,
        "status": "saved",
    }
    if warnings:
        result["warnings"] = warnings

    return StepResult(
        payload=json.dumps(result, indent=2),
        created=[ArtifactRef("saved_file", str(path), 0)],
        metrics={"total_rows": len(combined)},
    )
