"""transform_dataset tool — apply row-level transformations."""

import json
import random
import re
import unicodedata
from typing import Annotated, Any

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import DatasetUpdate, PerDataset


def _normalize_text(text: str) -> str:
    """Normalize text for near-dedup: lowercase, strip accents, collapse whitespace."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text


def transform_dataset(
    dataset_id: Annotated[str, "Dataset to transform"],
    operation: Annotated[
        str,
        "Operation: filter_by_length, remove_nulls, remove_duplicates, sample, reset",
    ],
    column: Annotated[str, "Target column for the operation"] = "",
    min_length: Annotated[int, "Minimum character length for filter_by_length"] = 0,
    max_length: Annotated[int, "Maximum character length for filter_by_length"] = 999999,
    sample_count: Annotated[int, "Number of rows for sample operation"] = 0,
    dedup_method: Annotated[str, "Deduplication method: 'exact' or 'near'"] = "exact",
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    original_dataset: Annotated[Any, PerDataset("original")] = None,
    text_column: Annotated[str, PerDataset("text_column")] = "",
    transforms: Annotated[list, PerDataset("transforms")] = None,
) -> StepResult | str:
    """Apply a row-level transformation to a dataset's working copy.

    Apply one operation at a time and re-check with
    explore_dataset.
    """
    valid_ops = (
        "filter_by_length",
        "remove_nulls",
        "remove_duplicates",
        "sample",
        "reset",
    )
    if operation not in valid_ops:
        return json.dumps(
            {"error": f"Unknown operation '{operation}'. Use: {', '.join(valid_ops)}"}
        )

    if operation == "reset":
        ds = original_dataset
        payload = json.dumps(
            {
                "dataset": dataset_id,
                "operation": "reset",
                "rows": len(ds),
                "message": "All transforms cleared.",
            }
        )
        return StepResult(
            payload=payload,
            created=[ArtifactRef("dataset", dataset_id, 0)],
            updates={f"active_dataset:{dataset_id}": ArtifactRef("dataset", dataset_id, 0)},
            metrics={"rows": len(ds), "operation": "reset"},
            dataset_updates={dataset_id: DatasetUpdate(dataset=ds, is_reset=True)},
        )

    ds = dataset
    rows_before = len(ds)

    if operation == "filter_by_length":
        col = column or text_column
        if col not in ds.column_names:
            new_ds = ds
        else:
            new_ds = ds.filter(
                lambda row: isinstance(row[col], str) and min_length <= len(row[col]) <= max_length
            )

    elif operation == "remove_nulls":
        col = column or text_column
        if col not in ds.column_names:
            new_ds = ds
        else:
            new_ds = ds.filter(lambda row: isinstance(row[col], str) and row[col].strip() != "")

    elif operation == "remove_duplicates":
        col = column or text_column
        if col not in ds.column_names:
            new_ds = ds
        else:
            values = ds[col]
            if dedup_method == "near":
                seen_set: set[str] = set()
                keep_indices = []
                for i, v in enumerate(values):
                    normed = _normalize_text(v) if isinstance(v, str) else ""
                    if normed not in seen_set:
                        seen_set.add(normed)
                        keep_indices.append(i)
            else:
                seen_hash: set[int] = set()
                keep_indices = []
                for i, v in enumerate(values):
                    h = hash(v) if isinstance(v, str) else hash(str(v))
                    if h not in seen_hash:
                        seen_hash.add(h)
                        keep_indices.append(i)
            new_ds = ds.select(keep_indices)

    elif operation == "sample":
        count = min(sample_count, rows_before) if sample_count > 0 else rows_before
        if count < rows_before:
            rng = random.Random(42)
            indices = sorted(rng.sample(range(rows_before), count))
            new_ds = ds.select(indices)
        else:
            new_ds = ds

    else:
        new_ds = ds

    rows_after = len(new_ds)
    transform_info = {
        "operation": operation,
        "column": column or None,
        "rows_before": rows_before,
        "rows_after": rows_after,
    }

    removed = rows_before - rows_after
    pct = removed / max(rows_before, 1) * 100

    result: dict = {
        "dataset": dataset_id,
        "operation": operation,
        "column": column or None,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": removed,
        "removal_rate": f"{pct:.1f}%",
        "transforms_total": len(transforms),
    }
    if operation == "filter_by_length":
        result["parameters"] = {"min_length": min_length, "max_length": max_length}
    elif operation == "remove_duplicates":
        result["parameters"] = {"method": dedup_method}
    elif operation == "sample":
        result["parameters"] = {"sample_count": sample_count}

    payload = json.dumps(result, indent=2)
    return StepResult(
        payload=payload,
        created=[ArtifactRef("dataset", dataset_id, 0)],
        updates={f"active_dataset:{dataset_id}": ArtifactRef("dataset", dataset_id, 0)},
        metrics={"rows_before": rows_before, "rows_after": rows_after, "operation": operation},
        dataset_updates={
            dataset_id: DatasetUpdate(dataset=new_ds, transform_info=transform_info)
        },
    )
