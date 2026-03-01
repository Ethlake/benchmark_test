"""vlm_filter tool — image integrity filtering for VLM datasets."""

import json
from collections import Counter
from typing import Annotated, Any

import structlog

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import DatasetUpdate, PerDataset
from dataagent_benchmark.infra.vlm_ops import FilterConfig, decide_keep

log = structlog.get_logger()


def vlm_filter(
    dataset_id: Annotated[str, "Dataset to filter"],
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    image_column: Annotated[str, PerDataset("image_column")] = "",
) -> StepResult:
    """Filter a VLM dataset by image integrity checks.

    Removes rows with missing/corrupt images, images with sides
    below 200px, and aspect ratios above 3:1. Operates on the
    working copy in-place. Text quality filtering is handled
    separately by quality_filter.
    """
    log.info("vlm_filter", dataset=dataset_id)
    ds = dataset
    original_count = len(ds)
    log.info("vlm_filter_start", dataset=dataset_id, rows=original_count)

    cfg = FilterConfig(image_field=image_column)

    kept_indices: list[int] = []
    reason_counter: Counter[str] = Counter()

    for idx in range(len(ds)):
        row = ds[idx]
        keep, reasons, _meta = decide_keep(row, cfg)
        if keep:
            kept_indices.append(idx)
        else:
            for r in reasons:
                reason_counter[r] += 1

    log.info("vlm_filter_done", dataset=dataset_id, kept=len(kept_indices), total=original_count)
    filtered = ds.select(kept_indices)

    payload = json.dumps(
        {
            "dataset": dataset_id,
            "original_rows": original_count,
            "kept": len(filtered),
            "rejected": original_count - len(filtered),
            "rejection_reasons": dict(reason_counter.most_common()),
        },
        indent=2,
    )
    return StepResult(
        payload=payload,
        created=[ArtifactRef("dataset", dataset_id, 0)],
        updates={f"active_dataset:{dataset_id}": ArtifactRef("dataset", dataset_id, 0)},
        metrics={"rows_kept": len(filtered), "rows_rejected": original_count - len(filtered)},
        dataset_updates={
            dataset_id: DatasetUpdate(
                dataset=filtered,
                transform_info={
                    "tool": "vlm_filter",
                    "original": original_count,
                    "kept": len(filtered),
                    "rejected": original_count - len(filtered),
                },
            )
        },
    )
