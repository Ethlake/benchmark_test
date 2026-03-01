"""rollback — undo the last N transforms on a dataset's working copy."""

import json
from typing import Annotated

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import DatasetUpdate, PerDataset


def rollback(
    dataset_id: Annotated[str, "Dataset to rollback"],
    steps: Annotated[int, "Number of versions to undo"] = 1,
    # --- Injected by env ---
    version_summary: Annotated[list, PerDataset("version_summary")] = None,
) -> StepResult | str:
    """Undo the last N transforms on a dataset, restoring a prior version.

    Unlike reset (which discards all transforms), rollback pops only the
    requested number of versions off the stack.  Use this to undo a filter
    or transform that was too aggressive without losing earlier work.
    """
    if not version_summary:
        return json.dumps({"error": "No transforms to rollback.", "dataset": dataset_id})

    steps_to_undo = min(steps, len(version_summary))
    undone = version_summary[-steps_to_undo:]
    remaining = len(version_summary) - steps_to_undo

    payload = json.dumps(
        {
            "dataset": dataset_id,
            "rollback_steps": steps_to_undo,
            "undone": undone,
            "remaining_versions": remaining,
        },
        indent=2,
    )
    return StepResult(
        payload=payload,
        created=[ArtifactRef("dataset", dataset_id, 0)],
        updates={f"active_dataset:{dataset_id}": ArtifactRef("dataset", dataset_id, 0)},
        metrics={"operation": "rollback", "steps_undone": steps_to_undo},
        dataset_updates={
            dataset_id: DatasetUpdate(
                dataset=None, is_rollback=True, rollback_steps=steps_to_undo
            )
        },
    )
