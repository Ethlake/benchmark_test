"""load_dataset tool — load a HuggingFace dataset into the store."""

import json
from typing import Annotated, Any

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import PerDataset


def load_dataset(
    dataset_id: Annotated[str, "Dataset name from the candidate pool"],
    # --- Injected by env ---
    dataset_config: Annotated[Any, PerDataset("config")] = None,
    working_dataset: Annotated[Any, PerDataset("working")] = None,
) -> StepResult:
    """Load a HuggingFace dataset into the working store so other tools can use it."""
    ds = working_dataset
    source = dataset_config.source.local_path or dataset_config.source.hf_path
    result = {
        "dataset": dataset_config.name,
        "source": source,
        "split": dataset_config.source.split,
        "num_rows": len(ds),
        "columns": ds.column_names,
    }
    return StepResult(
        payload=json.dumps(result, indent=2),
        created=[ArtifactRef("dataset", dataset_config.name, 0)],
        updates={
            f"active_dataset:{dataset_config.name}": ArtifactRef("dataset", dataset_config.name, 0)
        },
        metrics={"num_rows": len(ds)},
    )
