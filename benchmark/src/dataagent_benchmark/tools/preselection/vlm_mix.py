"""vlm_mix tool — multi-dataset mixing with sampling."""

import json
from typing import Annotated

import structlog
from datasets import concatenate_datasets

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import AllDatasets, DatasetUpdate, FromEnv
from dataagent_benchmark.infra.vlm_ops import ensure_same_schema, take_examples

log = structlog.get_logger()


def vlm_mix(
    sample_counts: Annotated[
        str, "Comma-separated sample counts per dataset (e.g. '5000,3000,2000')"
    ],
    sample_mode: Annotated[str, "How to select: 'first' or 'random'"] = "random",
    shuffle: Annotated[bool, "Shuffle after concatenation"] = True,
    seed: Annotated[int, "Random seed"] = 42,
    # --- Injected by env ---
    dataset_names: Annotated[list[str], FromEnv("dataset_names")] = None,
    all_working_datasets: Annotated[dict, AllDatasets("working")] = None,
) -> StepResult | str:
    """Sample K rows from each dataset and concatenate into one combined dataset.

    sample_counts must have one entry per dataset (same order as the
    candidate pool). Use -1 or 'all' to take all rows from a dataset.
    """
    log.info("vlm_mix", sample_counts=sample_counts, mode=sample_mode, shuffle=shuffle, seed=seed)
    names = dataset_names
    counts_raw = [c.strip() for c in sample_counts.split(",")]
    if len(counts_raw) != len(names):
        return json.dumps(
            {
                "error": (
                    f"Expected {len(names)} counts (one per dataset: {names}), "
                    f"got {len(counts_raw)}"
                )
            }
        )

    counts: list[int | None] = []
    for c in counts_raw:
        if c.lower() in ("all", "-1"):
            counts.append(None)
        else:
            counts.append(int(c))

    subsets = []
    per_dataset: list[dict] = []

    for name, take in zip(names, counts, strict=True):
        log.info("vlm_mix_dataset", name=name, take=take)
        ds = all_working_datasets[name]
        subset, idx = take_examples(ds, take, sample_mode, seed)
        if len(subset) > 0:
            subsets.append(subset)
        per_dataset.append(
            {
                "dataset": name,
                "original_rows": len(ds),
                "sampled": len(subset),
            }
        )

    if not subsets:
        return json.dumps({"error": "No rows after sampling."})

    log.info("vlm_mix_concat", subsets=len(subsets))
    ensure_same_schema(subsets)
    mixed = concatenate_datasets(subsets)
    if shuffle:
        mixed = mixed.shuffle(seed=seed)

    combined_name = "+".join(names)

    payload = json.dumps(
        {
            "combined_name": combined_name,
            "total_rows": len(mixed),
            "per_dataset": per_dataset,
            "columns": mixed.column_names,
            "shuffled": shuffle,
        },
        indent=2,
    )
    return StepResult(
        payload=payload,
        created=[ArtifactRef("dataset", combined_name, 0)],
        updates={f"active_dataset:{combined_name}": ArtifactRef("dataset", combined_name, 0)},
        metrics={"total_rows": len(mixed)},
        dataset_updates={
            combined_name: DatasetUpdate(
                dataset=mixed,
                transform_info={
                    "tool": "vlm_mix",
                    "sources": per_dataset,
                    "shuffle": shuffle,
                    "seed": seed,
                },
                is_new=True,
            )
        },
    )
