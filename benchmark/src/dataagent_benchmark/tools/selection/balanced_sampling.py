"""balanced_sampling tool — rebalance sample allocations."""

import json
from collections import Counter
from typing import Annotated, Any

from dataagent_benchmark.domain.tool_context import AllDatasets, FromEnv
from dataagent_benchmark.tools.postselection.utils import (
    _capped_sample,
    _task_balanced_sample,
    _temperature_sample,
)


def _detect_subsets(
    dataset: Any,
    col_stats: dict,
    ds_name: str,
    text_column: str,
) -> dict[str, dict]:
    """Detect subsets from categorical columns in the dataset."""
    total = len(dataset)

    for col, stats in col_stats.items():
        if stats.get("dtype") == "categorical" and col not in (text_column,):
            values = dataset[col]
            counts = Counter(v for v in values if isinstance(v, str) and v.strip())
            if len(counts) >= 2:
                return {val: {"size": cnt, "task_type": "qa"} for val, cnt in counts.items()}

    return {ds_name: {"size": total, "task_type": "qa"}}


def balanced_sampling(
    sample_counts: Annotated[
        str,
        "Comma-separated sample counts aligned with datasets (from mix_ratio output)",
    ],
    method: Annotated[
        str, "Sampling method: 'temperature', 'capped', or 'task_balanced'"
    ] = "temperature",
    alpha: Annotated[float, "Temperature exponent for temperature sampling (0.3-0.8)"] = 0.5,
    cap: Annotated[int, "Max examples per subset for capped sampling"] = 5_000,
    # --- Injected by env ---
    dataset_names: Annotated[list[str], FromEnv("dataset_names")] = None,
    all_working_datasets: Annotated[dict, AllDatasets("working")] = None,
    all_column_stats: Annotated[dict, AllDatasets("column_stats")] = None,
    all_dataset_configs: Annotated[dict, AllDatasets("config")] = None,
) -> str:
    """Rebalance sample allocations so dominant subsets don't overwhelm training.

    Use after mix_ratio.
    """
    names = dataset_names
    counts = [int(c.strip()) for c in sample_counts.split(",")]

    if len(names) != len(counts):
        return json.dumps(
            {"error": f"Mismatch: {len(names)} datasets but {len(counts)} sample counts."}
        )

    if method not in ("temperature", "capped", "task_balanced"):
        return json.dumps(
            {
                "error": (
                    f"Unknown method '{method}'. Use 'temperature', 'capped', or 'task_balanced'."
                )
            }
        )

    total_budget = sum(counts)
    dataset_results = []

    for ds_name, original_count in zip(names, counts, strict=False):
        ds = all_working_datasets.get(ds_name)
        if ds is None:
            dataset_results.append(
                {
                    "dataset": ds_name,
                    "warning": f"Dataset '{ds_name}' not loaded; kept as-is.",
                    "original_count": original_count,
                    "rebalanced_subsets": {ds_name: original_count},
                    "total_rebalanced": original_count,
                }
            )
            continue

        ds_config = all_dataset_configs.get(ds_name)
        text_col = ds_config.columns.text if ds_config and ds_config.columns else "text"
        cs = all_column_stats.get(ds_name, {})
        subsets = _detect_subsets(ds, cs, ds_name, text_col)

        if method == "temperature":
            rebalanced = _temperature_sample(subsets, original_count, alpha)
        elif method == "capped":
            rebalanced = _capped_sample(subsets, original_count, cap)
        else:
            rebalanced = _task_balanced_sample(subsets, original_count)

        dataset_results.append(
            {
                "dataset": ds_name,
                "original_count": original_count,
                "rebalanced_subsets": rebalanced,
                "total_rebalanced": sum(rebalanced.values()),
            }
        )

    output: dict = {
        "method": method,
        "total_budget": total_budget,
        "total_rebalanced": sum(d["total_rebalanced"] for d in dataset_results),
        "datasets": dataset_results,
    }
    if method == "temperature":
        output["alpha"] = alpha
    elif method == "capped":
        output["cap"] = cap

    return json.dumps(output, indent=2)
