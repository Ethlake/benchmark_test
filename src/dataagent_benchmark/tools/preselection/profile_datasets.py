"""profile_datasets tool — load and profile all candidate datasets at once."""

import json
import random
from typing import Annotated

import numpy as np
import structlog

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import AllDatasets, FromEnv

log = structlog.get_logger()

# Columns that contain quality ratings (float scores).
_RATING_COLUMNS = (
    "relevance_ratings",
    "image_correspondence_ratings",
    "visual_dependency_ratings",
    "formatting_ratings",
)

_MIN_COLUMNS = (
    "relevance_min",
    "image_correspondence_min",
    "visual_dependency_min",
    "formatting_min",
)


def _random_indices(total: int, n: int, seed: int) -> list[int]:
    """Return n random indices from [0, total), or all if n >= total."""
    if n >= total:
        return list(range(total))
    rng = random.Random(seed)
    return rng.sample(range(total), n)


def _compute_quality_score(ds, rating_cols: list[str], sample_n: int = 0, seed: int = 42) -> dict:
    """Compute per-rating-column means and an overall quality score.

    If sample_n > 0, randomly sample that many rows instead of using all.
    """
    total = len(ds)
    indices = _random_indices(total, sample_n, seed) if 0 < sample_n < total else None

    scores: dict[str, float] = {}
    for col in rating_cols:
        values = ds[col]
        if indices is not None:
            values = [values[i] for i in indices]
        flat: list[float] = []
        for v in values:
            if isinstance(v, list):
                flat.extend(float(x) for x in v if x is not None)
            elif v is not None:
                flat.append(float(v))
        if flat:
            scores[col] = round(float(np.mean(flat)), 4)
    if scores:
        scores["overall"] = round(float(np.mean(list(scores.values()))), 4)
    return scores


def _compute_text_stats(ds, text_col: str, sample_n: int = 0, seed: int = 42) -> dict:
    """Compute basic text statistics from the text column.

    Works with both plain strings and list-of-dict conversation formats.
    """
    total = len(ds)
    if text_col not in ds.column_names:
        return {}

    if sample_n > 0 and sample_n < total:
        indices = _random_indices(total, sample_n, seed)
    else:
        indices = list(range(min(total, 5000)))  # cap at 5k for speed

    turn_counts: list[int] = []
    char_counts: list[int] = []

    for idx in indices:
        val = ds[idx][text_col]
        if val is None:
            turn_counts.append(0)
            char_counts.append(0)
        elif isinstance(val, list):
            # Conversation format: list of dicts or list of strings
            turn_counts.append(len(val))
            total_chars = 0
            for turn in val:
                if isinstance(turn, dict):
                    total_chars += sum(len(str(v)) for v in turn.values())
                elif isinstance(turn, str):
                    total_chars += len(turn)
            char_counts.append(total_chars)
        elif isinstance(val, str):
            turn_counts.append(1)
            char_counts.append(len(val))

    if not char_counts:
        return {}

    turns_arr = np.array(turn_counts)
    chars_arr = np.array(char_counts)
    return {
        "avg_turns": round(float(np.mean(turns_arr)), 1),
        "min_turns": int(np.min(turns_arr)),
        "max_turns": int(np.max(turns_arr)),
        "avg_chars": round(float(np.mean(chars_arr)), 0),
        "min_chars": int(np.min(chars_arr)),
        "max_chars": int(np.max(chars_arr)),
        "rows_sampled_for_stats": len(indices),
    }


def profile_datasets(
    sample_n: Annotated[
        int,
        "Number of rows to sample per dataset for scoring (0 = all rows)",
    ] = 0,
    seed: Annotated[int, "Random seed for sampling"] = 42,
    # --- Injected by env ---
    dataset_names: Annotated[list[str], FromEnv("dataset_names")] = None,
    all_dataset_configs: Annotated[dict, AllDatasets("config")] = None,
    all_working_datasets: Annotated[dict, AllDatasets("working")] = None,
    profiling_sample_n: Annotated[int, FromEnv("profiling_sample_n")] = 0,
    sample_budget: Annotated[int, FromEnv("sample_budget")] = 100_000,
    all_full_row_counts: Annotated[dict, AllDatasets("full_row_count")] = None,
) -> StepResult:
    """Load all candidate datasets and compute quality profiles.

    Returns per-dataset statistics including row counts, columns,
    and quality scores derived from existing metadata.
    """
    # Config override wins over LLM-chosen sample_n
    if profiling_sample_n > 0:
        sample_n = profiling_sample_n

    log.info("profile_datasets", sample_n=sample_n, seed=seed)
    names = dataset_names
    if not names:
        return json.dumps({"error": "No datasets in candidate pool."})

    profiles: list[dict] = []
    created_refs: list[ArtifactRef] = []
    pointer_updates: dict[str, ArtifactRef] = {}

    for name in names:
        log.info("profile_dataset", name=name)
        ds_model = all_dataset_configs[name]

        # Include the text column so we can compute text stats
        text_col = ds_model.columns.text if ds_model.columns else None

        ds = all_working_datasets[name]
        columns = ds.column_names
        num_rows = all_full_row_counts.get(name) or len(ds)

        # Declare initial dataset artifact + pointer via StepResult
        ref = ArtifactRef("dataset", name, 0)
        created_refs.append(ref)
        pointer_updates[f"active_dataset:{name}"] = ref

        # Detect which rating columns exist
        rating_cols = [c for c in _RATING_COLUMNS if c in columns]
        min_cols = [c for c in _MIN_COLUMNS if c in columns]

        log.info(
            "profile_stats", name=name, rows=num_rows, rating_cols=rating_cols, text_col=text_col
        )
        quality_scores = (
            _compute_quality_score(ds, rating_cols, sample_n=sample_n, seed=seed)
            if rating_cols
            else {}
        )

        # Text statistics — always computed if text column is available
        text_stats: dict = {}
        if text_col and text_col in columns:
            log.info("profile_text_stats", name=name, col=text_col)
            text_stats = _compute_text_stats(ds, text_col, sample_n=sample_n, seed=seed)

        # Min thresholds (if present) — random sample
        min_thresholds: dict[str, float] = {}
        for col in min_cols:
            vals = ds[col]
            sample_idx = _random_indices(len(vals), min(100, len(vals)), seed)
            sample = [float(vals[i]) for i in sample_idx if vals[i] is not None]
            if sample:
                min_thresholds[col] = round(float(np.mean(sample)), 4)

        rows_sampled = min(sample_n, num_rows) if sample_n > 0 else num_rows

        # Report all columns from the original dataset schema (not just loaded ones)
        all_columns = []
        if ds_model.columns:
            if ds_model.columns.text:
                all_columns.append(ds_model.columns.text)
            if ds_model.columns.image:
                all_columns.append(ds_model.columns.image)
        if not all_columns:
            all_columns = columns

        profile: dict = {
            "dataset": name,
            "num_rows": num_rows,
            "rows_sampled": rows_sampled,
            "columns": all_columns,
            "source": ds_model.source.local_path or ds_model.source.hf_path,
        }
        if quality_scores:
            profile["quality_scores"] = quality_scores
        if text_stats:
            profile["text_stats"] = text_stats
        if min_thresholds:
            profile["min_thresholds"] = min_thresholds

        profiles.append(profile)

    # Summary for the LLM
    total_rows = sum(p["num_rows"] for p in profiles)
    quality_summary = {}
    for p in profiles:
        qs = p.get("quality_scores", {})
        if "overall" in qs:
            quality_summary[p["dataset"]] = qs["overall"]

    payload = json.dumps(
        {
            "num_datasets": len(profiles),
            "total_rows": total_rows,
            "sample_budget": sample_budget,
            "sample_n": sample_n if sample_n > 0 else "all",
            "seed": seed,
            "datasets": profiles,
            "quality_summary": quality_summary,
        },
        indent=2,
    )
    return StepResult(
        payload=payload,
        created=created_refs,
        updates=pointer_updates,
        metrics={"num_datasets": len(profiles), "total_rows": total_rows},
    )
