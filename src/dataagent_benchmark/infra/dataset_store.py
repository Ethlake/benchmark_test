"""Dataset store — lazy loading, caching, working copies, stats."""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

log = structlog.get_logger()


@dataclass
class VersionEntry:
    """A single snapshot in the dataset version stack."""

    dataset: Any  # HF Dataset
    transform_info: dict[str, Any] = field(default_factory=dict)
    step: int = 0


class DatasetStore:
    """Manages lazy-loaded HuggingFace datasets with working copies for transforms.

    Working copies are stored as a version stack (``_history``).  Each
    ``set_working`` call pushes a ``VersionEntry``, enabling rollback
    to any prior version without losing intermediate state.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Dataset] = {}
        self._history: dict[str, list[VersionEntry]] = {}
        self._stats_cache: dict[str, dict] = {}
        self._partial: set[str] = set()  # names loaded with column/row filtering
        self._full_row_counts: dict[str, int] = {}  # true row count before limiting
        self._max_rows: dict[str, int] = {}  # row limit used during partial load

    def load(
        self,
        name: str,
        hf_path: str = "",
        split: str = "train",
        hf_name: str | None = None,
        local_path: str = "",
        columns: list[str] | None = None,
        max_rows: int | None = None,
    ) -> Dataset:
        """Load a dataset from HF Hub or local path and cache it by name.

        For local paths, tries ``load_from_disk`` first (arrow format saved
        with ``save_to_disk``), then falls back to ``load_dataset`` (parquet,
        jsonl, csv).  If the result is a DatasetDict, indexes by *split*.

        If *columns* is given, only those columns are kept (via
        ``select_columns``).  This avoids memory-mapping large image
        columns when only metadata is needed.
        """
        log.info("dataset_load_start", name=name, split=split, columns=columns)

        if local_path:
            p = Path(local_path)
            if p.is_dir() and (p / "dataset_dict.json").exists():
                # Load only the requested split directory instead of the
                # entire DatasetDict — avoids reading all splits from disk.
                split_dir = p / split
                if split_dir.is_dir():
                    log.info("dataset_load_split_dir", path=str(split_dir))
                    ds = load_from_disk(str(split_dir))
                else:
                    import json as _json

                    dd_meta = _json.loads((p / "dataset_dict.json").read_text())
                    available = dd_meta.get("splits", list(dd_meta.keys()))
                    raise KeyError(
                        f"Split '{split}' not found at '{local_path}'. Available: {available}"
                    )
            elif p.is_dir():
                split_dir = p / split
                if split_dir.is_dir():
                    log.info("dataset_load_split_dir", path=str(split_dir))
                    ds = load_from_disk(str(split_dir))
                else:
                    log.info("dataset_load_disk", path=local_path)
                    loaded = load_from_disk(local_path)
                    ds = loaded[split] if isinstance(loaded, DatasetDict) else loaded
            else:
                log.info("dataset_load_file", path=local_path)
                ds = load_dataset(local_path, split=split)
        elif hf_path:
            kwargs: dict = {"path": hf_path, "split": split}
            if hf_name is not None:
                kwargs["name"] = hf_name
            log.info("dataset_load_hub", hf_path=hf_path, hf_name=hf_name)
            ds = load_dataset(**kwargs)
        else:
            raise ValueError(f"Dataset '{name}' has neither hf_path nor local_path set.")

        if columns is not None:
            available_cols = ds.column_names
            keep = [c for c in columns if c in available_cols]
            if keep and set(keep) != set(available_cols):
                log.info(
                    "dataset_select_columns",
                    name=name,
                    kept=len(keep),
                    total=len(available_cols),
                    columns=keep,
                )
                ds = ds.select_columns(keep)
                self._partial.add(name)

        self._full_row_counts[name] = len(ds)

        if max_rows is not None and max_rows > 0 and len(ds) > max_rows:
            log.info("dataset_row_limit", name=name, max_rows=max_rows, total_rows=len(ds))
            ds = ds.select(range(max_rows))
            self._partial.add(name)
            self._max_rows[name] = max_rows

        log.info("dataset_loaded", name=name, rows=len(ds), columns=ds.column_names)
        self._cache[name] = ds
        return ds

    def is_loaded(self, name: str, full: bool = False) -> bool:
        """Check if a dataset is loaded.

        If *full* is True, also checks that it was loaded without column
        filtering (i.e. it has all columns, not just a profiling subset).
        """
        if name not in self._cache:
            return False
        if full:
            return name not in self._partial
        return True

    def get(self, name: str) -> Dataset:
        """Return a cached dataset (must be loaded first via load())."""
        if name not in self._cache:
            raise KeyError(f"Dataset '{name}' not loaded. Call store.load() first.")
        return self._cache[name]

    def get_working(self, name: str) -> Dataset:
        """Return the working copy (top of version stack, or original)."""
        stack = self._history.get(name)
        if stack:
            return stack[-1].dataset
        return self.get(name)

    def set_working(
        self,
        name: str,
        ds: Dataset,
        transform_info: dict | None = None,
        step: int = 0,
    ) -> None:
        """Push a new version onto the stack; invalidate stats cache."""
        self._history.setdefault(name, []).append(
            VersionEntry(dataset=ds, transform_info=transform_info or {}, step=step)
        )
        self._stats_cache.pop(name, None)

    def rollback(self, name: str, steps: int = 1) -> dict:
        """Pop *steps* versions off the stack, returning rollback info."""
        stack = self._history.get(name, [])
        actually_popped = min(steps, len(stack))
        popped: list[VersionEntry] = []
        for _ in range(actually_popped):
            popped.append(stack.pop())
        self._stats_cache.pop(name, None)
        current = self.get_working(name)
        return {
            "versions_removed": actually_popped,
            "current_version": len(stack),
            "current_rows": len(current),
            "undone": [e.transform_info for e in reversed(popped)],
        }

    def reset(self, name: str) -> None:
        """Clear the version stack — back to the original cached dataset."""
        self._history.pop(name, None)
        self._stats_cache.pop(name, None)

    def ensure_full(self, name: str) -> int | None:
        """Reload dataset without column filtering if it was partially loaded.

        Preserves any row limit that was used. Returns the max_rows to
        pass to the subsequent load(), or None for no limit.
        """
        if name in self._partial:
            max_rows = self._max_rows.get(name)
            log.info("dataset_ensure_full", name=name, max_rows=max_rows)
            # Drop the partial copy so the next load() gets all columns
            self._cache.pop(name, None)
            self._history.pop(name, None)
            self._partial.discard(name)
            return max_rows
        return None

    def full_row_count(self, name: str) -> int | None:
        """Return the true row count before any row limiting, or None if unknown."""
        return self._full_row_counts.get(name)

    def reset_all(self) -> None:
        """Clear all cached data for a fresh session."""
        self._cache.clear()
        self._history.clear()
        self._stats_cache.clear()
        self._partial.clear()
        self._full_row_counts.clear()
        self._max_rows.clear()

    def transforms_applied(self, name: str) -> list[dict]:
        return [e.transform_info for e in self._history.get(name, [])]

    def version_count(self, name: str) -> int:
        """Return the number of versions on the stack for *name*."""
        return len(self._history.get(name, []))

    def version_summary(self, name: str) -> list[dict]:
        """Return a compact summary of each version for the LLM."""
        return [
            {
                "version": i + 1,
                "rows": len(e.dataset),
                "tool": e.transform_info.get("tool") or e.transform_info.get("operation", "?"),
                "step": e.step,
            }
            for i, e in enumerate(self._history.get(name, []))
        ]

    def column_stats(
        self,
        name: str,
        question_column: str = "question",
        answer_column: str = "answer",
        answer_marker: str | None = None,
    ) -> dict[str, dict]:
        """Compute column-level statistics from the working copy, cached."""
        if name in self._stats_cache:
            return self._stats_cache[name]
        ds = self.get_working(name)
        stats: dict[str, dict] = {}
        for col in ds.column_names:
            values = ds[col]
            total = len(values)
            if total == 0:
                stats[col] = {
                    "dtype": "str",
                    "count": 0,
                    "null_count": 0,
                    "empty_count": 0,
                }
                continue
            sample_for_type = values[:100]
            all_str = all(isinstance(v, str) for v in sample_for_type)
            if all_str:
                unique_count = len(set(values[: min(total, 10_000)]))
                is_categorical = unique_count <= 50 and col not in (
                    question_column,
                    answer_column,
                )
                if is_categorical:
                    null_count = sum(1 for v in values if v is None)
                    empty_count = sum(
                        1
                        for v in values
                        if v is not None and isinstance(v, str) and v.strip() == ""
                    )
                    stats[col] = {
                        "dtype": "categorical",
                        "count": total,
                        "null_count": null_count,
                        "empty_count": empty_count,
                        "unique": unique_count,
                    }
                    continue
                null_count = sum(1 for v in values if v is None)
                empty_count = sum(
                    1 for v in values if v is not None and isinstance(v, str) and v.strip() == ""
                )
                if total > 10_000:
                    rng = random.Random(42)
                    indices = rng.sample(range(total), 10_000)
                    sample_vals = [values[i] for i in indices]
                else:
                    sample_vals = values
                lengths = np.array([len(v) if isinstance(v, str) else 0 for v in sample_vals])
                col_stats: dict = {
                    "dtype": "str",
                    "count": total,
                    "null_count": null_count,
                    "empty_count": empty_count,
                    "len_mean": round(float(np.mean(lengths)), 1),
                    "len_std": round(float(np.std(lengths)), 1),
                    "len_min": int(np.min(lengths)),
                    "len_max": int(np.max(lengths)),
                    "len_p25": int(np.percentile(lengths, 25)),
                    "len_p50": int(np.percentile(lengths, 50)),
                    "len_p75": int(np.percentile(lengths, 75)),
                }
                patterns: dict[str, float] = {}
                if answer_marker and col == answer_column:
                    marker_count = sum(
                        1 for v in sample_vals if isinstance(v, str) and answer_marker in v
                    )
                    patterns[answer_marker] = round(marker_count / len(sample_vals), 3)
                if col == answer_column:
                    latex_count = sum(
                        1 for v in sample_vals if isinstance(v, str) and ("$" in v or "\\frac" in v)
                    )
                    if latex_count > len(sample_vals) * 0.05:
                        patterns["$"] = round(latex_count / len(sample_vals), 3)
                col_stats["patterns"] = patterns
                stats[col] = col_stats
            else:
                stats[col] = {"dtype": "other", "count": total}
        self._stats_cache[name] = stats
        return stats

    def sample_rows(self, name: str, n: int = 5) -> list[dict]:
        """Return n random rows from the working copy as dicts."""
        ds = self.get_working(name)
        total = len(ds)
        n = min(n, total)
        if n <= 0:
            return []
        rng = random.Random(42)
        indices = rng.sample(range(total), n)
        return [dict(ds[i]) for i in indices]
