"""Dataclasses describing tool side-effects and injection markers.

Tools return DatasetUpdate inside StepResult so the env can apply mutations
in _commit_step_result() instead of tools mutating the env directly.

The Injected marker hierarchy is used in Annotated type hints to declare
parameters that the env resolves before calling the tool.  Subclasses
encode the resolution strategy so the env dispatches structurally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Injection markers (used inside Annotated[T, ...])
# ---------------------------------------------------------------------------


class Injected:
    """Base marker — hidden from LLM schema, resolved by env.

    Subclass to encode the resolution strategy:

    - ``FromEnv(attr)``      — scalar read from ``env.<attr>``
    - ``PerDataset(key)``    — resolved using ``tool_args["dataset_id"]``
    - ``AllDatasets(key)``   — aggregated across ``env.dataset_names``
    - ``FromArtifact(...)``  — looked up in the artifact registry
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class FromEnv(Injected):
    """Scalar resolved from an env attribute.

    Usage::

        sample_budget: Annotated[int, FromEnv("sample_budget")]
        run_dir: Annotated[Any, FromEnv("run_dir")]
    """

    def __init__(self, attr: str) -> None:
        self.attr = attr

    def __repr__(self) -> str:
        return f"FromEnv({self.attr!r})"


class PerDataset(Injected):
    """Resolved per-dataset using ``tool_args["dataset_id"]``.

    Keys:
        ``"working"``         — working copy from store
        ``"original"``        — original cached dataset
        ``"config"``          — Dataset model from env.datasets
        ``"text_column"``     — text column name from config
        ``"image_column"``    — image column name from config
        ``"column_stats"``    — computed column statistics
        ``"transforms"``      — list of applied transforms
        ``"full_row_count"``  — true row count before limiting
        ``"version_count"``   — number of versions on the stack
        ``"version_summary"`` — compact summary of each version

    Usage::

        dataset: Annotated[Any, PerDataset("working")]
        text_column: Annotated[str, PerDataset("text_column")]
    """

    def __init__(self, key: str) -> None:
        self.key = key

    def __repr__(self) -> str:
        return f"PerDataset({self.key!r})"


class AllDatasets(Injected):
    """Aggregated across all datasets.  Same keys as PerDataset.

    Returns ``dict[str, T]`` mapping dataset name → resolved value.

    Usage::

        all_working: Annotated[dict, AllDatasets("working")]
        all_stats:   Annotated[dict, AllDatasets("column_stats")]
    """

    def __init__(self, key: str) -> None:
        self.key = key

    def __repr__(self) -> str:
        return f"AllDatasets({self.key!r})"


class FromArtifact(Injected):
    """Resolved from the artifact registry.

    Looks up ``registry.get_metadata(kind, tool_args[id_param])``
    and extracts ``metadata[metadata_key]``.  Falls back to the raw
    ID if the artifact hasn't been registered yet.

    Usage::

        adapter_path: Annotated[str, FromArtifact("model", "adapter_path", "model_id")]
    """

    def __init__(self, kind: str, metadata_key: str, id_param: str) -> None:
        self.kind = kind
        self.metadata_key = metadata_key
        self.id_param = id_param

    def __repr__(self) -> str:
        return f"FromArtifact({self.kind!r}, {self.metadata_key!r}, {self.id_param!r})"


# ---------------------------------------------------------------------------
# Dataset mutation descriptor (returned inside StepResult)
# ---------------------------------------------------------------------------


@dataclass
class DatasetUpdate:
    """Describes a dataset mutation for the env to apply."""

    dataset: Any  # HF Dataset (the new working copy)
    transform_info: dict[str, Any] | None = None
    is_new: bool = False  # True when creating a new combined dataset (vlm_mix)
    is_reset: bool = False  # True when resetting to original (transform_dataset reset)
    is_rollback: bool = False  # True when rolling back to a prior version
    rollback_steps: int = 0  # Number of versions to pop
