"""Core artifact types for versioned tracking of curation outputs.

ArtifactRef identifies a versioned artifact (dataset, recipe, model, etc.).
StepResult is returned by tools to declare what they created/updated.
TOOL_META provides static metadata about each tool's phase and capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataagent_benchmark.domain.models import DataRecipe
    from dataagent_benchmark.domain.tool_context import DatasetUpdate


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to a versioned artifact.

    ``version=0`` is a placeholder — the registry assigns a real
    monotonic version when the artifact is registered.
    """

    kind: str  # "dataset", "recipe", "model", "eval_result", "saved_file"
    id: str  # human name, e.g. "finevision5"
    version: int = 0  # 0 = placeholder, assigned by registry

    def __str__(self) -> str:
        return f"{self.kind}:{self.id}@v{self.version}"


@dataclass
class StepResult:
    """Structured return value from a tool execution.

    Tools return this instead of a plain ``str`` to declare artifacts
    created, pointer updates, and metrics.  ``env.step()`` processes
    the result: registering artifacts, committing pointer updates,
    and building the observation with a state header.

    Tools that haven't been migrated yet continue returning ``str`` —
    ``env.step()`` handles both types via ``isinstance`` check.
    """

    payload: str  # JSON the LLM sees
    created: list[ArtifactRef] = field(default_factory=list)
    updates: dict[str, ArtifactRef | None] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    # Dataset mutations for env to apply (tool declares, env executes)
    dataset_updates: dict[str, DatasetUpdate] = field(default_factory=dict)
    # Recipe for env to set as current_recipe
    recipe: DataRecipe | None = None


# ---------------------------------------------------------------------------
# Static tool metadata (phase, requirements, capabilities)
# ---------------------------------------------------------------------------

TOOL_META: dict[str, dict] = {
    # Preselection
    "profile_datasets": {
        "phase": "preselection",
        "requires": [],
        "gpu": False,
        "mutates_store": True,
    },
    "load_dataset": {
        "phase": "preselection",
        "requires": [],
        "gpu": False,
        "mutates_store": True,
    },
    "inspect_samples": {
        "phase": "preselection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    "compute_mix_ratio": {
        "phase": "preselection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    "vlm_filter": {
        "phase": "preselection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": True,
    },
    "quality_filter": {
        "phase": "preselection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": True,
    },
    "vlm_mix": {
        "phase": "preselection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": True,
    },
    "vlm_modify": {
        "phase": "preselection",
        "requires": ["dataset"],
        "gpu": True,
        "mutates_store": True,
    },
    "analyze_requirements": {
        "phase": "preselection",
        "requires": [],
        "gpu": False,
        "mutates_store": False,
    },
    # Selection
    "integrity_check": {
        "phase": "selection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    "text_quality_check": {
        "phase": "selection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    "dedup_check": {
        "phase": "selection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    "explore_dataset": {
        "phase": "selection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    "transform_dataset": {
        "phase": "selection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": True,
    },
    "balanced_sampling": {
        "phase": "selection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    # Postselection
    "save_to_disk": {
        "phase": "postselection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    "save_to_hf": {
        "phase": "postselection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": False,
    },
    "convert_format": {
        "phase": "postselection",
        "requires": ["dataset"],
        "gpu": False,
        "mutates_store": True,
    },
    "submit_finetune": {
        "phase": "postselection",
        "requires": ["dataset"],
        "gpu": True,
        "mutates_store": False,
    },
    "submit_eval": {
        "phase": "postselection",
        "requires": ["model"],
        "gpu": True,
        "mutates_store": False,
    },
    # Utilities
    "think": {
        "phase": "utility",
        "requires": [],
        "gpu": False,
        "mutates_store": False,
    },
}
