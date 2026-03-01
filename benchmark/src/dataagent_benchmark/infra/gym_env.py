"""Gymnasium adapter — wraps curation tools into the Gym Env protocol.

The env owns all episode state.  Tools declare dependencies via
``Annotated[T, FromEnv/PerDataset/AllDatasets/FromArtifact]`` markers —
the env resolves them before each call so tools are pure functions with
no reference to the env.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import structlog
from gymnasium import spaces

from dataagent_benchmark.domain.artifact_registry import ArtifactRegistry
from dataagent_benchmark.domain.artifacts import TOOL_META, ArtifactRef, StepResult
from dataagent_benchmark.domain.models import DataRecipe
from dataagent_benchmark.domain.tool_context import (
    AllDatasets,
    FromArtifact,
    FromEnv,
    Injected,
    PerDataset,
)
from dataagent_benchmark.domain.training_dataset import Dataset
from dataagent_benchmark.infra.config import load_config
from dataagent_benchmark.infra.dataset_store import DatasetStore

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Per-dataset resolver table — shared by PerDataset and AllDatasets markers
# ---------------------------------------------------------------------------

_PER_DATASET: dict[str, Callable[..., Any]] = {
    "working": lambda env, ds_name: env.store.get_working(ds_name),
    "original": lambda env, ds_name: env.store.get(ds_name),
    "config": lambda env, ds_name: env.datasets[ds_name],
    "text_column": lambda env, ds_name: env.datasets[ds_name].columns.text,
    "image_column": lambda env, ds_name: env.datasets[ds_name].columns.image,
    "column_stats": lambda env, ds_name: env.store.column_stats(
        ds_name,
        question_column=env.datasets[ds_name].columns.text,
        answer_column=env.datasets[ds_name].columns.text,
    ),
    "transforms": lambda env, ds_name: env.store.transforms_applied(ds_name),
    "full_row_count": lambda env, ds_name: env.store.full_row_count(ds_name),
    "version_count": lambda env, ds_name: env.store.version_count(ds_name),
    "version_summary": lambda env, ds_name: env.store.version_summary(ds_name),
}


def _find_repo_root() -> Path:
    """Walk up from this file looking for the directory that contains ``packages/``."""
    current = Path(__file__).resolve().parent
    for ancestor in (current, *current.parents):
        if (ancestor / "packages").is_dir():
            return ancestor
    return Path(__file__).resolve().parents[4]


def _default_output_base() -> Path:
    """Return the base output directory, respecting ``DATAAGENT_OUTPUT_DIR``."""
    env = os.environ.get("DATAAGENT_OUTPUT_DIR")
    return Path(env) if env else _find_repo_root() / "output"


class CurationEnv(gym.Env):
    """Gymnasium environment wrapping the data curation tool suite.

    All episode state lives here.  Tools declare dependencies via
    ``Annotated[T, FromEnv/PerDataset/AllDatasets/FromArtifact]``
    markers and the env resolves them structurally before each dispatch.

    observation: Dict with type, payload, step, reward_so_far
    action:      JSON string — {"tool": "<name>", "args": {<kwargs>}}
    reward:      0 for intermediate steps, eval accuracy on submit_eval
    terminated:  True after submit_eval returns
    truncated:   True when max_steps exceeded
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task_path: str,
        tool_names: list[str] | None = None,
        env_config: dict | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.agent_config, loaded_env_config, self.task = load_config(task_path)
        # Explicit env_config overrides what was loaded from YAML
        env_config = {**loaded_env_config, **(env_config or {})}
        self.max_steps = env_config.get("max_steps", 50)
        self.episodes = env_config.get("episodes", 1)
        self._env_config = env_config
        self.render_mode = render_mode

        # --- Task config (populated in reset via from_task) ---
        self.task_description: str = ""
        self.target_model: str = ""
        self.sample_budget: int = 100_000
        self.profiling_sample_n: int = 500
        self.output_path: str = ""
        self.output_base: Path = _default_output_base()
        self.run_dir: Path | None = None
        self.training_config: dict = {}
        self.eval_config: dict = {}
        self.datasets: dict[str, Dataset] = {}
        self.dataset_names: list[str] = []

        # --- Mutable episode state ---
        self.current_recipe: DataRecipe | None = None

        # --- Env-scoped dataset store ---
        self.store: DatasetStore = DatasetStore()

        # --- Artifact registry + pointers ---
        self.registry: ArtifactRegistry = ArtifactRegistry()
        self._pointers: dict[str, ArtifactRef | None] = {
            "active_recipe": None,
            "active_model": None,
            "active_eval": None,
        }
        self._event_log: list[dict[str, Any]] = []

        # --- Tools ---
        # _tools: FunctionTool dict (for LLM schema introspection via .tools property)
        # _raw_fns: raw callable dict (for env.step() to call directly)
        # _injected_cache: {tool_name: {param_name: Injected}} for resolver dispatch
        from dataagent_benchmark.tools import make_tools

        self._tools, self._raw_fns, self._injected_cache = make_tools(tool_names)

        # --- Gymnasium spaces ---
        self.observation_space = spaces.Dict(
            {
                "type": spaces.Text(min_length=1, max_length=32),
                "payload": spaces.Text(min_length=0, max_length=64_000),
                "step": spaces.Discrete(self.max_steps + 1),
                "reward_so_far": spaces.Box(low=-100.0, high=100.0, shape=(), dtype=np.float32),
            }
        )
        self.action_space = spaces.Text(min_length=1, max_length=8_000)

        # --- Episode tracking ---
        self._step_count = 0
        self._history: list[dict] = []
        self._terminated = False
        self._cumulative_reward = 0.0

    # ------------------------------------------------------------------
    # State management (moved from SessionContext)
    # ------------------------------------------------------------------

    def from_task(self, task: dict) -> None:
        """Parse a task dict into domain-model attributes on self."""
        self.task_description = task.get("task_description", "")
        self.target_model = task.get("target_model", "")
        self.sample_budget = task.get("sample_budget", 100_000)

        # output_path comes from env_config; fall back to task for compat
        raw_output = self._env_config.get(
            "output_path",
            task.get("output_path", "curated_dataset.parquet"),
        )
        if self.run_dir is not None:
            curated_dir = self.run_dir / "curated"
            curated_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = str(curated_dir / Path(raw_output).name)
        else:
            self.output_path = raw_output

        self.profiling_sample_n = task.get("profiling_sample_n", 0)
        self.training_config = task.get("training_config", {})

        # Parse datasets -> Dataset instances
        self.datasets = {}
        self.dataset_names = []
        for entry in task.get("datasets", []):
            if isinstance(entry, dict):
                ds = Dataset(**entry)
            else:
                continue
            self.datasets[ds.name] = ds
            self.dataset_names.append(ds.name)

        # Parse eval — supports both dict and legacy list formats
        eval_raw = task.get("eval")
        self.eval_config = {}
        if isinstance(eval_raw, dict):
            self.eval_config = eval_raw
        elif isinstance(eval_raw, list):
            self.eval_config = {"benchmarks": [e["name"] for e in eval_raw if isinstance(e, dict)]}

    def init_run(self, task: dict | None = None, *, run_id: str | None = None) -> Path:
        """Create a run directory and optionally parse task config."""
        if run_id is None:
            run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S_%f")
        self.run_dir = self.output_base / "dataagent" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if task is not None:
            self.from_task(task)
        return self.run_dir

    def get_dataset(self, name: str) -> Dataset:
        """Return dataset by name. Raises KeyError with available names."""
        if name not in self.datasets:
            raise KeyError(f"Unknown dataset '{name}'. Available: {self.dataset_names}")
        return self.datasets[name]

    def subdir(self, name: str) -> Path:
        """Return (and create) a named subdirectory under ``run_dir``."""
        if self.run_dir is None:
            raise RuntimeError("run_dir not set; call init_run() first")
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Tool preparation + artifact helpers
    # ------------------------------------------------------------------

    def _prepare_for_tool(self, tool_name: str, tool_args: dict) -> None:
        """Load/reload datasets as needed before tool execution.

        Ensures datasets are in the store before resolvers try to access
        them.  Also handles the partial→full reload for GPU/mutating tools.
        """
        meta = TOOL_META.get(tool_name, {})

        # profile_datasets needs all datasets loaded (partial — text cols only)
        if tool_name == "profile_datasets":
            for name in self.dataset_names:
                if not self.store.is_loaded(name):
                    ds_model = self.datasets[name]
                    text_cols = [ds_model.columns.text] if ds_model.columns else None
                    self.store.load(
                        name=name,
                        hf_path=ds_model.source.hf_path,
                        split=ds_model.source.split,
                        hf_name=ds_model.source.hf_name,
                        local_path=ds_model.source.local_path,
                        columns=text_cols,
                        max_rows=self.profiling_sample_n or None,
                    )
            return

        # load_dataset loads the specific dataset fully
        if tool_name == "load_dataset":
            ds_name = tool_args.get("dataset_id")
            if ds_name and not self.store.is_loaded(ds_name):
                ds_model = self.datasets[ds_name]
                self.store.load(
                    name=ds_name,
                    hf_path=ds_model.source.hf_path,
                    split=ds_model.source.split,
                    hf_name=ds_model.source.hf_name,
                    local_path=ds_model.source.local_path,
                )
            return

        # Tools that mutate the store or need GPU (images) require full data
        if meta.get("mutates_store") or meta.get("gpu"):
            budget_limit = self.sample_budget or 0
            ds_name = tool_args.get("dataset_id")
            if ds_name and not self.store.is_loaded(ds_name, full=True):
                log.info("prepare_for_tool_reload", tool=tool_name, dataset=ds_name)
                ds_model = self.get_dataset(ds_name)
                prior_limit = self.store.ensure_full(ds_name) or 0
                max_rows = max(prior_limit, budget_limit) or None
                self.store.load(
                    name=ds_model.name,
                    hf_path=ds_model.source.hf_path,
                    split=ds_model.source.split,
                    hf_name=ds_model.source.hf_name,
                    local_path=ds_model.source.local_path,
                    max_rows=max_rows,
                )

            # vlm_mix iterates all datasets, not just one
            if tool_name == "vlm_mix":
                for name in self.dataset_names:
                    if not self.store.is_loaded(name, full=True):
                        log.info("prepare_for_tool_reload", tool=tool_name, dataset=name)
                        ds_model = self.get_dataset(name)
                        prior_limit = self.store.ensure_full(name) or 0
                        max_rows = max(prior_limit, budget_limit) or None
                        self.store.load(
                            name=ds_model.name,
                            hf_path=ds_model.source.hf_path,
                            split=ds_model.source.split,
                            hf_name=ds_model.source.hf_name,
                            local_path=ds_model.source.local_path,
                            max_rows=max_rows,
                        )

    def _commit_step_result(self, tool_name: str, result: StepResult) -> None:
        """Register created artifacts, commit pointer updates, append event."""
        # Register created artifacts (replace placeholder version=0 with real)
        resolved_created: list[ArtifactRef] = []
        for ref in result.created:
            real_ref = self.registry.register(ref.kind, ref.id, result.metrics)
            resolved_created.append(real_ref)

        # Build a mapping from placeholder -> real for pointer resolution
        placeholder_to_real: dict[tuple[str, str], ArtifactRef] = {}
        for orig, real in zip(result.created, resolved_created, strict=True):
            placeholder_to_real[(orig.kind, orig.id)] = real

        # Commit pointer updates
        for pointer_name, ref in result.updates.items():
            if ref is None:
                self._pointers[pointer_name] = None
            elif ref.version == 0:
                # Resolve placeholder
                real = placeholder_to_real.get((ref.kind, ref.id))
                self._pointers[pointer_name] = real if real is not None else ref
            else:
                self._pointers[pointer_name] = ref

        # Apply dataset updates from tool
        for name, update in result.dataset_updates.items():
            if update.is_rollback:
                self.store.rollback(name, update.rollback_steps)
            elif update.is_reset:
                self.store.reset(name)
            elif update.is_new:
                self.store._cache[name] = update.dataset
                self.store.set_working(
                    name, update.dataset, update.transform_info, step=self._step_count
                )
            else:
                self.store.set_working(
                    name, update.dataset, update.transform_info, step=self._step_count
                )

        # Apply recipe update
        if result.recipe is not None:
            self.current_recipe = result.recipe

        # Append event
        self._event_log.append(
            {
                "step": self._step_count,
                "tool": tool_name,
                "created": [str(r) for r in resolved_created],
                "pointer_updates": {k: str(v) if v else None for k, v in result.updates.items()},
                "metrics": result.metrics,
            }
        )

    def _build_state_header(self) -> dict[str, Any]:
        """Build a state header for observations."""
        # Always show all task datasets — version 1 = original, increments with transforms
        active_datasets: dict[str, Any] = {}
        for ds_name in self.dataset_names:
            vc = self.store.version_count(ds_name)
            version = vc + 1  # v1 = original, v2 = first transform, etc.
            try:
                rows = len(self.store.get_working(ds_name))
            except KeyError:
                rows = "not loaded"
            active_datasets[ds_name] = {
                "id": f"{ds_name}@v{version}",
                "rows": rows,
                "version": version,
            }
        # Include any extra datasets created by tools (e.g. vlm_mix combined)
        for key, ref in self._pointers.items():
            if key.startswith("active_dataset:") and ref is not None:
                ds_name = key.split(":", 1)[1]
                if ds_name not in active_datasets:
                    vc = self.store.version_count(ds_name)
                    version = vc + 1
                    try:
                        rows = len(self.store.get_working(ds_name))
                    except KeyError:
                        rows = "?"
                    active_datasets[ds_name] = {
                        "id": f"{ds_name}@v{version}",
                        "rows": rows,
                        "version": version,
                    }

        return {
            "state": {
                "active_datasets": active_datasets,
                "active_recipe": str(self._pointers.get("active_recipe"))
                if self._pointers.get("active_recipe")
                else None,
                "active_model": str(self._pointers.get("active_model"))
                if self._pointers.get("active_model")
                else None,
                "active_eval": str(self._pointers.get("active_eval"))
                if self._pointers.get("active_eval")
                else None,
            },
            "budget": {
                "steps_remaining": self.max_steps - self._step_count,
                "sample_budget": self.sample_budget,
            },
        }

    # ------------------------------------------------------------------
    # Injected parameter resolution
    # ------------------------------------------------------------------

    def _resolve_injected(self, tool_name: str, tool_args: dict) -> dict[str, Any]:
        """Resolve Injected parameters for a tool call.

        Uses the ``_injected_cache`` (built by ``make_tools``) to find
        which parameters need injection, then dispatches structurally
        based on the marker subclass (FromEnv, PerDataset, AllDatasets,
        FromArtifact).
        """
        injected_map = self._injected_cache.get(tool_name, {})
        if not injected_map:
            return {}

        resolved: dict[str, Any] = {}
        for param_name, marker in injected_map.items():
            resolved[param_name] = self._resolve_one(marker, tool_args)
        return resolved

    def _resolve_one(self, marker: Injected, tool_args: dict) -> Any:
        """Resolve a single Injected marker to a concrete value."""
        if isinstance(marker, FromArtifact):
            artifact_id = tool_args.get(marker.id_param, "")
            meta = self.registry.get_metadata(marker.kind, artifact_id)
            if not meta:
                registered = [
                    f"{k[1]}@v{v}"
                    for k, v in self.registry._versions.items()
                    if k[0] == marker.kind
                ]
                raise ValueError(
                    f"Unknown {marker.kind} artifact '{artifact_id}'. "
                    f"Pass the job_id returned by the previous tool, not a filesystem path. "
                    f"Registered {marker.kind} artifacts: {registered or 'none yet'}"
                )
            return meta[marker.metadata_key]

        if isinstance(marker, PerDataset):
            ds_name = tool_args["dataset_id"]
            resolver = _PER_DATASET[marker.key]
            return resolver(self, ds_name)

        if isinstance(marker, AllDatasets):
            resolver = _PER_DATASET[marker.key]
            return {n: resolver(self, n) for n in self.dataset_names}

        if isinstance(marker, FromEnv):
            val = getattr(self, marker.attr)
            # Return a copy of mutable containers to prevent tool-side mutation
            if isinstance(val, list):
                return list(val)
            if isinstance(val, dict):
                return dict(val)
            return val

        raise TypeError(f"Unknown Injected subclass: {type(marker).__name__}({marker!r})")

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[Any]:
        """Expose FunctionTool instances for the LLM client to introspect."""
        return list(self._tools.values())

    def _make_obs(self, obs_type: str, payload: str) -> dict:
        """Build a Dict observation matching self.observation_space."""
        return {
            "type": obs_type,
            "payload": payload,
            "step": self._step_count,
            "reward_so_far": np.float32(self._cumulative_reward),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        del options
        super().reset(seed=seed)

        # Clear previous episode state
        self.store.reset_all()
        self.current_recipe = None
        self.registry.clear()
        self._pointers = {
            "active_recipe": None,
            "active_model": None,
            "active_eval": None,
        }
        self._event_log = []

        # Parse task config into domain attrs
        self.from_task(self.task)

        self._step_count = 0
        self._history = []
        self._terminated = False
        self._cumulative_reward = 0.0

        state_header = self._build_state_header()
        payload = json.dumps(
            {
                "type": "task",
                **state_header,
                "task_description": self.task["task_description"],
                "target_model": self.task["target_model"],
                "datasets": self.task.get("datasets", []),
                "sample_budget": self.task.get("sample_budget", 10_000),
                "eval": self.task.get("eval", {}),
            }
        )
        obs = self._make_obs("task", payload)
        info = {"step": 0, "available_tools": list(self._tools.keys())}
        return obs, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("Episode already terminated. Call reset().")

        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        reward = 0.0

        try:
            parsed = json.loads(action)
            tool_name = parsed["tool"]
            tool_args = parsed.get("args", {})
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            payload = json.dumps(
                {
                    "type": "error",
                    **self._build_state_header(),
                    "error": (
                        f"Invalid action format: {error}. Expected JSON with 'tool' "
                        "and 'args' keys."
                    ),
                    "example": (
                        '{"tool": "analyze_requirements", "args": {"task_description": "..."}}'
                    ),
                }
            )
            obs = self._make_obs("error", payload)
            info = {"step": self._step_count, "tool": None, "valid": False}
            reward = -0.1
            self._cumulative_reward += reward
            return obs, reward, False, truncated, info

        raw_fn = self._raw_fns.get(tool_name)
        if raw_fn is None:
            payload = json.dumps(
                {
                    "type": "error",
                    **self._build_state_header(),
                    "error": f"Unknown tool '{tool_name}'.",
                    "available_tools": list(self._tools.keys()),
                }
            )
            obs = self._make_obs("error", payload)
            info = {"step": self._step_count, "tool": tool_name, "valid": False}
            reward = -0.1
            self._cumulative_reward += reward
            return obs, reward, False, truncated, info

        try:
            # Prepare env state (ensure_full, load) before tool dispatch
            self._prepare_for_tool(tool_name, tool_args)

            # Resolve injected params and merge with LLM-provided args
            injected_kwargs = self._resolve_injected(tool_name, tool_args)
            call_kwargs = {**tool_args, **injected_kwargs}

            # Call the raw tool function directly — gets StepResult | str back
            # without FunctionTool stringifying the return value.
            result = raw_fn(**call_kwargs)

            if isinstance(result, StepResult):
                self._commit_step_result(tool_name, result)
                result_str = result.payload
            else:
                result_str = str(result)
        except Exception as error:
            result_str = json.dumps({"type": "error", "error": str(error)})
            reward = -0.1

        self._history.append(
            {
                "step": self._step_count,
                "tool": tool_name,
                "args": tool_args,
                "result_preview": result_str[:500],
            }
        )

        terminated = False
        is_valid = True
        obs_type = "tool_result"
        if tool_name == "submit_eval":
            terminated = True
            self._terminated = True
            try:
                eval_result = json.loads(result_str)
                if eval_result.get("status") == "failed":
                    is_valid = False
                    reward = -1.0
                    log.error(
                        "system_error",
                        step=self._step_count,
                        tool=tool_name,
                        job_id=eval_result.get("job_id"),
                        error=eval_result.get("error", "submit_eval failed"),
                    )
                    obs_type = "error"
                if is_valid:
                    if "results" in eval_result:
                        reward = float(eval_result["results"].get("accuracy", 0.0))
                    elif "accuracy" in eval_result:
                        reward = float(eval_result.get("accuracy", 0.0))
            except (json.JSONDecodeError, ValueError):
                is_valid = False
                reward = -1.0
                obs_type = "error"
                log.error(
                    "system_error",
                    step=self._step_count,
                    tool=tool_name,
                    error="submit_eval returned non-JSON or invalid payload",
                )

        self._cumulative_reward += reward

        state_header = self._build_state_header()
        payload = json.dumps(
            {"type": obs_type, **state_header, "tool": tool_name, "result": result_str}
        )
        obs = self._make_obs(obs_type, payload)
        info = {
            "step": self._step_count,
            "tool": tool_name,
            "valid": is_valid,
            "history_length": len(self._history),
        }

        if self.render_mode == "human":
            self._render_step(tool_name, tool_args, result_str, reward)

        return obs, reward, terminated, truncated, info

    async def astep(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """Async wrapper for step() to support async callers."""
        return await asyncio.to_thread(self.step, action)

    def _render_step(
        self,
        tool_name: str,
        tool_args: dict,
        result: str,
        reward: float,
    ) -> None:
        # Logged at debug level — the agent loop emits its own env_step at info.
        log.debug(
            "env_step_internal",
            step=self._step_count,
            tool=tool_name,
            args=json.dumps(tool_args)[:80],
            result_preview=result[:200],
            reward=reward,
        )

    def get_history(self) -> list[dict]:
        """Return the full tool call history for this episode."""
        return list(self._history)
