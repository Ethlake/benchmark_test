"""Task config loader for the recipe-first benchmark."""

from __future__ import annotations

from pathlib import Path

import yaml

_ENV_DEFAULTS = {"max_steps": 50, "episodes": 1}


def load_config(path: str | Path) -> tuple[dict, dict]:
    """Load task YAML and return ``(runtime_config, task_config)``.

    Expected schema:
    - benchmark: runtime options such as allowed_tools, constraints, run_budget
    - task: benchmark task definition used by curation/train/eval tools

    Optional legacy ``env`` keys are merged for internal runtime defaults.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    benchmark = raw.get("benchmark", {})
    env_legacy = raw.get("env", {})
    runtime_config = {**_ENV_DEFAULTS, **env_legacy, **benchmark}
    task = raw.get("task", {})
    return runtime_config, task
