"""Config loading — parse YAML into (agent_config, env_config, task) triple.

Supports two YAML layouts:

**Structured (preferred)**::

    agent:
      prompt_style: full
      tools: [...]
    env:
      episodes: 5
      max_steps: 50
      output_path: output.parquet
    task:
      task_description: ...
      target_model: ...

"""

from __future__ import annotations

from pathlib import Path

import yaml

_ENV_DEFAULTS = {"episodes": 1, "max_steps": 50}


def load_config(path: str | Path) -> tuple[dict, dict, dict]:
    """Load a YAML config file and return ``(agent_config, env_config, task)``.

    Returns
    -------
    agent_config : dict
        Agent-level settings (prompt_style, generation, tools, …).
    env_config : dict
        Environment settings (episodes, max_steps, output_path).
        Defaults to ``{"episodes": 1, "max_steps": 50}`` when absent.
    task : dict
        Task definition consumed by tools and prompt builder.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    agent_config = raw.get("agent", {})
    env_config = {**_ENV_DEFAULTS, **raw.get("env", {})}
    task = raw["task"]

    return agent_config, env_config, task
