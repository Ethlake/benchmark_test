from __future__ import annotations

from pathlib import Path
from typing import Any

from dataagent_benchmark.infra.gym_env import CurationEnv


def _read_task_tool_allowlist(task_path: str) -> list[str] | None:
    """
    从 task.yaml 里读 agent.tools 作为 allowlist。
    如果没有写，就返回 None（表示不限制）。
    """
    try:
        import yaml  # pyyaml
    except Exception:
        return None

    d: dict[str, Any] = yaml.safe_load(Path(task_path).read_text(encoding="utf-8")) or {}
    agent = d.get("agent") or {}
    tools = agent.get("tools")
    if isinstance(tools, list) and len(tools) > 0:
        return [str(x) for x in tools]
    return None


def build_env(
    task_path: str,
    tool_names: list[str] | None = None,
    *,
    env_config: dict | None = None,
    render_mode: str | None = "human",
) -> CurationEnv:
    """
    Phase A：创建环境（裁判）。
    如果 tool_names 没给，就默认用 task.yaml 里的 agent.tools（公平比较的关键）。
    """
    if tool_names is None:
        tool_names = _read_task_tool_allowlist(task_path)

    return CurationEnv(
        task_path=task_path,
        tool_names=tool_names,
        env_config=env_config,
        render_mode=render_mode,
    )
