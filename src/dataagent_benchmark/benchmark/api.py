from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from dataagent_benchmark.benchmark.env import build_env


@dataclass
class StepOut:
    obs: dict
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class BenchmarkAPI:
    """
    Phase A Benchmark Facade.

    - reset() -> (obs, info)     (会自动 init_run，保证 save_to_disk 等工具可用)
    - step(tool, args) -> StepOut
    - 一些 getter：task/agent_config/tools/max_steps/episodes
    """

    def __init__(
        self,
        task_path: str,
        *,
        tool_names: list[str] | None = None,
        env_config: dict | None = None,
        render_mode: str | None = "human",
    ) -> None:
        self.task_path = task_path
        self.env = build_env(
            task_path=task_path,
            tool_names=tool_names,
            env_config=env_config,
            render_mode=render_mode,
        )
        self._last_info: dict | None = None
        self._run_counter = 0

    # ---- getters (对外稳定) ----
    @property
    def task(self) -> dict:
        return self.env.task

    @property
    def agent_config(self) -> dict | None:
        return self.env.agent_config

    @property
    def tools(self):
        # autogen_core BaseTool list
        return self.env.tools

    @property
    def max_steps(self) -> int:
        return self.env.max_steps

    @property
    def episodes(self) -> int:
        return self.env.episodes

    def _new_run_id(self) -> str:
        self._run_counter += 1
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S_%f")
        return f"{ts}_run{self._run_counter:03d}"

    def reset(self, *, seed: int | None = None, run_id: str | None = None) -> tuple[dict, dict]:
        # 关键：每次 reset 都保证 run_dir 已建立
        rid = run_id or self._new_run_id()
        self.env.init_run(self.env.task, run_id=rid)

        obs, info = self.env.reset(seed=seed)
        self._last_info = info
        return obs, info

    def step(self, tool: str, args: dict[str, Any] | None = None) -> StepOut:
        action = json.dumps({"tool": tool, "args": args or {}})
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        return StepOut(obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

    def available_tools(self) -> list[str]:
        if self._last_info and "available_tools" in self._last_info:
            return list(self._last_info["available_tools"])
        return []

    def close(self) -> None:
        self.env.close()
