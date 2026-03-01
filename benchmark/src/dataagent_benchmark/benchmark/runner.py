from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from dataagent_benchmark.benchmark.api import BenchmarkAPI, StepOut


@dataclass
class Action:
    tool: str
    args: dict[str, Any]


def _normalize_action(x: Any) -> Action:
    """
    External agent can return:
      - {"tool": "...", "args": {...}}
      - ("tool_name", {...})
      - Action(tool, args)
    """
    if isinstance(x, Action):
        return x
    if isinstance(x, tuple) and len(x) == 2:
        tool, args = x
        return Action(tool=str(tool), args=dict(args or {}))
    if isinstance(x, dict) and "tool" in x:
        return Action(tool=str(x["tool"]), args=dict(x.get("args") or {}))
    raise TypeError(f"Invalid action returned by agent: {x!r}")


@dataclass
class RunSummary:
    episode: int
    stop_reason: str
    iterations: int
    total_wall_s: float
    total_reward: float
    tool_calls: int
    errors: int


async def run_external_agent(
    bench: BenchmarkAPI,
    *,
    agent_spec: str,
) -> list[RunSummary]:
    """
    Standalone benchmark runner for external agents.
    External agent must implement act(obs)->tool-call.
    """
    from dataagent_benchmark.benchmark.loader import load_agent

    agent = load_agent(agent_spec)
    summaries: list[RunSummary] = []

    for ep in range(1, bench.episodes + 1):
        obs, info = bench.reset()
        if hasattr(agent, "reset"):
            agent.reset(obs, info)

        run_start = time.perf_counter()
        total_reward = 0.0
        stop_reason = "max_iterations"
        errors = 0
        tool_calls = 0

        cur_obs = obs
        for step_idx in range(bench.max_steps):
            action_raw = agent.act(cur_obs)
            act = _normalize_action(action_raw)

            out: StepOut = bench.step(act.tool, act.args)
            tool_calls += 1
            total_reward += out.reward

            is_error = not out.info.get("valid", True)
            if is_error:
                errors += 1

            if hasattr(agent, "observe"):
                agent.observe(out)

            cur_obs = out.obs

            if out.terminated:
                stop_reason = "env_error" if is_error else "env_terminated"
                break
            if out.truncated:
                stop_reason = "env_truncated"
                break

        wall = time.perf_counter() - run_start
        summaries.append(
            RunSummary(
                episode=ep,
                stop_reason=stop_reason,
                iterations=min(bench.max_steps, step_idx + 1),
                total_wall_s=wall,
                total_reward=total_reward,
                tool_calls=tool_calls,
                errors=errors,
            )
        )

    bench.close()
    return summaries
