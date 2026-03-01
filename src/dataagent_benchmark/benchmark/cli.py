from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from dataagent_benchmark.benchmark.api import BenchmarkAPI
from dataagent_benchmark.benchmark.runner import run_external_agent, run_policy_agent


def _write_summary(run_dir: str | None, summaries) -> None:
    if not run_dir:
        return
    out = Path(run_dir) / "bench_summary.json"
    payload = []
    for s in summaries:
        payload.append(
            {
                "episode": s.episode,
                "stop_reason": s.stop_reason,
                "iterations": s.iterations,
                "total_wall_s": s.total_wall_s,
                "total_reward": s.total_reward,
                "tool_calls": len(s.steps),
                "errors": sum(1 for x in s.steps if getattr(x, "is_error", False)),
                "tools": [
                    {"tool": x.tool_name, "is_error": x.is_error}
                    for x in s.steps
                ],
            }
        )
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out}")


def bench(
    ctx: typer.Context,
    task: Annotated[
        Path | None,
        typer.Option("--task", "-t", help="Task YAML path (optional). If omitted, use parent TASK positional."),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", "-a", help='Agent: "task" | "random" | "llm" | "module:Class"'),
    ] = "task",
    render: Annotated[
        bool, typer.Option("--render/--no-render", help="Render env steps.")
    ] = False,
    log_llm_requests: Annotated[
        bool, typer.Option("--log-llm-requests", help="Dump LLM HTTP requests (policy agent).")
    ] = False,
) -> None:
    """
    Benchmark runner (Phase A).

    推荐用法：
      curate <TASK.yaml> bench --agent task
    """
    task_path = task
    if task_path is None and ctx.parent is not None:
        parent_task = ctx.parent.params.get("task")
        if parent_task is not None:
            task_path = parent_task

    if task_path is None:
        raise typer.BadParameter("Missing task. Use: curate <TASK.yaml> bench ...")

    bench_api = BenchmarkAPI(str(task_path), render_mode="human" if render else None)

    if agent in ("task", "random", "llm"):
        policy_override = None if agent == "task" else agent
        summaries = asyncio.run(
            run_policy_agent(
                bench_api,
                policy_override=policy_override,
                log_llm_requests=log_llm_requests,
            )
        )
    else:
        summaries = asyncio.run(run_external_agent(bench_api, agent_spec=agent))

    for i, s in enumerate(summaries):
        print(f"[bench episode {i}] stop={s.stop_reason}  steps={s.iterations}  wall={s.total_wall_s:.1f}s")

    # 把 summary 写到 env 的 run_dir（BenchmarkAPI.reset 自动 init_run 了）
    _write_summary(getattr(bench_api.env, "run_dir", None), summaries)
