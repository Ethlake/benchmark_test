from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from dataagent_benchmark.benchmark.api import BenchmarkAPI


def bench_spec(
    ctx: typer.Context,
    task: Annotated[
        Path | None,
        typer.Option("--task", "-t", help="Task YAML (optional). If omitted, use parent TASK positional."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write spec JSON to this path (optional)."),
    ] = None,
    pretty: Annotated[
        bool, typer.Option("--pretty/--no-pretty", help="Pretty JSON formatting.")
    ] = True,
) -> None:
    """
    Export a machine-readable benchmark spec (task + action space + schemas).

    Recommended:
      curate <TASK.yaml> bench-spec --out bench_spec.json
    """
    task_path = task
    if task_path is None and ctx.parent is not None:
        parent_task = ctx.parent.params.get("task")
        if parent_task is not None:
            task_path = parent_task

    if task_path is None:
        raise typer.BadParameter("Missing task. Use: curate <TASK.yaml> bench-spec ...")

    b = BenchmarkAPI(str(task_path), render_mode=None)
    obs, info = b.reset()

    spec = {
        "task_path": str(task_path),
        "episodes": b.episodes,
        "max_steps": b.max_steps,
        "available_tools": info.get("available_tools", []),
        "task": b.task,                 # 原样导出（外部组可读）
        "agent_config": b.agent_config, # 原样导出（外部组可读）
        "observation_schema": {
            "type": "string (e.g., 'task' on reset, 'tool_result' after step)",
            "payload": "any JSON-serializable object",
        },
        "action_schema": {
            "tool": "string (must be one of available_tools)",
            "args": "object/dict",
        },
    }

    b.close()

    txt = json.dumps(spec, indent=2 if pretty else None, ensure_ascii=False)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt, encoding="utf-8")
        print(f"[bench-spec] wrote {out}")
    else:
        print(txt)
