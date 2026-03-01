from __future__ import annotations

import asyncio
from pathlib import Path
import typer

from dataagent_benchmark.benchmark.api import BenchmarkAPI
from dataagent_benchmark.benchmark.runner import run_external_agent
from dataagent_benchmark.benchmark.spec_cli import bench_spec as _bench_spec

app = typer.Typer(add_completion=False)


@app.command("run")
def run(
    task: Path,
    agent: str = typer.Option(..., "--agent", "-a", help='External agent spec "module:Class"'),
    render: bool = typer.Option(False, "--render/--no-render"),
) -> None:
    b = BenchmarkAPI(str(task), render_mode="human" if render else None)
    summaries = asyncio.run(run_external_agent(b, agent_spec=agent))
    for i, s in enumerate(summaries):
        print(f"[bench episode {i}] stop={s.stop_reason} steps={s.iterations} wall={s.total_wall_s:.2f}s")


@app.command("spec")
def spec(
    task: Path,
    out: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    _bench_spec(typer.Context(app), task=task, out=out)


def main():
    app()
