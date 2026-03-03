from __future__ import annotations

import json
from pathlib import Path

import typer

from dataagent_benchmark.benchmark.recipe_env import DataCurationEnv
from dataagent_benchmark.domain.recipe import DataRecipe

app = typer.Typer(add_completion=False)


@app.command("describe")
def describe(task: Path) -> None:
    """Print the agent-facing task specification for one run."""
    env = DataCurationEnv(task)
    task_spec = env.reset()
    print(task_spec.model_dump_json(indent=2))


@app.command("run")
def run(task: Path, recipe: Path) -> None:
    """Submit one complete recipe and print the final observation."""
    env = DataCurationEnv(task)
    env.reset()
    recipe_payload = json.loads(recipe.read_text(encoding="utf-8"))
    obs = env.submit_recipe(DataRecipe.model_validate(recipe_payload))
    print(obs.model_dump_json(indent=2))


def main() -> None:
    app()
