"""Example of a simple recipe generator using the task specification."""

from __future__ import annotations

from pathlib import Path

from dataagent_benchmark import DataCurationEnv
from dataagent_benchmark.domain.recipe import DataRecipe, ToolCall


def make_recipe_from_task(task_spec) -> DataRecipe:
    dataset_names = [d["name"] for d in task_spec.candidate_datasets]
    ratio = 1.0 / max(len(dataset_names), 1)
    pipeline = [ToolCall(tool="load_dataset", args={"dataset_id": ds}) for ds in dataset_names]
    pipeline.append(ToolCall(tool="vlm_mix", args={"sample_k": 10}))
    return DataRecipe(mixing_ratios={ds: ratio for ds in dataset_names}, pipeline=pipeline)


def main() -> None:
    env = DataCurationEnv(Path("tasks/test_random.yaml"))
    task_spec = env.reset()
    recipe = make_recipe_from_task(task_spec)
    obs = env.submit_recipe(recipe)
    print(obs.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
