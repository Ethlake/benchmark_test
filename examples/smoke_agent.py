"""Minimal recipe-based benchmark example."""

from __future__ import annotations

from pathlib import Path

from dataagent_benchmark import DataCurationEnv
from dataagent_benchmark.domain.recipe import DataRecipe, ToolCall


def build_recipe() -> DataRecipe:
    return DataRecipe(
        mixing_ratios={"vqav2": 0.7, "captcha": 0.3},
        pipeline=[
            ToolCall(tool="load_dataset", args={"dataset_id": "vqav2"}),
            ToolCall(tool="load_dataset", args={"dataset_id": "captcha"}),
            ToolCall(tool="vlm_mix", args={"sample_k": 25}),
        ],
    )


def main() -> None:
    env = DataCurationEnv(Path("tasks/smoke_train_eval.yaml"))
    task_spec = env.reset()
    print("Task tools:", [tool["name"] for tool in task_spec.available_tools])

    recipe = build_recipe()
    observation = env.submit_recipe(recipe)
    print(observation.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
