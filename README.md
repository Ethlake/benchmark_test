# DataAgent Benchmark (Recipe-Based)

This repository is a **recipe-based data curation benchmark**.

Primary interaction flow:

```text
reset() -> TaskDescription
submit_recipe(DataRecipe) -> Observation
```

The benchmark exposes task metadata and tool specifications to the agent, then executes the full recipe internally (curation tools, training, and evaluation).

## Core API

```python
from dataagent_benchmark import DataCurationEnv
from dataagent_benchmark.domain.recipe import DataRecipe, ToolCall

env = DataCurationEnv("tasks/smoke_train_eval.yaml")
task_spec = env.reset()

recipe = DataRecipe(
    mixing_ratios={"vqav2": 0.7, "captcha": 0.3},
    pipeline=[
        ToolCall(tool="load_dataset", args={"dataset_id": "vqav2"}),
        ToolCall(tool="load_dataset", args={"dataset_id": "captcha"}),
        ToolCall(tool="vlm_mix", args={"sample_k": 25}),
    ],
)

observation = env.submit_recipe(recipe)
```

## CLI

Print task specification:

```bash
dataagent-bench describe tasks/smoke_train_eval.yaml
```

Run a recipe:

```bash
dataagent-bench run tasks/smoke_train_eval.yaml path/to/recipe.json
```

`recipe.json` format:

```json
{
  "mixing_ratios": {"vqav2": 0.7, "captcha": 0.3},
  "pipeline": [
    {"tool": "load_dataset", "args": {"dataset_id": "vqav2"}},
    {"tool": "load_dataset", "args": {"dataset_id": "captcha"}},
    {"tool": "vlm_mix", "args": {"sample_k": 25}}
  ]
}
```

## Notes

- Training backend is reused from `packages/train`.
- Evaluation backend reuses VLMEvalKit integration via `submit_eval`.
- Tool execution is benchmark-owned; agents only submit a full `DataRecipe`.
