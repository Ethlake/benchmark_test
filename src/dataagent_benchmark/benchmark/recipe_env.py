"""Primary recipe-based benchmark environment."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dataagent_benchmark.domain.recipe import DataRecipe, ModelResponse, Observation, TaskDescription
from dataagent_benchmark.infra.gym_env import CurationEnv
from dataagent_benchmark.tools.registry import get_tool_specs


class DataCurationEnv:
    """Recipe-first benchmark API.

    Main interaction: reset() -> submit_recipe(recipe) -> Observation.
    """

    def __init__(self, task_path: str | Path):
        self.task_path = Path(task_path)
        self._raw = yaml.safe_load(self.task_path.read_text(encoding="utf-8")) or {}
        self.task = self._raw.get("task", {})
        self.benchmark = self._raw.get("benchmark", {})
        self._run_idx = 0
        self._last_observation: Observation | None = None
        allowed_tools = self.benchmark.get("allowed_tools")
        self._env = CurationEnv(task_path=str(self.task_path), tool_names=allowed_tools, render_mode=None)
        self._tool_specs = [asdict(s) for s in get_tool_specs(allowed_tools)]

    def _new_run_id(self) -> str:
        self._run_idx += 1
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        return f"{ts}_recipe{self._run_idx:03d}"

    def reset(self) -> TaskDescription:
        run_id = self._new_run_id()
        self._env.init_run(self._env.task, run_id=run_id)
        self._env.reset()
        return TaskDescription(
            run_id=run_id,
            task_description=self.task.get("task_description", ""),
            candidate_datasets=self.task.get("datasets", []),
            budget={
                "sample_budget": int(self.task.get("sample_budget", 0)),
                "run_budget": int(self.benchmark.get("run_budget", self.task.get("sample_budget", 0))),
            },
            constraints=list(self.benchmark.get("constraints", [])),
            available_tools=self._tool_specs,
            training_config=self.task.get("training_config", {}),
            evaluation_config=self.task.get("eval", {}),
        )

    def _validate_recipe(self, recipe: DataRecipe) -> None:
        tool_names = {t["name"] for t in self._tool_specs}
        for call in recipe.pipeline:
            if call.tool not in tool_names:
                raise ValueError(f"Tool '{call.tool}' is not available for this task")

        if recipe.mixing_ratios:
            candidates = {d.get("name") for d in self.task.get("datasets", [])}
            missing = [k for k in recipe.mixing_ratios if k not in candidates]
            if missing:
                raise ValueError(f"Unknown datasets in mixing_ratios: {missing}")

    def submit_recipe(self, recipe: DataRecipe | dict[str, Any]) -> Observation:
        if isinstance(recipe, dict):
            recipe = DataRecipe.model_validate(recipe)
        self._validate_recipe(recipe)

        trace: list[dict] = []
        for call in recipe.pipeline:
            action = json.dumps({"tool": call.tool, "args": call.args})
            obs, reward, terminated, truncated, info = self._env.step(action)
            trace.append(
                {
                    "tool": call.tool,
                    "args": call.args,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "valid": info.get("valid", True),
                    "payload": obs.get("payload", ""),
                }
            )
            if terminated or truncated:
                break

        executed = [x["tool"] for x in trace]
        available_names = {t["name"] for t in self._tool_specs}
        finetune_dataset = "mixed"
        if recipe.mixing_ratios:
            finetune_dataset = next(iter(recipe.mixing_ratios.keys()))
        elif self.task.get("datasets"):
            finetune_dataset = self.task["datasets"][0].get("name", "mixed")

        if "submit_finetune" not in executed and "submit_finetune" in available_names:
            obs, reward, terminated, truncated, info = self._env.step(
                json.dumps({"tool": "submit_finetune", "args": {"dataset_id": finetune_dataset}})
            )
            trace.append({"tool": "submit_finetune", "reward": reward, "valid": info.get("valid", True), "payload": obs.get("payload", "")})

        if "submit_eval" not in executed and "submit_eval" in available_names:
            obs, reward, terminated, truncated, info = self._env.step(
                json.dumps({"tool": "submit_eval", "args": {}})
            )
            trace.append({"tool": "submit_eval", "reward": reward, "valid": info.get("valid", True), "payload": obs.get("payload", "")})

        eval_payload = {}
        for item in reversed(trace):
            if item["tool"] == "submit_eval":
                try:
                    eval_payload = json.loads(item.get("payload") or "{}")
                except json.JSONDecodeError:
                    eval_payload = {}
                break

        observation = Observation(
            run_id=self._env.run_dir.name if self._env.run_dir else "",
            iteration=1,
            benchmark_scores=eval_payload.get("scores", {}),
            training_loss_curve=[],
            remaining_budget=self.task.get("sample_budget"),
            artifacts={"run_dir": str(self._env.run_dir) if self._env.run_dir else None},
            tool_trace=trace,
        )
        self._last_observation = observation
        return observation

    def get_detailed_responses(self, query_ids: list[str]) -> list[ModelResponse]:
        return [ModelResponse(query_id=qid) for qid in query_ids]
