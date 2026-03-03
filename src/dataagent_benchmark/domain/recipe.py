"""Recipe-based benchmark models."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    """One tool invocation in a complete curation recipe."""

    tool: str = Field(..., description="Registered benchmark tool name.")
    args: dict = Field(default_factory=dict, description="Tool arguments.")


class DataRecipe(BaseModel):
    """Complete curation plan submitted once per run."""

    pipeline: list[ToolCall] = Field(default_factory=list)
    mixing_ratios: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ratios(self) -> "DataRecipe":
        if self.mixing_ratios:
            total = sum(self.mixing_ratios.values())
            if abs(total - 1.0) > 1e-6 and abs(total - 100.0) > 1e-6:
                raise ValueError("mixing_ratios must sum to 1.0 or 100.0")
        return self


class TaskDescription(BaseModel):
    run_id: str
    task_description: str
    candidate_datasets: list[dict]
    budget: dict
    constraints: list[str] = Field(default_factory=list)
    available_tools: list[dict]
    training_config: dict = Field(default_factory=dict)
    evaluation_config: dict = Field(default_factory=dict)


class Observation(BaseModel):
    run_id: str
    iteration: int
    benchmark_scores: dict[str, float] = Field(default_factory=dict)
    training_loss_curve: list[float] = Field(default_factory=list)
    remaining_budget: int | None = None
    artifacts: dict = Field(default_factory=dict)
    tool_trace: list[dict] = Field(default_factory=list)


class ModelResponse(BaseModel):
    query_id: str
    prompt: str | None = None
    response: str | None = None
    score: float | None = None
