"""Telemetry structs for agent runs — pure data, no dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    """Telemetry for a single tool execution."""

    iteration: int
    tool_name: str
    tool_args: dict[str, Any]
    result_preview: str
    is_error: bool
    prompt_tokens: int
    completion_tokens: int
    llm_wall_s: float
    tool_wall_s: float
    thought: str | None = None
    result_full: str | None = None


@dataclass
class RunSummary:
    """Aggregate telemetry for a complete agent run."""

    stop_reason: str = "max_iterations"
    iterations: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_wall_s: float = 0.0
    final_response: str | None = None
    system_prompt: str | None = None
    episode: int = 0
    total_reward: float = 0.0
