"""Agent configuration — pure data, no dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AgentConfig:
    """Knobs for the curation agent loop."""

    stream: bool = True
    max_repeat_calls: int = 3
    max_consecutive_text: int = 3
    llm_retries: int = 3
    result_preview_chars: int = 200
    log_dir: str = ""
    log_llm_requests: bool = False
    tools: list[str] | None = None
