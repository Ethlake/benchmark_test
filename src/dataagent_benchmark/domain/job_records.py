"""Job record domain models for async finetune and eval tracking."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FinetuneJobRecord(BaseModel):
    """Tracks the state of an asynchronous finetune job."""

    job_id: str
    status: str = "submitted"
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    adapter_path: str | None = None
    error: str | None = None


class EvalJobRecord(BaseModel):
    """Tracks the state of an asynchronous evaluation job."""

    job_id: str
    status: str = "submitted"
    results: dict[str, float | int | str] = Field(default_factory=dict)
    error: str | None = None
