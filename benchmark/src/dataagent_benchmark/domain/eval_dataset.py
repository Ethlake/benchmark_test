"""Evaluation dataset domain model."""

from __future__ import annotations

from pydantic import BaseModel

from dataagent_benchmark.domain.dataset_schema import ColumnSchema, DatasetSource


class EvalDataset(BaseModel, extra="forbid"):
    """A single evaluation benchmark split."""

    name: str
    source: DatasetSource
    columns: ColumnSchema = ColumnSchema()
    num_samples: int | None = None
