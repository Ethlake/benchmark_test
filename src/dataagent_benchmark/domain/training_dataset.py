"""Training dataset domain model."""

from __future__ import annotations

from pydantic import BaseModel

from dataagent_benchmark.domain.dataset_schema import ColumnSchema, DatasetSource


class Dataset(BaseModel, extra="forbid"):
    """A single candidate training dataset."""

    name: str
    source: DatasetSource
    columns: ColumnSchema = ColumnSchema()
