"""Shared schema building blocks for dataset configurations.

``DatasetSource`` describes *where* to load data from (HuggingFace Hub or
local path).  ``ColumnSchema`` names the columns that carry the text and
image payloads.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator


class DatasetSource(BaseModel, extra="forbid"):
    """Location of a dataset — at least one of *hf_path* or *local_path* required."""

    hf_path: str = ""
    local_path: str = ""
    split: str = "train"
    hf_name: str | None = None

    @model_validator(mode="after")
    def _require_source(self) -> Self:
        if not self.hf_path and not self.local_path:
            raise ValueError("At least one of hf_path or local_path is required")
        return self


class ColumnSchema(BaseModel, extra="forbid"):
    """Column names for the text and image payloads."""

    text: str = "texts"
    image: str = "images"
