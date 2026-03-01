"""Domain models for the data curation pipeline.

Pure Pydantic models with no internal dependencies.
"""

from pydantic import BaseModel, Field


class RecipeComponent(BaseModel):
    """One dataset slot in a data mixture recipe."""

    dataset: str = Field(..., description="Dataset name.")
    category: str = Field(
        default="general",
        description=(
            "Role in the mixture: general, domain, ocr_doc_chart, unanswerable, or a custom label."
        ),
    )
    ratio_pct: float = Field(..., ge=0.0, le=100.0, description="Requested share of the mix (%).")
    sample_count: int = Field(..., ge=0, description="Computed number of samples.")
    capped: bool = Field(
        default=False,
        description="True when sample_count was capped at the dataset's size.",
    )
    quality_score: float | None = Field(
        default=None,
        description="Quality score used in auto-compute mode (0-1).",
    )
    effective_size: int | None = Field(
        default=None,
        description="Working copy row count used in auto-compute mode.",
    )
    weight: float | None = Field(
        default=None,
        description="Raw weight from auto-compute formula.",
    )


class DataRecipe(BaseModel):
    """A structured data mixture recipe."""

    components: list[RecipeComponent] = Field(..., description="Ordered list of dataset slots.")
    total_samples: int = Field(..., description="Sum of allocated sample counts.")
    total_requested: int = Field(..., description="Original total budget before capping.")
    warnings: list[str] = Field(default_factory=list)
    mode: str = Field(
        default="manual",
        description="How ratios were determined: 'manual' or 'auto_quality_weighted'.",
    )
    formula: str | None = Field(
        default=None,
        description="Formula used in auto-compute mode.",
    )
