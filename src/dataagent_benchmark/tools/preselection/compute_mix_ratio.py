"""compute_mix_ratio tool — build the final data mixture recipe."""

import json
import math
from typing import Annotated

import structlog

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.models import MixRecipe, RecipeComponent
from dataagent_benchmark.domain.tool_context import FromEnv

log = structlog.get_logger()


def compute_mix_ratio(
    ratios: Annotated[
        str,
        "Comma-separated ratio percentages that sum to 100 "
        "(e.g. '60,40'). Leave empty for auto-compute.",
    ] = "",
    quality_scores: Annotated[
        str, "Comma-separated quality scores (0-1) per dataset for auto-compute mode"
    ] = "",
    effective_sizes: Annotated[
        str, "Comma-separated working copy row counts per dataset for auto-compute mode"
    ] = "",
    categories: Annotated[
        str,
        "Comma-separated category labels aligned with datasets (defaults to 'general')",
    ] = "",
    total_samples: Annotated[int, "Total number of samples in the final mix"] = 0,
    # --- Injected by env ---
    dataset_names: Annotated[list[str], FromEnv("dataset_names")] = None,
    sample_budget: Annotated[int, FromEnv("sample_budget")] = 100_000,
) -> str:
    """Compute a data mixture recipe with per-dataset sample allocations.

    Accepts either manual ratio percentages or quality_scores +
    effective_sizes for automatic weighting (quality * sqrt(size)).
    """
    log.info("compute_mix_ratio", ratios=ratios, total_samples=total_samples)
    names = dataset_names
    if total_samples <= 0:
        total_samples = sample_budget

    has_ratios = bool(ratios.strip())
    has_quality = bool(quality_scores.strip())
    has_sizes = bool(effective_sizes.strip())

    # Parse categories
    if categories.strip():
        cats = [c.strip() for c in categories.split(",")]
        if len(cats) != len(names):
            return json.dumps({"error": "Mismatch: dataset count != category count."})
    else:
        cats = ["general"] * len(names)

    if has_ratios:
        # Manual mode
        ratio_vals = [float(r.strip()) for r in ratios.split(",")]
        if len(names) != len(ratio_vals):
            return json.dumps({"error": "Mismatch: dataset count != ratio count."})

        total_ratio = sum(ratio_vals)
        components: list[RecipeComponent] = []
        allocation: dict[str, int] = {}
        warnings: list[str] = []

        for name, ratio, cat in zip(names, ratio_vals, cats, strict=False):
            count = int(total_samples * ratio / total_ratio)
            allocation[name] = count
            pct = round(ratio / total_ratio * 100, 2)
            components.append(
                RecipeComponent(
                    dataset=name,
                    category=cat,
                    ratio_pct=pct,
                    sample_count=count,
                    capped=False,
                )
            )

        recipe = MixRecipe(
            components=components,
            total_samples=sum(allocation.values()),
            total_requested=total_samples,
            warnings=warnings,
            mode="manual",
        )
        log.info("recipe_computed", mode="manual", total_samples=recipe.total_samples)
        result: dict = {
            "recipe": json.loads(recipe.model_dump_json()),
            "allocation": allocation,
            "total": recipe.total_samples,
        }
        if warnings:
            result["warnings"] = warnings
        return StepResult(
            payload=json.dumps(result, indent=2),
            created=[ArtifactRef("recipe", "mix", 0)],
            updates={"active_recipe": ArtifactRef("recipe", "mix", 0)},
            metrics={"total_samples": recipe.total_samples, "mode": "manual"},
            recipe=recipe,
        )

    elif has_quality and has_sizes:
        # Auto-compute mode
        q_scores = [float(q.strip()) for q in quality_scores.split(",")]
        e_sizes = [int(float(s.strip())) for s in effective_sizes.split(",")]

        if len(names) != len(q_scores):
            return json.dumps({"error": "Mismatch: dataset count != quality_scores count."})
        if len(names) != len(e_sizes):
            return json.dumps({"error": "Mismatch: dataset count != effective_sizes count."})

        weights = [q * math.sqrt(s) for q, s in zip(q_scores, e_sizes, strict=False)]
        total_weight = sum(weights)
        ratio_vals = [w / total_weight * 100 for w in weights]

        components = []
        allocation = {}
        warnings = []
        computation = []

        for name, ratio, cat, q, s, w in zip(
            names, ratio_vals, cats, q_scores, e_sizes, weights, strict=False
        ):
            count = int(total_samples * ratio / 100)
            capped = False
            if count > s:
                warnings.append(
                    f"'{name}' effective_size is {s} samples but {count} requested; capping."
                )
                count = s
                capped = True
            allocation[name] = count
            pct = round(ratio, 2)
            components.append(
                RecipeComponent(
                    dataset=name,
                    category=cat,
                    ratio_pct=pct,
                    sample_count=count,
                    capped=capped,
                    quality_score=q,
                    effective_size=s,
                    weight=round(w, 4),
                )
            )
            computation.append(
                {
                    "dataset": name,
                    "quality_score": q,
                    "effective_size": s,
                    "weight": round(w, 4),
                    "ratio_pct": pct,
                    "sample_count": count,
                    "capped": capped,
                }
            )

        formula = "weight = quality_score * sqrt(effective_size)"
        recipe = MixRecipe(
            components=components,
            total_samples=sum(allocation.values()),
            total_requested=total_samples,
            warnings=warnings,
            mode="auto_quality_weighted",
            formula=formula,
        )
        log.info(
            "recipe_computed", mode="auto_quality_weighted", total_samples=recipe.total_samples
        )
        result = {
            "recipe": json.loads(recipe.model_dump_json()),
            "allocation": allocation,
            "total": recipe.total_samples,
            "computation": {
                "formula": formula,
                "per_dataset": computation,
                "total_weight": round(total_weight, 4),
            },
        }
        if warnings:
            result["warnings"] = warnings
        return StepResult(
            payload=json.dumps(result, indent=2),
            created=[ArtifactRef("recipe", "mix", 0)],
            updates={"active_recipe": ArtifactRef("recipe", "mix", 0)},
            metrics={"total_samples": recipe.total_samples, "mode": "auto_quality_weighted"},
            recipe=recipe,
        )

    elif has_quality or has_sizes:
        return json.dumps(
            {
                "error": (
                    "Auto-compute requires both quality_scores "
                    "and effective_sizes. Provide both, or "
                    "provide ratios for manual mode."
                )
            }
        )
    else:
        return json.dumps(
            {
                "error": (
                    "No ratios or quality_scores/effective_sizes "
                    "provided. Provide ratios for manual mode, "
                    "or quality_scores and effective_sizes "
                    "for auto-compute."
                )
            }
        )
