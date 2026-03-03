"""convert_format tool — convert dataset rows into a unified chat format."""

import json
from typing import Annotated, Any

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import DatasetUpdate, PerDataset
from dataagent_benchmark.tools.postselection.utils import _FORMAT_TEMPLATES


def convert_format(
    dataset_id: Annotated[str, "Dataset to convert"],
    target_format: Annotated[str, "Target format: 'llava', 'qwen', or 'chatml'"] = "chatml",
    answer_style: Annotated[
        str, "Answer style: short_answer, step_by_step, latex_proof"
    ] = "short_answer",
    normalize_answers: Annotated[
        bool, "Normalise answers: strip markers, consistent casing"
    ] = True,
    add_unanswerable: Annotated[
        bool, "Add 'cannot determine' examples to reduce hallucination"
    ] = False,
    unanswerable_ratio: Annotated[
        float, "Fraction of unanswerable samples to inject (0-0.1)"
    ] = 0.03,
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    text_column: Annotated[str, PerDataset("text_column")] = "",
    image_column: Annotated[str, PerDataset("image_column")] = "",
) -> StepResult | str:
    """Convert dataset rows into a unified chat instruction format.

    Supports chatml/llava/qwen. Final step before training.
    """
    from datasets import Dataset, concatenate_datasets

    fmt = _FORMAT_TEMPLATES.get(target_format)
    if fmt is None:
        available_fmts = ", ".join(_FORMAT_TEMPLATES.keys())
        return json.dumps(
            {"error": f"Unknown format '{target_format}'. Available: {available_fmts}"}
        )

    has_images = bool(image_column and image_column in dataset.column_names)
    user_tpl = fmt["user_template"] if has_images else fmt["user_template_no_image"]

    ds = dataset
    rows_before = len(ds)

    normalization = []
    if normalize_answers:
        normalization.extend(
            [
                "Consistent casing for label/unit tokens.",
                "Remove trailing whitespace and normalise spacing.",
                "Validate numeric answers (strip $, %, commas).",
            ]
        )
        if answer_style == "latex_proof":
            normalization.append("Validate LaTeX syntax in solutions.")

    def _normalize_text(value: object) -> str:
        text = value if isinstance(value, str) else str(value)
        return " ".join(text.strip().split())

    def _convert_row(row: dict) -> dict:
        text_val = row.get(text_column, "")
        text = text_val if isinstance(text_val, str) else str(text_val)
        if normalize_answers:
            text = _normalize_text(text)

        converted = {
            "user": user_tpl.format(instruction=text),
            "assistant": fmt["assistant_template"].format(response=text),
            "source_dataset": dataset_id,
        }
        if fmt["system_prompt"]:
            converted["system"] = fmt["system_prompt"]
        return converted

    converted_ds = ds.map(_convert_row, remove_columns=ds.column_names)

    unanswerable_count = int(rows_before * unanswerable_ratio) if add_unanswerable else 0
    if unanswerable_count > 0:
        unanswerable_rows: dict[str, list[str]] = {
            "user": [
                user_tpl.format(
                    instruction=("The question cannot be answered from the given information.")
                )
            ]
            * unanswerable_count,
            "assistant": [
                fmt["assistant_template"].format(
                    response="I cannot determine the answer from the given information."
                )
            ]
            * unanswerable_count,
            "source_dataset": [dataset_id] * unanswerable_count,
        }
        if fmt["system_prompt"]:
            unanswerable_rows["system"] = [fmt["system_prompt"]] * unanswerable_count
        converted_ds = concatenate_datasets([converted_ds, Dataset.from_dict(unanswerable_rows)])

    rows_after = len(converted_ds)

    sample_converted = dict(converted_ds[0]) if rows_after > 0 else {}

    result = {
        "dataset": dataset_id,
        "original_samples": rows_before,
        "target_format": target_format,
        "format_description": fmt["description"],
        "field_mapping": {
            "text_column": text_column,
            "has_images": has_images,
        },
        "templates": {
            "system": fmt["system_prompt"] or None,
            "user": user_tpl,
            "assistant": fmt["assistant_template"],
        },
        "normalization": {
            "enabled": normalize_answers,
            "rules": normalization,
            "answer_style": answer_style,
        },
        "unanswerable": {
            "enabled": add_unanswerable,
            "count": unanswerable_count,
            "response_template": ("I cannot determine the answer from the given information."),
        },
        "output_samples": rows_after,
        "output_columns": converted_ds.column_names,
        "sample_conversion": sample_converted,
        "applied": True,
    }

    payload = json.dumps(result, indent=2)
    return StepResult(
        payload=payload,
        created=[ArtifactRef("dataset", dataset_id, 0)],
        updates={f"active_dataset:{dataset_id}": ArtifactRef("dataset", dataset_id, 0)},
        metrics={"rows_before": rows_before, "rows_after": rows_after},
        dataset_updates={
            dataset_id: DatasetUpdate(
                dataset=converted_ds,
                transform_info={
                    "operation": "convert_format",
                    "target_format": target_format,
                    "rows_before": rows_before,
                    "rows_after": rows_after,
                    "normalize_answers": normalize_answers,
                    "add_unanswerable": add_unanswerable,
                    "unanswerable_count": unanswerable_count,
                },
            )
        },
    )
