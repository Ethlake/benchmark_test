"""vlm_modify tool — LLM-based text rewriting for VLM datasets."""

import json
import random
from typing import Annotated, Any

import structlog

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import DatasetUpdate, PerDataset
from dataagent_benchmark.infra.vlm_ops import (
    LLMConfig,
    build_vision_messages,
    call_frontier,
    maybe_resize_pil,
    parse_texts_json,
    safe_open_image,
)

log = structlog.get_logger()


def vlm_modify(
    dataset_id: Annotated[str, "Dataset to modify"],
    prompt_template: Annotated[str, "Prompt text sent to the VLM with the image"],
    num_samples: Annotated[int, "Number of rows to rewrite"] = 100,
    model: Annotated[str, "Litellm model name"] = "gpt-4o-mini",
    turns: Annotated[int, "Expected conversation turns per image"] = 5,
    temperature: Annotated[float, "Generation temperature"] = 0.2,
    max_tokens: Annotated[int, "Max tokens per response"] = 1200,
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    image_column: Annotated[str, PerDataset("image_column")] = "",
    text_column: Annotated[str, PerDataset("text_column")] = "",
) -> StepResult:
    """Rewrite or augment text annotations using a frontier vision model.

    Sends each selected row's image(s) + prompt_template to a VLM and
    replaces the text column with the model's response. Use this to
    improve, rephrase, or regenerate conversation data for any dataset.
    """
    log.info("vlm_modify", dataset=dataset_id, num_samples=num_samples, model=model, turns=turns)
    ds = dataset
    image_col = image_column
    text_col = text_column
    total_rows = len(ds)

    num_samples = min(num_samples, total_rows)
    rng = random.Random(42)
    target_indices = set(rng.sample(range(total_rows), num_samples))

    llm_cfg = LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    log.info(
        "vlm_modify_start", dataset=dataset_id, target_rows=num_samples, total_rows=total_rows
    )

    rewritten = 0
    failures = 0
    sample_outputs: list[dict] = []

    # Extract a mutable copy of just the text column — avoids materializing images
    text_values = list(ds[text_col])

    for idx in target_indices:
        images_raw = ds[idx][image_col]
        if not isinstance(images_raw, list) or len(images_raw) == 0:
            failures += 1
            continue

        pil_images = []
        try:
            for im_obj in images_raw:
                im = safe_open_image(im_obj)
                if im is None:
                    raise ValueError("Could not decode image")
                im = maybe_resize_pil(im, max_side=1024)
                pil_images.append(im)
        except Exception:
            failures += 1
            continue

        messages = build_vision_messages(prompt_template, pil_images)
        try:
            raw_response = call_frontier(messages, llm_cfg)
            new_texts = parse_texts_json(raw_response)
            if len(new_texts) > turns:
                new_texts = new_texts[:turns]
            text_values[idx] = new_texts
            rewritten += 1
            if len(sample_outputs) < 3:
                sample_outputs.append(
                    {"idx": idx, "turns": len(new_texts), "preview": new_texts[0]}
                )
        except Exception:
            failures += 1

    log.info("vlm_modify_done", dataset=dataset_id, rewritten=rewritten, failures=failures)

    # Replace only the text column via Arrow set_column — shares all other column buffers
    import pyarrow as pa
    from datasets import Dataset as HFDataset

    table = ds.data  # underlying pyarrow.Table
    col_idx = table.column_names.index(text_col)
    new_col = pa.array(text_values, type=table.schema.field(text_col).type)
    new_table = table.set_column(col_idx, text_col, new_col)
    new_ds = HFDataset(new_table, features=ds.features)

    payload = json.dumps(
        {
            "dataset": dataset_id,
            "total_rows": total_rows,
            "rewritten": rewritten,
            "failures": failures,
            "model": model,
            "sample_outputs": sample_outputs,
        },
        indent=2,
    )
    return StepResult(
        payload=payload,
        created=[ArtifactRef("dataset", dataset_id, 0)],
        updates={f"active_dataset:{dataset_id}": ArtifactRef("dataset", dataset_id, 0)},
        metrics={"rewritten": rewritten, "failures": failures},
        dataset_updates={
            dataset_id: DatasetUpdate(
                dataset=new_ds,
                transform_info={
                    "tool": "vlm_modify",
                    "rewritten": rewritten,
                    "failures": failures,
                    "model": model,
                },
            )
        },
    )
