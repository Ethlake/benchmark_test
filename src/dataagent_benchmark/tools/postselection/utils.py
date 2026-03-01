"""Shared helpers for postselection tools."""

import math

# ---------------------------------------------------------------------------
# Mix-ratio helpers
# ---------------------------------------------------------------------------


def _temperature_sample(subsets: dict[str, dict], budget: int, alpha: float) -> dict[str, int]:
    weights: dict[str, float] = {}
    for name, meta in subsets.items():
        weights[name] = math.pow(meta["size"], alpha)
    total_w = sum(weights.values()) or 1.0
    return {name: int(budget * w / total_w) for name, w in weights.items()}


def _capped_sample(subsets: dict[str, dict], budget: int, cap: int) -> dict[str, int]:
    capped = {name: min(meta["size"], cap) for name, meta in subsets.items()}
    capped_total = sum(capped.values()) or 1
    return {name: int(budget * cnt / capped_total) for name, cnt in capped.items()}


def _task_balanced_sample(subsets: dict[str, dict], budget: int) -> dict[str, int]:
    tasks: dict[str, list[str]] = {}
    for name, meta in subsets.items():
        tt = meta.get("task_type", "other")
        tasks.setdefault(tt, []).append(name)
    num_tasks = len(tasks) or 1
    per_task = budget // num_tasks
    result: dict[str, int] = {}
    for task_subsets in tasks.values():
        per_subset = per_task // len(task_subsets)
        for name in task_subsets:
            result[name] = per_subset
    return result


_FORMAT_TEMPLATES: dict[str, dict] = {
    "llava": {
        "description": "LLaVA-style conversation with <image> token in user turn.",
        "user_template": "<image>\n{instruction}",
        "user_template_no_image": "{instruction}",
        "assistant_template": "{response}",
        "system_prompt": "",
    },
    "qwen": {
        "description": "Qwen2.5-VL chat template with image placeholders.",
        "user_template": "<|im_start|>user\n<image>{instruction}<|im_end|>",
        "user_template_no_image": "<|im_start|>user\n{instruction}<|im_end|>",
        "assistant_template": "<|im_start|>assistant\n{response}<|im_end|>",
        "system_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
    },
    "chatml": {
        "description": "Generic ChatML instruction format.",
        "user_template": "<|im_start|>user\n{instruction}<|im_end|>",
        "user_template_no_image": "<|im_start|>user\n{instruction}<|im_end|>",
        "assistant_template": "<|im_start|>assistant\n{response}<|im_end|>",
        "system_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
    },
}


# ---------------------------------------------------------------------------
# Save helpers (shared by save_to_disk and save_to_hf)
# ---------------------------------------------------------------------------


def _parse_selection(
    datasets_csv: str,
    sample_counts_csv: str,
) -> tuple[list[str], list[int], str | None]:
    names = [n.strip() for n in datasets_csv.split(",") if n.strip()]
    counts = [int(c.strip()) for c in sample_counts_csv.split(",") if c.strip()]
    if len(names) != len(counts):
        return [], [], "Mismatch: dataset count != sample_count count."
    if not names:
        return [], [], "No datasets provided."
    return names, counts, None


def _build_combined_dataset(
    store_or_dict,
    names: list[str],
    counts: list[int],
):
    from datasets import concatenate_datasets

    parts = []
    included = []
    warnings = []

    for name, count in zip(names, counts, strict=False):
        try:
            if isinstance(store_or_dict, dict):
                ds = store_or_dict[name]
            else:
                ds = store_or_dict.get_working(name)
        except KeyError:
            warnings.append(f"Dataset '{name}' not loaded, skipped.")
            continue
        total = len(ds)
        if count > 0 and count < total:
            ds = ds.select(range(count))
        elif count > total:
            warnings.append(f"'{name}' has only {total} rows, using all.")
        parts.append(ds)
        included.append(name)

    if not parts:
        return None, included, warnings

    return concatenate_datasets(parts), included, warnings
