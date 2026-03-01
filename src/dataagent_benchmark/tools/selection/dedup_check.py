"""dedup_check tool — measure exact or near-duplicate rates."""

import json
import random
import re
import unicodedata
from typing import Annotated, Any

from dataagent_benchmark.domain.tool_context import PerDataset


def _normalize_text(text: str) -> str:
    """Normalize text for near-dedup: lowercase, strip accents, collapse whitespace."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text


def dedup_check(
    dataset_id: Annotated[str, "Dataset to deduplicate"],
    method: Annotated[
        str, "Dedup method: 'exact' (hash-based) or 'near' (normalized text)"
    ] = "exact",
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    text_column: Annotated[str, PerDataset("text_column")] = "",
) -> str:
    """Measure exact or near-duplicate rates.

    High near-dup rates (>5%) indicate template-generated data.
    """
    if method not in ("exact", "near"):
        return json.dumps({"error": f"Unknown method '{method}'. Use 'exact' or 'near'."})

    ds = dataset
    total = len(ds)
    texts = ds[text_column]

    max_sample = 50_000
    if total > max_sample:
        rng = random.Random(42)
        indices = rng.sample(range(total), max_sample)
        sample = [texts[i] for i in indices]
    else:
        sample = list(texts)

    if method == "exact":
        seen: set[int] = set()
        dup_count = 0
        for v in sample:
            h = hash(v) if isinstance(v, str) else hash(str(v))
            if h in seen:
                dup_count += 1
            else:
                seen.add(h)
    else:
        seen_norm: set[str] = set()
        dup_count = 0
        for v in sample:
            normed = _normalize_text(v) if isinstance(v, str) else ""
            if normed in seen_norm:
                dup_count += 1
            else:
                seen_norm.add(normed)

    sample_size = len(sample)
    dup_rate = dup_count / max(sample_size, 1)
    duplicates = int(total * dup_rate)

    return json.dumps(
        {
            "dataset": dataset_id,
            "original_samples": total,
            "method": method,
            "duplicates": duplicates,
            "unique": total - duplicates,
            "dedup_rate": round(dup_rate * 100, 2),
        },
        indent=2,
    )
