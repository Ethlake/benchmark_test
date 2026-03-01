"""quality_filter tool — text quality, alignment, and deduplication."""

import hashlib
import json
import re
from collections import Counter
from typing import Annotated, Any

import numpy as np
import structlog

from dataagent_benchmark.domain.artifacts import ArtifactRef, StepResult
from dataagent_benchmark.domain.tool_context import DatasetUpdate, PerDataset

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Text quality heuristics
# ---------------------------------------------------------------------------

_NA_PATTERNS = re.compile(
    r"^(n/?a|none|unknown|not available|no answer|null|unanswerable|sorry|i don'?t know)\.?$",
    re.IGNORECASE,
)

_SINGLE_WORD_SPAM = re.compile(r"^(\w+)(\s+\1){2,}$")  # "yes yes yes"


def _extract_texts(row: dict, text_col: str) -> list[dict]:
    """Return list of {user, assistant} dicts from the text column."""
    val = row.get(text_col)
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for item in val:
            if isinstance(item, dict):
                u = item.get("user", item.get("q", ""))
                a = item.get("assistant", item.get("a", ""))
                out.append({"user": str(u), "assistant": str(a)})
        return out
    if isinstance(val, str):
        return [{"user": "", "assistant": val}]
    return []


def _text_quality_check(
    turns: list[dict],
    min_answer_words: int,
    max_answer_chars: int,
    min_question_words: int,
) -> tuple[bool, list[str]]:
    """Check text quality. Returns (keep, reasons)."""
    reasons: list[str] = []
    if not turns:
        reasons.append("no_text_turns")
        return False, reasons

    for turn in turns:
        answer = turn.get("assistant", "").strip()
        question = turn.get("user", "").strip()

        # N/A or junk answers
        if _NA_PATTERNS.match(answer):
            reasons.append("na_answer")
            break

        # Single-word spam
        if _SINGLE_WORD_SPAM.match(answer):
            reasons.append("spam_answer")
            break

        # Too few words in answer
        if min_answer_words > 0 and len(answer.split()) < min_answer_words:
            reasons.append("short_answer")
            break

        # Answer too long
        if max_answer_chars > 0 and len(answer) > max_answer_chars:
            reasons.append("long_answer")
            break

        # Ambiguous single-word question
        if min_question_words > 0 and len(question.split()) < min_question_words:
            reasons.append("short_question")
            break

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Image-text alignment (heuristic, no CLIP)
# ---------------------------------------------------------------------------


def _alignment_check(
    turns: list[dict],
    min_caption_words: int,
    max_caption_words: int,
) -> tuple[bool, list[str]]:
    """Check caption/answer length sanity as a proxy for alignment."""
    reasons: list[str] = []
    if not turns:
        return True, reasons

    # Check the first turn's answer as the primary caption
    answer = turns[0].get("assistant", "").strip()
    word_count = len(answer.split())

    if min_caption_words > 0 and word_count < min_caption_words:
        reasons.append("caption_too_short")
    if max_caption_words > 0 and word_count > max_caption_words:
        reasons.append("caption_too_long")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _image_hash(img_obj) -> str | None:
    """Compute a simple average hash from a PIL image or HF image object."""
    try:
        from PIL import Image

        if isinstance(img_obj, Image.Image):
            img = img_obj
        elif isinstance(img_obj, dict) and img_obj.get("bytes"):
            from io import BytesIO

            img = Image.open(BytesIO(img_obj["bytes"]))
        elif isinstance(img_obj, dict) and img_obj.get("path"):
            img = Image.open(img_obj["path"])
        else:
            img = img_obj  # HF datasets.Image auto-decodes to PIL

        if not isinstance(img, Image.Image):
            return None

        # Average hash: resize to 8x8 grayscale, threshold by mean
        small = img.convert("L").resize((8, 8))
        pixels = np.array(small, dtype=np.float32).flatten()
        avg = pixels.mean()
        bits = (pixels > avg).astype(np.uint8)
        return "".join(str(b) for b in bits)
    except Exception:
        return None


def _text_fingerprint(turns: list[dict]) -> str:
    """Compute a hash fingerprint of the concatenated text."""
    text = " ".join(f"{t.get('user', '')} {t.get('assistant', '')}" for t in turns).strip().lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


def quality_filter(
    dataset_id: Annotated[str, "Dataset to filter"],
    min_answer_words: Annotated[int, "Min words per answer (0=disable)"] = 2,
    max_answer_chars: Annotated[int, "Max chars per answer (0=disable)"] = 0,
    min_question_words: Annotated[int, "Min words per question (0=disable)"] = 0,
    min_caption_words: Annotated[int, "Min words in first answer for alignment (0=disable)"] = 3,
    max_caption_words: Annotated[int, "Max words in first answer (0=disable)"] = 0,
    dedup_images: Annotated[bool, "Remove near-duplicate images"] = True,
    dedup_texts: Annotated[bool, "Remove duplicate text content"] = True,
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    text_column: Annotated[str, PerDataset("text_column")] = "",
    image_column: Annotated[str | None, PerDataset("image_column")] = None,
) -> StepResult:
    """Filter a dataset by text quality, alignment, and deduplication.

    Removes low-quality text (N/A answers, spam, too short), checks
    caption length sanity, and deduplicates by image hash and text
    fingerprint. Operates on the working copy.
    """
    log.info("quality_filter", dataset=dataset_id)
    ds = dataset
    text_col = text_column
    image_col = image_column
    original_count = len(ds)
    log.info(
        "quality_filter_start",
        dataset=dataset_id,
        rows=original_count,
        text_col=text_col,
        image_col=image_col,
    )

    kept_indices: list[int] = []
    reason_counter: Counter[str] = Counter()
    seen_image_hashes: set[str] = set()
    seen_text_fps: set[str] = set()
    dedup_image_count = 0
    dedup_text_count = 0

    for idx in range(original_count):
        row = ds[idx]
        turns = _extract_texts(row, text_col)

        # --- Text quality ---
        keep, reasons = _text_quality_check(
            turns,
            min_answer_words=min_answer_words,
            max_answer_chars=max_answer_chars,
            min_question_words=min_question_words,
        )
        if not keep:
            for r in reasons:
                reason_counter[r] += 1
            continue

        # --- Alignment check ---
        keep, reasons = _alignment_check(
            turns,
            min_caption_words=min_caption_words,
            max_caption_words=max_caption_words,
        )
        if not keep:
            for r in reasons:
                reason_counter[r] += 1
            continue

        # --- Text dedup ---
        if dedup_texts and turns:
            fp = _text_fingerprint(turns)
            if fp in seen_text_fps:
                dedup_text_count += 1
                reason_counter["duplicate_text"] += 1
                continue
            seen_text_fps.add(fp)

        # --- Image dedup ---
        if dedup_images and image_col and image_col in ds.column_names:
            images = row.get(image_col, [])
            if isinstance(images, list) and len(images) > 0:
                h = _image_hash(images[0])
                if h is not None:
                    if h in seen_image_hashes:
                        dedup_image_count += 1
                        reason_counter["duplicate_image"] += 1
                        continue
                    seen_image_hashes.add(h)

        kept_indices.append(idx)

    filtered = ds.select(kept_indices)
    log.info("quality_filter_done", dataset=dataset_id, kept=len(filtered), total=original_count)

    payload = json.dumps(
        {
            "dataset": dataset_id,
            "original_rows": original_count,
            "kept": len(filtered),
            "rejected": original_count - len(filtered),
            "rejection_reasons": dict(reason_counter.most_common()),
            "dedup_stats": {
                "duplicate_images": dedup_image_count,
                "duplicate_texts": dedup_text_count,
            },
        },
        indent=2,
    )
    return StepResult(
        payload=payload,
        created=[ArtifactRef("dataset", dataset_id, 0)],
        updates={f"active_dataset:{dataset_id}": ArtifactRef("dataset", dataset_id, 0)},
        metrics={"rows_kept": len(filtered), "rows_rejected": original_count - len(filtered)},
        dataset_updates={
            dataset_id: DatasetUpdate(
                dataset=filtered,
                transform_info={
                    "tool": "quality_filter",
                    "original": original_count,
                    "kept": len(filtered),
                    "rejected": original_count - len(filtered),
                },
            )
        },
    )
