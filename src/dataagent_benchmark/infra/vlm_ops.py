"""VLM dataset operations — core logic extracted from vlmtools/.

Provides reusable functions for filtering, mixing, and modifying
VLM datasets. Tool wrappers in tools/preselection/ call these.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Image utilities (from filter.py)
# ---------------------------------------------------------------------------


def _get_lanczos():
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        return Image.LANCZOS


def safe_open_image(obj: Any) -> Image.Image | None:
    """Decode an image from common HF datasets.Image shapes.

    Handles PIL.Image, dict with bytes/path, and string paths.
    Returns RGB PIL Image or None on failure.
    """
    try:
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, dict):
            if obj.get("bytes") is not None:
                return Image.open(BytesIO(obj["bytes"])).convert("RGB")
            if obj.get("path") is not None:
                return Image.open(obj["path"]).convert("RGB")
        if isinstance(obj, (str, os.PathLike)):
            return Image.open(str(obj)).convert("RGB")
    except Exception:
        return None
    return None


def image_basic_stats(img: Image.Image) -> dict[str, Any]:
    w, h = img.size
    ar = max(w / max(h, 1), h / max(w, 1))
    return {
        "w": int(w),
        "h": int(h),
        "min_side": int(min(w, h)),
        "max_side": int(max(w, h)),
        "aspect": float(ar),
    }


# ---------------------------------------------------------------------------
# Text heuristics (from filter.py)
# ---------------------------------------------------------------------------


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


# ---------------------------------------------------------------------------
# Filter config & decision (from filter_refined.py)
# ---------------------------------------------------------------------------

# Hard-coded filter thresholds (DO NOT pass via tool args)
MIN_SIDE = 200
MAX_ASPECT = 3.0


@dataclass
class FilterConfig:
    image_field: str


def decide_keep(ex: dict[str, Any], cfg: FilterConfig) -> tuple[bool, list[str], dict[str, Any]]:
    """Decide whether to keep a row based on image integrity.

    Checks: image presence, decodability, minimum side length, aspect ratio.
    Text quality is handled separately by quality_filter.
    """
    reasons: list[str] = []
    meta: dict[str, Any] = {}

    # Only check the FIRST image.
    imgs = ex.get(cfg.image_field)
    if not isinstance(imgs, list) or len(imgs) == 0:
        reasons.append("missing_image")
        return False, reasons, meta

    im = safe_open_image(imgs[0])
    if im is None:
        reasons.append("corrupt_image")
        return False, reasons, meta

    stats = image_basic_stats(im)
    meta["image_stats"] = [stats]

    if MIN_SIDE > 0 and stats["min_side"] < MIN_SIDE:
        reasons.append(f"too_small_image(min_side<{MIN_SIDE})")

    if MAX_ASPECT > 0 and stats["aspect"] > MAX_ASPECT:
        reasons.append(f"extreme_aspect(>{MAX_ASPECT})")

    keep = len(reasons) == 0
    return keep, reasons, meta


# ---------------------------------------------------------------------------
# Mix helpers (from mix.py)
# ---------------------------------------------------------------------------


def take_examples(
    ds: Any,  # datasets.Dataset
    take: int | None,
    sample_mode: str,
    seed: int,
) -> tuple[Any, list[int]]:
    """Take K examples from a dataset.

    Args:
        take: None means all, 0 means none.
        sample_mode: 'first' or 'random'.
    """
    n = len(ds)
    if take is None or take >= n:
        return ds, list(range(n))
    if take <= 0:
        return ds.select([]), []
    if sample_mode == "first":
        idx = list(range(take))
        return ds.select(idx), idx
    if sample_mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=take, replace=False).tolist()
        return ds.select(idx), idx
    raise ValueError(f"Unknown sample_mode: {sample_mode}")


def ensure_same_schema(datasets_list: list) -> None:
    """Enforce identical features across datasets for safe concatenation."""
    if not datasets_list:
        return
    base = datasets_list[0].features
    for i, ds in enumerate(datasets_list[1:], start=1):
        if ds.features != base:
            raise ValueError(
                "Datasets schema/features mismatch; cannot concatenate safely.\n"
                f"- Dataset[0] columns: {datasets_list[0].column_names}\n"
                f"- Dataset[{i}] columns: {ds.column_names}\n"
                "Tip: ensure all input datasets use the same format/fields, "
                "or pre-process them to a common schema before mixing."
            )


# ---------------------------------------------------------------------------
# Modify helpers (from modify.py)
# ---------------------------------------------------------------------------


def maybe_resize_pil(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so max(w,h) <= max_side, keeping aspect ratio."""
    if max_side is None or max_side <= 0:
        return img
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    im = img.copy()
    im.thumbnail((max_side, max_side), _get_lanczos())
    return im


def pil_to_data_uri(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    """Convert PIL image to data URI for OpenAI-compatible vision APIs."""
    buf = BytesIO()
    save_kwargs: dict[str, Any] = {}
    if fmt.upper() in {"JPG", "JPEG"}:
        save_kwargs.update({"quality": quality, "optimize": True})
        mime = "image/jpeg"
        fmt_use = "JPEG"
    else:
        mime = "image/png"
        fmt_use = "PNG"
    img.save(buf, format=fmt_use, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_vision_messages(prompt_text: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    """Build a single user message with text prompt and image(s)."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for im in images:
        uri = pil_to_data_uri(im, fmt="JPEG")
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return [{"role": "user", "content": content}]


def parse_texts_json(raw: str) -> list[dict[str, str]]:
    """Parse VLM output into [{user, assistant}, ...] format.

    Accepts: bare list, {texts: [...]}, or {qas: [{q, a}, ...]}.
    """
    raw = raw.strip()
    start_candidates = [raw.find("["), raw.find("{")]
    start_candidates = [x for x in start_candidates if x != -1]
    if start_candidates:
        start = min(start_candidates)
        end = max(raw.rfind("]"), raw.rfind("}"))
        raw_json = raw[start : end + 1] if end > start else raw
    else:
        raw_json = raw

    obj = json.loads(raw_json)

    if isinstance(obj, list):
        texts = obj
    elif isinstance(obj, dict) and "texts" in obj and isinstance(obj["texts"], list):
        texts = obj["texts"]
    elif isinstance(obj, dict) and "qas" in obj and isinstance(obj["qas"], list):
        texts = [{"user": qa.get("q", ""), "assistant": qa.get("a", "")} for qa in obj["qas"]]
    else:
        raise ValueError("JSON parsed but did not match expected schema (list or {texts:[...]})")

    out: list[dict[str, str]] = []
    for t in texts:
        if not isinstance(t, dict):
            raise ValueError("texts element is not an object/dict")
        if "user" in t and "assistant" in t:
            out.append({"user": str(t["user"]), "assistant": str(t["assistant"])})
        elif "q" in t and "a" in t:
            out.append({"user": str(t["q"]), "assistant": str(t["a"])})
        else:
            raise ValueError("texts element missing user/assistant (or q/a) keys")
    return out


@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: int = 120
    max_retries: int = 5
    retry_backoff: float = 1.5
    api_base: str | None = None
    api_key: str | None = None


def call_frontier(messages: list[dict[str, Any]], cfg: LLMConfig) -> str:
    """Call a frontier VLM via litellm with retries."""
    import litellm

    litellm.drop_params = True
    litellm.suppress_debug_info = True

    last_err: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = litellm.completion(
                model=cfg.model,
                messages=messages,
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout,
            )
            # Extract content text
            try:
                return resp["choices"][0]["message"]["content"]
            except Exception:
                return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff * attempt)
                continue
            break
    raise RuntimeError(f"LLM call failed after {cfg.max_retries} retries") from last_err
