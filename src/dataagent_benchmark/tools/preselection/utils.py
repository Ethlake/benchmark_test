"""Shared helpers for preselection tools."""

_REASONING_MARKERS = ["Step", "step", "1.", "1)", "First,", "therefore"]


def _detect_behaviors(task_description: str) -> list[str]:
    """Return heuristic behavior tags from task description."""
    text = task_description.lower()
    tags = []
    if any(k in text for k in ["ocr", "document", "receipt", "chart"]):
        tags.append("ocr")
    if any(k in text for k in ["reason", "proof", "step", "math"]):
        tags.append("reasoning")
    if not tags:
        tags.append("general_qa")
    return tags


def _generate_quality_note(row: dict, meta) -> str:
    """Generate an automated quality note for a sample row."""
    issues: list[str] = []
    q = row.get(meta.question_field, "")
    a = row.get(meta.answer_field, "")

    if not isinstance(q, str) or q.strip() == "":
        issues.append("empty question field")
    if not isinstance(a, str) or a.strip() == "":
        issues.append("empty answer field")
    elif len(a) < 10:
        issues.append(f"very short answer ({len(a)} chars)")

    if meta.answer_marker and isinstance(a, str) and meta.answer_marker not in a:
        issues.append(f"missing expected '{meta.answer_marker}' marker")

    step_styles = ("step_by_step", "step_by_step_with_marker", "latex_proof")
    if meta.answer_style in step_styles and isinstance(a, str) and a.count("\n") < 1:
        has_step_marker = any(m in a for m in _REASONING_MARKERS)
        if not has_step_marker:
            issues.append("no reasoning steps detected")

    if issues:
        return "Issue: " + "; ".join(issues)
    return "Good: passes automated quality checks."
