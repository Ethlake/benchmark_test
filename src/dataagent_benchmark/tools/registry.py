"""Recipe-facing tool registry and metadata utilities."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from dataagent_benchmark.domain.tool_context import Injected
from dataagent_benchmark.tools.postselection import _register_postselection_tools
from dataagent_benchmark.tools.preselection import _register_preselection_tools
from dataagent_benchmark.tools.selection import _register_selection_tools
from dataagent_benchmark.tools.utilities import _register_utility_tools


@dataclass(slots=True)
class DataToolSpec:
    """Agent-facing specification for one benchmark-owned tool."""

    name: str
    description: str
    parameter_schema: dict[str, Any]
    defaults: dict[str, Any]
    constraints: list[str]
    applicability: str


_TOOL_CONSTRAINTS: dict[str, list[str]] = {
    "load_dataset": ["dataset_id must exist in task.candidate_datasets"],
    "compute_mix_ratio": ["Ratios are normalized to 100%"],
    "vlm_mix": ["sample_k must not exceed sample budget"],
    "submit_finetune": ["Requires at least one curated dataset artifact"],
    "submit_eval": ["Requires active_model artifact from finetune stage"],
}


def _is_injected(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(isinstance(x, Injected) for x in get_args(annotation))


def _annotation_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    args = [getattr(a, "__name__", str(a)) for a in get_args(annotation)]
    return f"{getattr(origin, '__name__', str(origin))}[{', '.join(args)}]"


def _build_parameter_schema(fn: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = inspect.signature(fn)
    params: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if _is_injected(parameter.annotation):
            continue
        required = parameter.default is inspect._empty
        params[name] = {
            "type": _annotation_name(parameter.annotation),
            "required": required,
        }
        if not required:
            params[name]["default"] = parameter.default
            defaults[name] = parameter.default
    return params, defaults


def get_tool_specs(allowed_tools: list[str] | None = None) -> list[DataToolSpec]:
    """Return standardized recipe-facing tool specifications."""
    entries = (
        _register_preselection_tools()
        + _register_selection_tools()
        + _register_postselection_tools()
        + _register_utility_tools()
    )
    specs: list[DataToolSpec] = []
    allow_set = set(allowed_tools) if allowed_tools else None

    for fn, description in entries:
        if allow_set is not None and fn.__name__ not in allow_set:
            continue
        schema, defaults = _build_parameter_schema(fn)
        specs.append(
            DataToolSpec(
                name=fn.__name__,
                description=description,
                parameter_schema=schema,
                defaults=defaults,
                constraints=_TOOL_CONSTRAINTS.get(fn.__name__, []),
                applicability="task-level",
            )
        )
    return specs
