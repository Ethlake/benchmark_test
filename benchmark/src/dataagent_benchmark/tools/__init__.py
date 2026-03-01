"""Tool registry — grouped by pipeline stage.

Provides a make_tools() factory that creates FunctionTools with
Injected parameters stripped from the LLM-visible schema.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any, get_args, get_origin

from autogen_core.tools import FunctionTool

from dataagent_benchmark.domain.tool_context import Injected
from dataagent_benchmark.tools.postselection import _register_postselection_tools
from dataagent_benchmark.tools.preselection import _register_preselection_tools
from dataagent_benchmark.tools.selection import _register_selection_tools
from dataagent_benchmark.tools.utilities import _register_utility_tools


def _get_injected_params(fn: Callable) -> dict[str, Injected]:
    """Introspect a function's annotations to find Injected-marked params.

    Returns a dict mapping param_name -> Injected instance.
    """
    hints = {}
    try:
        hints = fn.__annotations__
    except AttributeError:
        return {}

    result: dict[str, Injected] = {}
    sig = inspect.signature(fn)

    for name, _param in sig.parameters.items():
        annotation = hints.get(name)
        if annotation is None:
            continue
        # Check if it's Annotated[T, ..., Injected("key")]
        if get_origin(annotation) is not None:
            args = get_args(annotation)
            for arg in args:
                if isinstance(arg, Injected):
                    result[name] = arg
                    break
    return result


def _make_schema_fn(fn: Callable, injected_params: set[str]) -> Callable:
    """Create a wrapper with injected params stripped from the signature.

    The wrapper's __signature__ and __annotations__ are modified so
    FunctionTool only generates schema for the LLM-visible params.
    """
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name not in injected_params]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper.__signature__ = sig.replace(parameters=params)
    wrapper.__annotations__ = {
        k: v for k, v in fn.__annotations__.items() if k not in injected_params
    }
    if "return" in fn.__annotations__:
        wrapper.__annotations__["return"] = fn.__annotations__["return"]
    return wrapper


def make_tools(
    names: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Callable], dict[str, dict[str, Injected]]]:
    """Create tool instances with Injected params stripped from schema.

    Returns ``(schema_tools, raw_fns, injected_cache)`` where:
    - *schema_tools* maps tool name -> FunctionTool (for LLM schema)
    - *raw_fns* maps tool name -> the raw callable (full signature)
    - *injected_cache* maps tool name -> {param_name: Injected} for
      each tool, so the env can resolve dependencies at dispatch time
    """
    all_entries = (
        _register_preselection_tools()
        + _register_selection_tools()
        + _register_postselection_tools()
        + _register_utility_tools()
    )

    schema_tools: dict[str, Any] = {}
    raw_fns: dict[str, Callable] = {}
    injected_cache: dict[str, dict[str, Injected]] = {}

    for fn, description in all_entries:
        name = fn.__name__
        injected = _get_injected_params(fn)
        injected_cache[name] = injected
        schema_fn = _make_schema_fn(fn, set(injected.keys()))
        schema_tools[name] = FunctionTool(schema_fn, description=description)
        raw_fns[name] = fn

    if names is not None:
        name_set = set(names)
        schema_tools = {n: t for n, t in schema_tools.items() if n in name_set}
        raw_fns = {n: f for n, f in raw_fns.items() if n in name_set}
        injected_cache = {n: c for n, c in injected_cache.items() if n in name_set}

    return schema_tools, raw_fns, injected_cache
