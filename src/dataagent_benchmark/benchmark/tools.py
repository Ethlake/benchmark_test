from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from dataagent_benchmark.tools import make_tools
from dataagent_benchmark.domain.tool_context import Injected


@dataclass(frozen=True)
class ToolsBundle:
    schema_tools: dict[str, Any]                 # tool_name -> FunctionTool (给 LLM/schema 看)
    raw_fns: dict[str, Callable]                 # tool_name -> python callable (env 真正执行)
    injected_cache: dict[str, dict[str, Injected]]  # tool_name -> injected params


def get_tools_bundle(tool_names: list[str] | None = None) -> ToolsBundle:
    """
    Benchmark 对外统一入口：获取工具集合（含 schema/raw/injected）。
    """
    schema_tools, raw_fns, injected_cache = make_tools(tool_names)
    return ToolsBundle(schema_tools=schema_tools, raw_fns=raw_fns, injected_cache=injected_cache)
