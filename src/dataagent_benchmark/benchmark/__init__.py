"""
Benchmark Facade (Phase A)

目标：给外部 agent 一个稳定入口：
- build_env(): 创建环境
- get_tools_bundle(): 获取工具（含 schema/raw/injected）
- lazy 工具入口：submit_finetune / submit_eval（避免 import-time 依赖炸裂）
"""

from __future__ import annotations

from dataagent_benchmark.infra.gym_env import CurationEnv
from dataagent_benchmark.tools import make_tools

from dataagent_benchmark.benchmark.env import build_env
from dataagent_benchmark.benchmark.tools import ToolsBundle, get_tools_bundle


def submit_finetune(*args, **kwargs):
    # Lazy import: 避免别人只 import benchmark 就要求训练环境齐全
    from dataagent_benchmark.tools.postselection.submit_finetune import submit_finetune as _fn
    return _fn(*args, **kwargs)


def submit_eval(*args, **kwargs):
    from dataagent_benchmark.tools.postselection.submit_eval import submit_eval as _fn
    return _fn(*args, **kwargs)


__all__ = [
    "CurationEnv",
    "make_tools",
    "build_env",
    "ToolsBundle",
    "get_tools_bundle",
    "submit_finetune",
    "submit_eval",
]
