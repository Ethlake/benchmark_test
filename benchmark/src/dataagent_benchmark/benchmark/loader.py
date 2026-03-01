from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def _ensure_cwd_on_syspath() -> None:
    """
    让 console_script 方式启动时也能 import 当前目录下的 agent 文件（如 min_agent.py）。
    不影响正常的已安装模块导入。
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def load_object(spec: str) -> Any:
    """
    spec 格式： "some.module:ClassName" 或 "some.module:callable"
    也支持当前目录下的 <module>.py（因为我们会把 cwd 加入 sys.path）。
    """
    if ":" not in spec:
        raise ValueError(f"Agent spec must be like module:Class, got: {spec}")
    mod, name = spec.split(":", 1)

    _ensure_cwd_on_syspath()

    m = importlib.import_module(mod)
    return getattr(m, name)


def load_agent(spec: str) -> Any:
    obj = load_object(spec)
    if isinstance(obj, type):
        return obj()
    if callable(obj):
        return obj()
    raise TypeError(f"Loaded object is not a class/callable: {spec}")
