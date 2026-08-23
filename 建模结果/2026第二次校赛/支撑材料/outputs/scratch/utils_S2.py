# -*- coding: utf-8 -*-
"""utils-S2.py 的 Python 可导入别名（文件名带连字符无法直接 import）。

verify-S2-* 与 S3-e3-fewshot.py 从 utils 导入的函数实际定义于 utils-S2.py，
本模块转发 utils-S2.py 的全部命名空间，供 `from utils_S2 import ...` 使用。
"""
import importlib.util
import sys
from pathlib import Path

_impl = Path(__file__).resolve().parent / "utils-S2.py"
_spec = importlib.util.spec_from_file_location("_utils_s2_impl", _impl)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules["_utils_s2_impl"] = _mod

# 转发全部公开名称
for _name in dir(_mod):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_mod, _name)
