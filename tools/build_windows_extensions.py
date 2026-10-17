"""在当前 worktree 构建 Windows 扩展，避免共享入口写死旧源码路径。"""

import os
from pathlib import Path
import runpy

import torch.utils.cpp_extension as cpp_extension


def main():
    """保留 Torch 编译器检查，仅修正无控制台进程的 MSVC 输出解码。"""
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    runpy.run_path(str(root / "setup.py"), run_name="__main__")


if __name__ == "__main__":
    main()
