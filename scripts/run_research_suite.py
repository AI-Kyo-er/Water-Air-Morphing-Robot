"""一键生成控制/感知基准、结果表与图件。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    "control/scripts/run_allocation_benchmark.py",
    "perception/scripts/run_reliability_benchmark.py",
    "autonomy/scripts/run_brain_demo.py",
    "scripts/build_results.py",
)


def main() -> None:
    """使用当前 Python 环境串行生成结果，子任务失败即报告。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for step in STEPS:
        print(f"执行：{step}", flush=True)
        subprocess.run([sys.executable, str(ROOT / step)], check=True, env=env)
    print("结果与图件已生成到 experiments/。")


if __name__ == "__main__":
    main()
