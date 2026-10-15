"""从任意工作目录执行正式测试和旧版最小验证。"""

from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """运行 unittest 模块及现有函数式测试，失败时返回非零状态。"""
    suite = unittest.TestSuite()
    for module in ("control.tests.test_constrained_allocation",
                   "perception.tests.test_reliability", "autonomy.tests.test_mission",
                   "scripts.tests.test_telemetry", "perception.tests.test_vision_pipeline",
                   "autonomy.tests.test_navigation", "autonomy.tests.test_brain"):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = 0
    for relative in ("control/tests/test_allocation.py", "control/tests/test_dynamics.py",
                     "perception/tests/test_detectors.py"):
        scope = runpy.run_path(str(ROOT / relative))
        for name, fn in sorted(scope.items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn()
                    print(f"通过 {relative}::{name}")
                except Exception as exc:
                    failures += 1
                    print(f"失败 {relative}::{name}: {exc}")
    if not result.wasSuccessful() or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
