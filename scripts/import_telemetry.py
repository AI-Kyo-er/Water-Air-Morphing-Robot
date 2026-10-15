"""将后续采集的控制日志标准化为统一 CSV，并计算跟踪与执行器指标。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.research_common import write_csv

REQUIRED = ("time_s", "x_m", "y_m", "z_m", "ref_x_m", "ref_y_m", "ref_z_m",
            "roll_rad", "pitch_rad", "yaw_rad", "thrust_1_n", "thrust_2_n",
            "thrust_3_n", "thrust_4_n", "tilt_1_rad", "tilt_2_rad",
            "tilt_3_rad", "tilt_4_rad")


def read_telemetry(path: Path) -> dict[str, np.ndarray]:
    """严格检查必要列、有限数值和单调时间，不生成缺失的测量值。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"缺少必要列：{sorted(missing)}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("至少需要两行遥测记录")
    data = {name: np.array([float(row[name]) for row in rows]) for name in REQUIRED}
    if not np.all(np.isfinite(np.column_stack(list(data.values())))):
        raise ValueError("必要列必须为有限数值")
    if data["time_s"][0] < 0 or np.any(np.diff(data["time_s"]) <= 0):
        raise ValueError("时间必须非负且严格递增")
    if np.any(np.column_stack([data[f"thrust_{i}_n"] for i in range(1, 5)]) < 0):
        raise ValueError("当前模型的推力记录必须非负")
    return data


def telemetry_metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    """从输入记录计算指标；不将电气功率与模型诱导功率混用。"""
    position = np.column_stack([data[f"{axis}_m"] for axis in "xyz"])
    reference = np.column_stack([data[f"ref_{axis}_m"] for axis in "xyz"])
    error = np.linalg.norm(position - reference, axis=1)
    tilt = np.column_stack([data[f"tilt_{i}_rad"] for i in range(1, 5)])
    time_delta = np.diff(data["time_s"])
    rate = np.diff(tilt, axis=0) / time_delta[:, None]
    return {"duration_s": float(data["time_s"][-1] - data["time_s"][0]),
            "position_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "position_p95_error_m": float(np.percentile(error, 95)),
            "depth_rmse_m": float(np.sqrt(np.mean((position[:, 2] - reference[:, 2]) ** 2))),
            "tilt_rate_max_deg_s": float(np.rad2deg(np.abs(rate).max()))}


def main() -> None:
    """显式声明数据来源，把真实日志接入同一套分析入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-kind", choices=("measured", "development"), required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output_dir.resolve()
    if output / "telemetry.csv" == source:
        raise ValueError("输出不能覆盖输入文件")
    data = read_telemetry(source)
    rows = [{name: float(values[k]) for name, values in data.items()}
            for k in range(len(data["time_s"]))]
    write_csv(output / "telemetry.csv", rows)
    report = {"session_id": args.session_id, "source_kind": args.source_kind,
              "source_declaration": "operator_supplied", "rows": len(rows),
              "metrics": telemetry_metrics(data)}
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
