"""研究基准的 CSV、元数据和指标公共工具。"""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    """写入小体积 UTF-8 CSV，保留明确列名与单位。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, configs: list[Path], seeds: list[int],
                   description: str) -> None:
    """保存配置、随机种子与运行环境；不存令牌、主机标识或私有绝对路径。"""
    metadata = {
        "description": description,
        "seeds": seeds,
        "configs": [str(p.relative_to(ROOT)).replace("\\", "/") for p in configs],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def classification_metrics(truth: np.ndarray, probability: np.ndarray,
                           prediction: np.ndarray) -> dict[str, float]:
    """计算帧级检测指标；时间迟滞引入的延迟同样计入误差。"""
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    tp = np.sum(truth & prediction)
    fp = np.sum(~truth & prediction)
    fn = np.sum(truth & ~prediction)
    tn = np.sum(~truth & ~prediction)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "brier_score": float(np.mean((probability - truth.astype(float)) ** 2)),
            "decision_switches": int(np.count_nonzero(np.diff(prediction.astype(int))))}
