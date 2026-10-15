"""相机检测、目标跟踪、图像质量与融合证据输出。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from perception.fusion.reliability import SensorEvidence
from .detector import ColorDetector, Detection, TargetDetector, YoloDetector
from .tracking import TargetTracker, Track
from .yolo_backend import validate_frame


@dataclass
class VisionResult:
    """单帧输出：检测、轨迹、质量、目标方位角与融合证据。"""

    detections: list[Detection]
    tracks: list[Track]
    quality: float
    target_bearing_rad: float | None
    evidence: SensorEvidence


def create_detector(raw: dict, config_directory: Path) -> TargetDetector:
    """构造本地 YOLO 后端；未放置权重时回退到颜色基线。

    默认配置指向 `models/` 下的本地 ONNX 权重（不入库）。若权重不存在，则退回
    无需权重的颜色基线，保证推理链、跟踪、质量与证据接口仍可端到端运行。
    """
    if raw["backend"] == "color":
        return ColorDetector(raw["hsv_lower"], raw["hsv_upper"], raw["min_area_px"])
    weights = raw.get("weights")
    weights_path = Path(weights).expanduser() if weights else None
    if weights_path is not None and not weights_path.is_absolute():
        weights_path = config_directory / weights_path
    if weights_path is None or not weights_path.is_file():
        return ColorDetector(raw["hsv_lower"], raw["hsv_upper"], raw["min_area_px"])
    return YoloDetector(str(weights_path), conf=raw["confidence"], imgsz=raw["image_size"],
                        backend=raw["backend"], class_names=tuple(raw["class_names"]),
                        output_format=raw["output_format"], iou=raw["iou"],
                        max_detections=raw["max_detections"],
                        allowed_classes=tuple(raw["allowed_classes"]))


class VisionPipeline:
    """树莓派侧单帧流水线，不把单目框大小直接解释为距离。"""

    def __init__(self, detector: TargetDetector, tracker: TargetTracker,
                 quality_config: dict, evidence_config: dict):
        self.detector, self.tracker = detector, tracker
        self.quality_config, self.evidence_config = quality_config, evidence_config
        if quality_config["blur_reference"] <= 0 or not (
            0 < quality_config["brightness_floor"] < quality_config["brightness_ceiling"] < 255):
            raise ValueError("图像质量参数无效")
        if evidence_config["score_scale"] <= 0 or not 0 < evidence_config["horizontal_fov_deg"] < 180:
            raise ValueError("视觉证据参数无效")

    @classmethod
    def from_yaml(cls, path: str | Path) -> VisionPipeline:
        """加载配置和本地模型路径。"""
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(create_detector(raw["detector"], path.parent),
                   TargetTracker(**raw["tracking"]), raw["quality"], raw["evidence"])

    def process(self, frame: np.ndarray, timestamp_s: float) -> VisionResult:
        """从 BGR 帧产生当帧证据；预测轨迹不充当新鲜检测。"""
        import cv2

        validate_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        cfg = self.quality_config
        exposure = np.clip(min(brightness / cfg["brightness_floor"],
                               (255-brightness)/(255-cfg["brightness_ceiling"])), 0, 1)
        quality = float(exposure * np.clip(sharpness / cfg["blur_reference"], 0, 1))
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, timestamp_s)
        targets = [tr for tr in tracks if tr.visible and tr.confirmed and
                   tr.label in self.evidence_config["target_labels"]]
        best = max(targets, key=lambda tr: tr.confidence, default=None)
        score = best.confidence * self.evidence_config["score_scale"] if best else 0.0
        bearing = None
        if best is not None:
            width = frame.shape[1]
            focal = width / (2 * np.tan(np.deg2rad(self.evidence_config["horizontal_fov_deg"]) / 2))
            bearing = float(np.arctan2((best.box[0]+best.box[2])/2 - width/2, focal))
        evidence = SensorEvidence("vision", score, quality, timestamp_s,
                                  available=True, correlation_group="vision")
        return VisionResult(detections, list(tracks), quality, bearing, evidence)


def annotate(frame: np.ndarray, result: VisionResult) -> np.ndarray:
    """绘制框与轨迹 ID，返回新的图像。"""
    import cv2

    output = frame.copy()
    for tr in result.tracks:
        if not tr.visible:
            continue
        x1, y1, x2, y2 = tr.box.astype(int)
        color = (40, 200, 80) if tr.confirmed else (50, 170, 230)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"{tr.label} #{tr.track_id} {tr.confidence:.2f}",
                    (x1, max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output
