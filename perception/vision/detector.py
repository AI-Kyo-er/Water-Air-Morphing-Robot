"""视觉目标检测接口。

面向树莓派部署的轻量检测模块。提供统一接口 `TargetDetector`：
    - `YoloDetector`：本地 YOLO 权重推理，支持 ONNX Runtime / Ultralytics；
    - `ColorDetector`：无需权重的经典基线（颜色阈值 + 连通域），用于快速验证
      与回退。

输出统一为检测框列表，供融合层与上层调度使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    class_id: int = 0


class TargetDetector:
    """检测器接口。"""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class ColorDetector(TargetDetector):
    """基于颜色阈值的经典检测基线（无需模型权重）。

    对 HSV 空间中指定色相范围的目标进行阈值分割，取最大连通域作为检测结果。
    """

    def __init__(self, hsv_lower=(100, 80, 50), hsv_upper=(130, 255, 255),
                 min_area: int = 200):
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> list[Detection]:
        import cv2  # 延迟导入，便于无 OpenCV 环境下导入本模块

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: list[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            confidence = float(np.clip(area / max(w * h, 1), 0, 1))
            dets.append(Detection(x, y, x + w, y + h, "target", confidence))
        return dets


class YoloDetector(TargetDetector):
    """本地权重检测，默认 CPU；ONNX 明确指定导出输出格式与类别表。"""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 320,
                 backend: str = "onnxruntime", class_names: tuple[str, ...] = ("target",),
                 output_format: str = "raw_v8", iou: float = 0.45,
                 max_detections: int = 100, allowed_classes: tuple[int, ...] = ()):
        from .yolo_backend import YoloBackend

        self._backend = YoloBackend(weights, backend, class_names, output_format,
                                    conf, iou, imgsz, max_detections, allowed_classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """接收 BGR uint8 图像，返回原图像素坐标框。"""
        return self._backend.detect(frame)
