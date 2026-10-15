"""按类别、预测框与空间门限关联目标，保留短时轨迹 ID。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection
from .yolo_backend import box_iou


@dataclass
class Track:
    """图像坐标轨迹；预测状态不自动作为新的检测证据。"""

    track_id: int
    label: str
    box: np.ndarray
    velocity_px_s: np.ndarray
    confidence: float
    last_seen_s: float
    hit_streak: int = 1
    confirmed: bool = False
    visible: bool = True


class TargetTracker:
    """常速度预测加匈牙利关联，适用于轻量多目标跟踪。"""

    def __init__(self, min_hits: int = 3, max_age_s: float = 0.6,
                 iou_threshold: float = 0.15, center_gate_px: float = 80,
                 confidence_alpha: float = 0.4):
        if min_hits < 1 or max_age_s <= 0 or center_gate_px <= 0:
            raise ValueError("跟踪参数必须为正")
        if not 0 <= iou_threshold <= 1 or not 0 < confidence_alpha <= 1:
            raise ValueError("IoU 和置信度平滑参数无效")
        self.min_hits, self.max_age_s = min_hits, max_age_s
        self.iou_threshold, self.center_gate_px = iou_threshold, center_gate_px
        self.alpha = confidence_alpha
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_time: float | None = None

    def update(self, detections: list[Detection], now_s: float) -> list[Track]:
        """按严格递增采集时间更新，返回仍有效的可见/短时预测轨迹。"""
        if not np.isfinite(now_s) or now_s < 0 or (self._last_time is not None and now_s <= self._last_time):
            raise ValueError("跟踪时间应非负且严格递增")
        for det in detections:
            if not np.all(np.isfinite([det.x1, det.y1, det.x2, det.y2, det.confidence])) or (
                    det.x2 <= det.x1 or det.y2 <= det.y1 or not 0 <= det.confidence <= 1):
                raise ValueError("检测框或置信度无效")
        dt = 0 if self._last_time is None else now_s - self._last_time
        self._last_time = now_s
        self.tracks = [tr for tr in self.tracks if now_s - tr.last_seen_s <= self.max_age_s]
        old_boxes = [tr.box.copy() for tr in self.tracks]
        for tr in self.tracks:
            tr.box += np.tile(tr.velocity_px_s * dt, 2)
            tr.visible = False
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=float).reshape(-1, 4)
        cost = np.full((len(self.tracks), len(detections)), 1e6)
        for i, tr in enumerate(self.tracks):
            if len(boxes):
                overlap = box_iou(tr.box, boxes)
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                center = (tr.box[:2] + tr.box[2:]) / 2
                distances = np.linalg.norm(centers - center, axis=1)
                for j, det in enumerate(detections):
                    if det.label == tr.label and (overlap[j] >= self.iou_threshold or
                                                  distances[j] <= self.center_gate_px):
                        cost[i, j] = 1 - overlap[j] + distances[j] / self.center_gate_px
        matched_detections = set()
        if cost.size:
            ii, jj = linear_sum_assignment(cost)
            for i, j in zip(ii, jj):
                if cost[i, j] >= 1e6:
                    continue
                tr, det = self.tracks[i], detections[j]
                old_center = (old_boxes[i][:2] + old_boxes[i][2:]) / 2
                new_center = (boxes[j, :2] + boxes[j, 2:]) / 2
                tr.velocity_px_s = (new_center - old_center) / max(dt, 1e-9)
                tr.box = boxes[j].copy()
                tr.confidence = (1-self.alpha) * tr.confidence + self.alpha * det.confidence
                tr.last_seen_s, tr.visible = now_s, True
                tr.hit_streak += 1
                tr.confirmed |= tr.hit_streak >= self.min_hits
                matched_detections.add(j)
        for tr in self.tracks:
            if not tr.visible:
                tr.hit_streak = 0
        for j, det in enumerate(detections):
            if j not in matched_detections:
                self.tracks.append(Track(self._next_id, det.label, boxes[j].copy(),
                                         np.zeros(2), det.confidence, now_s,
                                         confirmed=self.min_hits == 1))
                self._next_id += 1
        return self.tracks
