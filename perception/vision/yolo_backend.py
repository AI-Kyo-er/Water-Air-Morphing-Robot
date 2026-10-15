"""本地 YOLO 推理与导出后处理，不隐式下载模型或启用 GPU。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .detector import Detection


def validate_frame(frame: np.ndarray) -> None:
    """校验 BGR uint8 图像。"""
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("输入必须是非空 H×W×3 BGR 图像")
    if frame.dtype != np.uint8:
        raise ValueError("输入图像应为 uint8")


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比例缩放并填充为方形，返回 NCHW RGB float32 张量。"""
    import cv2

    validate_frame(frame)
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = cv2.resize(frame, (nw, nh))
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, scale, (left, top)


def box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """计算 xyxy 框的交并比。"""
    low = np.maximum(box[:2], others[:, :2])
    high = np.minimum(box[2:], others[:, 2:])
    intersection = np.prod(np.maximum(high - low, 0), axis=1)
    area = np.prod(np.maximum(box[2:] - box[:2], 0))
    areas = np.prod(np.maximum(others[:, 2:] - others[:, :2], 0), axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-12)


def decode_yolo(output: np.ndarray, frame_shape: tuple[int, ...], scale: float,
                padding: tuple[int, int], class_names: tuple[str, ...],
                output_format: str, conf: float, iou: float, max_detections: int,
                allowed_classes: tuple[int, ...] = ()) -> list[Detection]:
    """支持 raw_v5、raw_v8、xyxy6，置信度及坐标由明确导出格式解释。"""
    raw = np.asarray(output, dtype=float)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2:
        raise ValueError("检测输出应为二维或 batch=1 三维张量")
    columns = {"raw_v5": 5 + len(class_names), "raw_v8": 4 + len(class_names), "xyxy6": 6}
    if output_format not in columns:
        raise ValueError("未知 YOLO 输出格式")
    expected = columns[output_format]
    if raw.shape[1] != expected and raw.shape[0] == expected:
        raw = raw.T
    if raw.shape[1] != expected:
        raise ValueError(f"输出维度 {raw.shape} 与类别表/格式不一致，需要 {expected} 列")
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    if output_format == "xyxy6":
        boxes, scores = raw[:, :4].copy(), raw[:, 4]
        valid_class = (raw[:, 5] >= 0) & (raw[:, 5] < len(class_names)) & (raw[:, 5] == np.floor(raw[:, 5]))
        raw, boxes, scores = raw[valid_class], boxes[valid_class], scores[valid_class]
        classes = raw[:, 5].astype(int)
    else:
        start = 5 if output_format == "raw_v5" else 4
        classes = np.argmax(raw[:, start:], axis=1)
        scores = raw[np.arange(len(raw)), start + classes]
        if output_format == "raw_v5":
            scores = scores * raw[:, 4]
        boxes = raw[:, :4].copy()
        boxes[:, :2] = raw[:, :2] - raw[:, 2:4] / 2
        boxes[:, 2:4] = raw[:, :2] + raw[:, 2:4] / 2
    keep = (scores >= conf) & (scores <= 1)
    if allowed_classes:
        keep &= np.isin(classes, allowed_classes)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / scale
    h, w = frame_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    order = np.argsort(-scores, kind="stable")
    selected = []
    while len(order) and len(selected) < max_detections:
        index = order[0]
        selected.append(index)
        rest = order[1:]
        suppress = (classes[rest] == classes[index]) & (box_iou(boxes[index], boxes[rest]) > iou)
        order = rest[~suppress]
    return [Detection(int(np.floor(boxes[i, 0])), int(np.floor(boxes[i, 1])),
                      int(np.ceil(boxes[i, 2])), int(np.ceil(boxes[i, 3])),
                      class_names[classes[i]], float(scores[i]), int(classes[i])) for i in selected]


class YoloBackend:
    """延迟加载本地权重；提供 ONNX Runtime、OpenCV DNN 与 Ultralytics 后端。"""

    def __init__(self, weights: str, backend: str, names: tuple[str, ...],
                 output_format: str, conf: float, iou: float, imgsz: int,
                 max_detections: int, allowed_classes: tuple[int, ...]):
        self.weights = Path(weights).expanduser()
        if not self.weights.is_file():
            raise FileNotFoundError(f"请提供本地模型权重：{self.weights}")
        if backend not in ("onnxruntime", "opencv", "ultralytics"):
            raise ValueError("未知推理后端")
        if not names or not 0 < conf <= 1 or not 0 < iou <= 1 or imgsz < 32 or max_detections < 1:
            raise ValueError("检测配置无效")
        if any(c < 0 or c >= len(names) for c in allowed_classes):
            raise ValueError("类别过滤索引超出类别表")
        if backend != "ultralytics" and (self.weights.suffix.lower() != ".onnx" or
                                         output_format not in ("raw_v5", "raw_v8", "xyxy6")):
            raise ValueError("ONNX 后端需要 .onnx 文件及明确输出格式")
        self.backend, self.names, self.output_format = backend, names, output_format
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.max_detections, self.allowed_classes = max_detections, allowed_classes
        self.model = None

    def _load(self) -> None:
        if self.backend == "onnxruntime":
            import onnxruntime as ort
            self.model = ort.InferenceSession(str(self.weights), providers=["CPUExecutionProvider"])
        elif self.backend == "opencv":
            import cv2
            self.model = cv2.dnn.readNetFromONNX(str(self.weights))
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights), task="detect")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """在 CPU 上推理，返回原图坐标检测框。"""
        validate_frame(frame)
        if self.model is None:
            self._load()
        if self.backend == "ultralytics":
            result = self.model.predict(source=frame, device="cpu", conf=self.conf, iou=self.iou,
                                        imgsz=self.imgsz, max_det=self.max_detections,
                                        classes=list(self.allowed_classes) or None, verbose=False)[0]
            if result.boxes is None:
                return []
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            packed = np.column_stack([boxes, scores, classes])
            # 后端已在原图空间输出框；复用验证与类别内 NMS。
            return decode_yolo(packed, frame.shape, 1, (0, 0), self.names, "xyxy6",
                               self.conf, self.iou, self.max_detections, self.allowed_classes)
        tensor, scale, padding = letterbox(frame, self.imgsz)
        if self.backend == "onnxruntime":
            inputs = self.model.get_inputs()
            if len(inputs) != 1 or inputs[0].type != "tensor(float)":
                raise ValueError("目前支持单个 float32 图像输入")
            outputs = self.model.run(None, {inputs[0].name: tensor})
            if len(outputs) != 1:
                raise ValueError("需要单一聚合检测输出，不能直接解释多头原始输出")
            output = outputs[0]
        else:
            self.model.setInput(tensor)
            output = self.model.forward()
        return decode_yolo(output, frame.shape, scale, padding, self.names, self.output_format,
                           self.conf, self.iou, self.max_detections, self.allowed_classes)
