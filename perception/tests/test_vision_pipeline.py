"""视觉导出后处理、目标关联、质量与当帧证据的正式验证。"""

import tempfile
import subprocess
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from perception.vision.detector import Detection, YoloDetector
from perception.vision.pipeline import VisionPipeline
from perception.vision.tracking import TargetTracker
from perception.vision.yolo_backend import decode_yolo, letterbox

ROOT = Path(__file__).resolve().parents[2]


class VisionTests(unittest.TestCase):
    def decode(self, output, fmt="raw_v8", names=("target",), shape=(320, 320, 3),
               scale=1, pad=(0, 0), allowed=()):
        return decode_yolo(np.array(output), shape, scale, pad, names, fmt,
                           0.25, 0.45, 100, allowed)

    def test_v8_letterbox_coordinate_restoration(self):
        result = self.decode([[[160, 160, 100, 50, 0.8]]], shape=(320, 640, 3),
                             scale=0.5, pad=(0, 80))
        self.assertEqual((result[0].x1, result[0].y1, result[0].x2, result[0].y2),
                         (220, 110, 420, 210))

    def test_v5_objectness_multiplies_class_score(self):
        result = self.decode([[[160, 160, 50, 50, 0.9, 0.8]]], "raw_v5")
        self.assertAlmostEqual(result[0].confidence, 0.72)

    def test_transposed_v8_and_empty_output(self):
        output = np.array([[160, 160, 50, 50, 0.8], [200, 200, 20, 20, 0.1]]).T[None]
        self.assertEqual(len(self.decode(output)), 1)
        self.assertEqual(self.decode(np.empty((1, 0, 5))), [])

    def test_class_aware_nms_and_filter(self):
        output = [[[10, 10, 80, 80, 0.9, 0], [11, 11, 79, 79, 0.8, 0],
                   [10, 10, 80, 80, 0.85, 1]]]
        result = self.decode(output, "xyxy6", ("a", "b"))
        self.assertEqual(len(result), 2)
        self.assertEqual(len(self.decode(output, "xyxy6", ("a", "b"), allowed=(1,))), 1)

    def test_bad_output_layout_rejected(self):
        with self.assertRaises(ValueError):
            self.decode(np.zeros((1, 2, 84)))

    def test_nan_and_degenerate_boxes_discarded(self):
        self.assertEqual(self.decode([[[0, 0, 0, 1, 0.9, 0], [0, 0, 30, 30, np.nan, 0]]],
                                     "xyxy6"), [])

    def test_letterbox_tensor_rgb_and_shape(self):
        frame = np.zeros((160, 320, 3), np.uint8)
        frame[:] = (10, 20, 30)
        tensor, scale, padding = letterbox(frame, 320)
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual(padding, (0, 80))
        np.testing.assert_allclose(tensor[0, :, 100, 100], np.array([30, 20, 10])/255)

    def test_backend_dispatch_with_explicit_inference_fixture(self):
        """用固定会话输出检验调度，不能作为真实模型推理准确率验证。"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.onnx"
            path.write_bytes(b"test-session-fixture")
            detector = YoloDetector(str(path))
            session = Mock()
            session.get_inputs.return_value = [Mock(name="images", type="tensor(float)")]
            session.get_inputs.return_value[0].name = "images"
            session.run.return_value = [np.array([[[160, 160, 40, 40, 0.9]]])]
            detector._backend.model = session
            result = detector.detect(np.zeros((320, 320, 3), np.uint8))
            self.assertEqual(len(result), 1)
            self.assertEqual(session.run.call_args.args[1]["images"].dtype, np.float32)

    def test_missing_model_fails_without_download(self):
        with self.assertRaises(FileNotFoundError):
            YoloDetector("missing-local-weight.onnx")

    def test_track_identity_confirmation_and_expiry(self):
        tracker = TargetTracker(min_hits=2)
        a = tracker.update([Detection(10, 10, 30, 30, "target", 0.9)], 0)[0]
        b = tracker.update([Detection(12, 10, 32, 30, "target", 0.9)], 0.1)[0]
        self.assertEqual(a.track_id, b.track_id)
        self.assertTrue(b.confirmed)
        self.assertFalse(tracker.update([], 0.2)[0].visible)
        self.assertEqual(tracker.update([], 1.0), [])

    def test_labels_are_not_cross_associated(self):
        tracker = TargetTracker(min_hits=1)
        tracker.update([Detection(10, 10, 30, 30, "a", 0.9)], 0)
        result = tracker.update([Detection(10, 10, 30, 30, "b", 0.9)], 0.1)
        self.assertEqual(len(result), 2)

    def test_quality_and_confirmation_evidence(self):
        import cv2
        pipeline = VisionPipeline.from_yaml(ROOT / "perception/configs/vision.yaml")
        frame = np.full((240, 320, 3), 100, np.uint8)
        cv2.rectangle(frame, (110, 80), (190, 160), (255, 0, 0), -1)
        pipeline.process(frame, 0)
        pipeline.process(frame, 0.1)
        result = pipeline.process(frame, 0.2)
        self.assertGreater(result.evidence.score, 0)
        self.assertTrue(0 < result.quality <= 1)
        absent = pipeline.process(np.full_like(frame, 100), 0.3)
        self.assertEqual(absent.evidence.score, 0)
        self.assertIsNone(absent.target_bearing_rad)

    def test_image_cli_writes_detection_record(self):
        import cv2
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "marker.png"
            frame = np.full((240, 320, 3), 100, np.uint8)
            cv2.rectangle(frame, (110, 80), (190, 160), (255, 0, 0), -1)
            cv2.imwrite(str(source), frame)
            subprocess.run([sys.executable, str(ROOT / "perception/scripts/run_vision.py"),
                            "--source", str(source), "--output-dir", str(directory / "output")],
                           check=True, capture_output=True)
            records = (directory / "output/vision.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["detections"][0]["label"], "target")
            self.assertTrue((directory / "output/frame_000000.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
