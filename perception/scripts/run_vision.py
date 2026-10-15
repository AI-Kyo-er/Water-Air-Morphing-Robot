"""从摄像头、视频或图片运行视觉检测与跟踪，保存逐帧 JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception.vision.pipeline import VisionPipeline, annotate


def main() -> None:
    """输入通过 --source 指定，模型仅从本地配置加载。"""
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "perception/configs/vision.yaml")
    parser.add_argument("--source", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=30)
    args = parser.parse_args()
    if args.max_frames < 1 or args.save_every < 1:
        raise ValueError("帧数与保存间隔应为正")
    pipeline = VisionPipeline.from_yaml(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.source
    still = Path(source).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    capture = None if still else cv2.VideoCapture(int(source) if source.isdigit() else source)
    if capture is not None and not capture.isOpened():
        capture.release()
        raise RuntimeError("不能打开相机或视频")
    fps = capture.get(cv2.CAP_PROP_FPS) if capture is not None else 0
    started = time.monotonic()
    try:
        with (args.output_dir / "vision.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(args.max_frames):
                if still:
                    frame = cv2.imread(source)
                    if frame is None:
                        raise RuntimeError("不能读取图片")
                else:
                    success, frame = capture.read()
                    if not success:
                        break
                timestamp = index / fps if not source.isdigit() and fps > 0 else time.monotonic() - started
                before = time.perf_counter()
                result = pipeline.process(frame, timestamp)
                record = {"frame_index": index, "timestamp_s": timestamp,
                          "inference_ms": (time.perf_counter()-before)*1000,
                          "quality": result.quality, "bearing_rad": result.target_bearing_rad,
                          "detections": [asdict(det) for det in result.detections],
                          "evidence": asdict(result.evidence),
                          "tracks": [{"id": tr.track_id, "label": tr.label,
                                      "box": tr.box.tolist(), "confirmed": tr.confirmed,
                                      "visible": tr.visible} for tr in result.tracks]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % args.save_every == 0:
                    cv2.imwrite(str(args.output_dir / f"frame_{index:06d}.jpg"), annotate(frame, result))
                if still:
                    break
    finally:
        if capture is not None:
            capture.release()


if __name__ == "__main__":
    main()
