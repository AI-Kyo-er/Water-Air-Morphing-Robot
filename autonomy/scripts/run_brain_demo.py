"""视觉标记检测到任务确认、导航暂停及返航的统一大脑开发演示。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autonomy.brain import PerceptionNavigationBrain
from autonomy.mission import MissionConfig, MissionManager, Observation
from autonomy.navigation.navigator import Navigator
from perception.fusion.reliability import FusionConfig, GaussianCalibration, ReliabilityFusion
from perception.vision.pipeline import VisionPipeline, annotate
from scripts.research_common import write_csv, write_metadata

OUT = ROOT / "experiments/system/vision_navigation"


def main() -> None:
    """颜色基线运行完整链路；真实 YOLO 权重沿相同接口接入。"""
    import cv2

    vision_config = ROOT / "perception/configs/vision.yaml"
    nav_config = ROOT / "autonomy/configs/navigation.yaml"
    demo_config = ROOT / "autonomy/configs/brain_demo.yaml"
    raw = yaml.safe_load(demo_config.read_text(encoding="utf-8"))
    mission_config = MissionConfig.from_yaml(ROOT / "autonomy/configs/mission.yaml")
    navigator = Navigator.from_yaml(nav_config)
    navigator.grid.set_known_free()
    brain = PerceptionNavigationBrain(
        VisionPipeline.from_yaml(vision_config),
        ReliabilityFusion({"vision": GaussianCalibration(**raw["calibration"])},
                          FusionConfig.from_yaml(ROOT / "perception/configs/reliability.yaml")),
        MissionManager(replace(mission_config, initial_state=raw["initial_state"])), navigator,
        {key: np.array(value) for key, value in raw["goals"].items()})
    rng = np.random.default_rng(raw["seed"])
    pose = np.array(raw["start_pose_xy_yaw"], dtype=float)
    speed, rate, dt = 0.0, 0.0, raw["dt_s"]
    rows = []
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for index in range(raw["max_steps"]):
        t = index*dt
        width, height = raw["image_size_wh"]
        frame = np.clip(raw["background_mean"]+rng.normal(
            0, raw["background_std"], (height, width, 3)), 0, 255).astype(np.uint8)
        if raw["marker_interval_s"][0] <= t < raw["marker_interval_s"][1]:
            x, y, w, h = raw["marker_xywh"]
            x += int(raw["marker_motion_amplitude_px"]*np.sin(t))
            cv2.rectangle(frame, (x, y), (x+w, y+h), tuple(raw["marker_color_bgr"]), -1)
        progress = np.clip((t-raw["exit_start_s"])/raw["exit_duration_s"], 0, 1)
        z = -mission_config.survey_depth_m + progress*(
            mission_config.air_altitude_m+mission_config.survey_depth_m)
        result = brain.step(Observation(t, z, max(0, -z)), pose, speed, rate, dt, frame=frame)
        rows.append({"time_s": round(t, 3), "source_kind": "simulation",
                     "detector_backend": "color", "x_m": pose[0], "y_m": pose[1],
                     "target_posterior": result.fusion.posterior,
                     "target_declared": int(result.fusion.declared),
                     "mission_state": result.mission.state,
                     "navigation_status": result.navigation.status,
                     "speed_m_s": result.navigation.command.speed_m_s,
                     "yaw_rate_rad_s": result.navigation.command.yaw_rate_rad_s,
                     "vision_quality": result.vision.quality,
                     "confirmed_tracks": sum(tr.confirmed and tr.visible for tr in result.vision.tracks)})
        if index in raw["snapshot_indices"]:
            cv2.imwrite(str(figure_dir / f"tracked_marker_{index:03d}.jpg"), annotate(frame, result.vision))
        speed, rate = result.navigation.command.speed_m_s, result.navigation.command.yaw_rate_rad_s
        pose[:2] += speed*dt*np.array([np.cos(pose[2]+rate*dt/2), np.sin(pose[2]+rate*dt/2)])
        pose[2] += rate*dt
        if result.mission.state == "complete":
            break
    write_csv(OUT / "data/brain_trace.csv", rows)
    write_csv(OUT / "data/task_events.csv", brain.mission.events)
    write_metadata(OUT / "run_metadata.json", [vision_config, nav_config, demo_config, Path(__file__),
                   ROOT / "autonomy/brain.py", ROOT / "perception/vision/pipeline.py",
                   ROOT / "perception/vision/tracking.py", ROOT / "perception/vision/detector.py",
                   ROOT / "autonomy/navigation/navigator.py", ROOT / "autonomy/navigation/planner.py",
                   ROOT / "autonomy/configs/mission.yaml", ROOT / "perception/configs/reliability.yaml"], [raw["seed"]],
                   "视觉标记检测、目标确认、导航暂停与返航的端到端流程。")
    print(f"视觉导航链完成 {len(rows)} 帧，状态：{result.mission.state}")


if __name__ == "__main__":
    main()
