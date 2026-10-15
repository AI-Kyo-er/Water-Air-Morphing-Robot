"""视觉、可信融合、任务管理与导航在树莓派侧的统一大脑入口。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from perception.fusion.reliability import FusionResult, ReliabilityFusion, SensorEvidence
from perception.vision.pipeline import VisionPipeline, VisionResult
from .mission import BrainCommand, MissionManager, Observation, MissionState
from .navigation.navigator import Navigator, NavigationResult


@dataclass
class BrainOutput:
    """大脑完整输出；导航速度为沿路径目标，偏航率交小脑跟踪。"""

    vision: VisionResult | None
    fusion: FusionResult
    mission: BrainCommand
    navigation: NavigationResult


class PerceptionNavigationBrain:
    """接收外部位姿与可选相机帧，输出任务及可行路径目标。"""

    def __init__(self, vision: VisionPipeline | None, fusion: ReliabilityFusion,
                 mission: MissionManager, navigator: Navigator,
                 goals: dict[str, np.ndarray]):
        self.vision, self.fusion, self.mission, self.navigator = vision, fusion, mission, navigator
        self.goals = {key: np.asarray(value, float).copy() for key, value in goals.items()}
        if any(goal.shape != (2,) or not np.all(np.isfinite(goal)) for goal in self.goals.values()):
            raise ValueError("任务路径目标应为有限二维坐标")

    def step(self, observation: Observation, pose_xy_yaw: np.ndarray,
             speed_m_s: float, yaw_rate_rad_s: float, dt_s: float,
             frame: np.ndarray | None = None,
             evidence: list[SensorEvidence] | None = None) -> BrainOutput:
        """先更新感知与任务，再按任务目标进行规划；相机缺帧不复用旧证据。"""
        pose = np.asarray(pose_xy_yaw, float)
        if pose.shape != (3,) or not np.all(np.isfinite([*pose, dt_s])) or dt_s <= 0:
            raise ValueError("位姿与周期无效")
        current = list(evidence or [])
        visual = None
        if frame is not None:
            if self.vision is None:
                raise ValueError("未配置视觉模块")
            visual = self.vision.process(frame, observation.timestamp_s)
            current.append(visual.evidence)
        fusion = self.fusion.update(current, observation.timestamp_s)
        active = [ev for ev in current if fusion.weights.get(ev.name, 0) > 0]
        age = min((observation.timestamp_s-ev.timestamp_s for ev in active), default=
                  self.mission.config.sensor_timeout_s+dt_s)
        flags = {}
        state = self.mission.state.value
        if state in self.goals:
            arrived = np.linalg.norm(pose[:2]-self.goals[state]) <= self.navigator.config.goal_tolerance_m
            if state == "air_transit":
                flags["at_search_area"] = arrived
            elif state == "survey":
                flags["survey_complete"] = arrived
            elif state == "return":
                flags["at_home"] = arrived
        obs = replace(observation, target_posterior=fusion.posterior,
                      target_declared=fusion.declared, sensor_support=fusion.support,
                      sensor_age_s=age, **flags)
        command = self.mission.update(obs)
        enabled = command.state in self.goals and command.state in (
            MissionState.AIR_TRANSIT.value, MissionState.SURVEY.value, MissionState.RETURN.value)
        if enabled:
            self.navigator.set_goal(self.goals[command.state])
        speed_limit = command.forward_speed_m_s if command.forward_speed_m_s > 0 else None
        navigation = self.navigator.step(pose, speed_m_s, yaw_rate_rad_s, observation.timestamp_s,
                                         dt_s, enabled=enabled, speed_limit=speed_limit)
        if enabled:
            command = replace(command, forward_speed_m_s=navigation.command.speed_m_s)
        return BrainOutput(visual, fusion, command, navigation)
