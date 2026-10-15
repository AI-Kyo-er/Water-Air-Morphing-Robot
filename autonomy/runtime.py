"""感知—任务—分配的主机侧协调入口，不直接写入飞控或 PWM。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from control.cm_uav.constrained_allocation import AllocationResult, ConstrainedAllocator
from perception.fusion.reliability import FusionResult, ReliabilityFusion, SensorEvidence

from .mission import BrainCommand, MissionManager, Observation


@dataclass
class RuntimeResult:
    """完整一周期的感知、决策与执行器分配结果。"""

    perception: FusionResult
    brain: BrainCommand
    allocation: AllocationResult | None


class LayeredRuntime:
    """以时间戳观测协调三个模块；控制目标由外部小脑控制器生成。

    感知/任务可低频调用，实际控制器必须按各自周期运行。
    分配使用实测介质而非任务命令，避免把入水意图当作已入水。
    """

    def __init__(self, fusion: ReliabilityFusion, mission: MissionManager,
                 allocator: ConstrainedAllocator):
        self.fusion, self.mission, self.allocator = fusion, mission, allocator
        self.medium_name = "air"
        self._last_time = -np.inf

    def step(self, observation: Observation, evidence: list[SensorEvidence],
             desired_wrench: np.ndarray | Callable[[BrainCommand, Observation], np.ndarray],
             previous: np.ndarray, dt_s: float,
             health: np.ndarray | None = None) -> RuntimeResult:
        """传入小脑回调可在本周期依据新任务生成广义力，单位 N 与 N·m。"""
        now = observation.timestamp_s
        if not np.all(np.isfinite([now, dt_s])) or now < self._last_time or dt_s <= 0:
            raise ValueError("时间应单调且 dt_s 必须为正")
        self._last_time = now
        perception = self.fusion.update(evidence, now)
        accepted = [ev for ev in evidence if perception.weights.get(ev.name, 0) > 0]
        age = min((now - ev.timestamp_s for ev in accepted), default=
                  self.mission.config.sensor_timeout_s + dt_s)
        obs = replace(observation, target_posterior=perception.posterior,
                      target_declared=perception.declared,
                      sensor_support=perception.support, sensor_age_s=age)
        brain = self.mission.update(obs)
        band = self.mission.config.surface_band_m
        if observation.z_m < -band:
            self.medium_name = "water"
        elif observation.z_m > band:
            self.medium_name = "air"
        if brain.mode == "standby":
            return RuntimeResult(perception, brain, None)
        target = desired_wrench(brain, obs) if callable(desired_wrench) else desired_wrench
        allocation = self.allocator.allocate(
            target, previous, self.allocator.params.medium(self.medium_name),
            dt_s, health=health, air_fraction=float(self.medium_name == "air"))
        return RuntimeResult(perception, brain, allocation)
