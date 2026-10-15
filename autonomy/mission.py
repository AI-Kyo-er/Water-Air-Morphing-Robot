"""大脑任务状态机：以可解释规则协调感知、跨介质模式与控制目标。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import numpy as np
import yaml


class MissionState(str, Enum):
    """任务阶段与保护状态。"""

    PREFLIGHT = "preflight"
    AIR_TRANSIT = "air_transit"
    ENTER_WATER = "enter_water"
    SURVEY = "survey"
    CONFIRM = "confirm"
    EXIT_WATER = "exit_water"
    RETURN = "return"
    HOLD = "hold"
    RECOVERY = "recovery"
    COMPLETE = "complete"
    ABORT = "abort"


@dataclass(frozen=True)
class MissionConfig:
    """浅水原型任务参数；扩展作业深度须重新验证硬件。"""

    initial_state: str = "preflight"
    health_min: float = 0.55
    battery_return_fraction: float = 0.22
    max_depth_m: float = 0.5
    survey_depth_m: float = 0.3
    air_altitude_m: float = 2.0
    surface_band_m: float = 0.05
    obstacle_hold_m: float = 0.8
    link_timeout_s: float = 1.5
    sensor_timeout_s: float = 1.0
    confirm_dwell_s: float = 0.8
    hold_recover_s: float = 1.0
    enter_timeout_s: float = 8.0
    exit_timeout_s: float = 8.0
    survey_timeout_s: float = 30.0
    recovery_timeout_s: float = 10.0
    complete_altitude_m: float = 1.5
    survey_speed_m_s: float = 0.25
    return_speed_m_s: float = 0.15

    @classmethod
    def from_yaml(cls, path: str | Path) -> MissionConfig:
        """读取任务配置。"""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["mission"]
        return cls(**raw)


@dataclass(frozen=True)
class Observation:
    """一次任务观测；高度 z 向上，depth 向下，均以水面为零。"""

    timestamp_s: float
    z_m: float
    depth_m: float
    target_posterior: float = 0.0
    target_declared: bool = False
    sensor_support: float = 0.0
    sensor_age_s: float = 0.0
    actuator_health: float = 1.0
    battery_fraction: float = 1.0
    link_age_s: float = 0.0
    obstacle_distance_m: float = 10.0
    flight_ready: bool = False
    at_search_area: bool = False
    survey_complete: bool = False
    at_home: bool = False


@dataclass(frozen=True)
class BrainCommand:
    """大脑到小脑的意图，不包含直接电机 PWM。"""

    state: str
    mode: str
    target_z_m: float
    forward_speed_m_s: float
    request_confirmation: bool
    reason: str

    def to_dict(self) -> dict:
        """生成串口/日志可以传输的结构。"""
        return asdict(self)


class MissionManager:
    """由观测触发的分层任务调度器；失联后保持锁存保护状态。"""

    def __init__(self, config: MissionConfig | None = None):
        self.config = config or MissionConfig()
        cfg = self.config
        numeric = [v for k, v in asdict(cfg).items() if k != "initial_state"]
        if not np.all(np.isfinite(numeric)) or min(numeric) <= 0:
            raise ValueError("任务参数必须为有限正数")
        if cfg.survey_depth_m > cfg.max_depth_m:
            raise ValueError("巡检深度不能超过最大作业深度")
        self.state = MissionState(cfg.initial_state)
        self.entered_at = 0.0
        self.events: list[dict] = []
        self._last_time = -np.inf
        self._confirm_since: float | None = None
        self._clear_since: float | None = None

    def _transition(self, state: MissionState, now: float, reason: str) -> None:
        previous = self.state
        self.state, self.entered_at = state, now
        self._confirm_since = None
        self._clear_since = None
        self.events.append({"timestamp_s": now, "from": previous.value,
                            "to": state.value, "reason": reason})

    def update(self, obs: Observation) -> BrainCommand:
        """每帧最多发生一次状态转移，优先处理健康与任务边界。"""
        cfg, now = self.config, obs.timestamp_s
        values = [v for v in asdict(obs).values() if not isinstance(v, bool)]
        if not np.all(np.isfinite(values)) or now < 0 or now < self._last_time:
            raise ValueError("观测数值必须有限，时间必须非负且单调")
        if not 0 <= obs.battery_fraction <= 1 or not 0 <= obs.actuator_health <= 1:
            raise ValueError("电量和执行器健康度必须位于 [0,1]")
        if not 0 <= obs.target_posterior <= 1 or min(
                obs.depth_m, obs.sensor_age_s, obs.link_age_s, obs.sensor_support) < 0:
            raise ValueError("概率、深度和观测新鲜度无效")
        if self._last_time == -np.inf:
            self.entered_at = now
        self._last_time = now
        state = self.state
        elapsed = now - self.entered_at
        underwater = obs.z_m < -cfg.surface_band_m
        reason = "保持当前任务阶段"
        target: MissionState | None = None
        terminal = state in (MissionState.COMPLETE, MissionState.ABORT)
        if not terminal and state != MissionState.RECOVERY and (
                obs.actuator_health < cfg.health_min or obs.depth_m > cfg.max_depth_m
                or obs.link_age_s > cfg.link_timeout_s):
            target = MissionState.RECOVERY if underwater else MissionState.ABORT
            reason = "执行器、深度或通信状态触发保护"
        elif state == MissionState.RECOVERY:
            if not underwater:
                target, reason = MissionState.ABORT, "已到水面，任务锁存终止"
            elif elapsed > cfg.recovery_timeout_s:
                target, reason = MissionState.ABORT, "恢复等待超时"
        elif not terminal and state not in (MissionState.RETURN, MissionState.EXIT_WATER,
                                            MissionState.PREFLIGHT) and (
                obs.battery_fraction < cfg.battery_return_fraction):
            target = MissionState.EXIT_WATER if underwater else MissionState.RETURN
            reason = "电量不足，提前结束任务"
        elif state in (MissionState.SURVEY, MissionState.CONFIRM) and (
                obs.sensor_support <= 0 or obs.sensor_age_s > cfg.sensor_timeout_s
                or obs.obstacle_distance_m < cfg.obstacle_hold_m):
            target, reason = MissionState.HOLD, "感知不可用或障碍距离不足"
        elif state == MissionState.PREFLIGHT and obs.flight_ready:
            target, reason = MissionState.AIR_TRANSIT, "起航条件满足"
        elif state == MissionState.AIR_TRANSIT and obs.at_search_area:
            target, reason = MissionState.ENTER_WATER, "到达巡检区域"
        elif state == MissionState.ENTER_WATER:
            if underwater and obs.depth_m >= cfg.survey_depth_m:
                target, reason = MissionState.SURVEY, "达到巡检深度"
            elif elapsed > cfg.enter_timeout_s:
                target, reason = MissionState.ABORT, "入水阶段超时"
        elif state == MissionState.SURVEY:
            if obs.survey_complete or elapsed > cfg.survey_timeout_s:
                target, reason = MissionState.EXIT_WATER, "巡检结束"
            elif obs.target_declared:
                target, reason = MissionState.CONFIRM, "发现候选目标"
        elif state == MissionState.CONFIRM:
            if not obs.target_declared:
                target, reason = MissionState.SURVEY, "候选证据消退"
            else:
                if self._confirm_since is None:
                    self._confirm_since = now
                if now - self._confirm_since >= cfg.confirm_dwell_s:
                    target, reason = MissionState.EXIT_WATER, "目标持续确认，记录并返航"
        elif state == MissionState.HOLD:
            clear = (obs.sensor_support > 0 and obs.sensor_age_s <= cfg.sensor_timeout_s
                     and obs.obstacle_distance_m >= cfg.obstacle_hold_m)
            if clear:
                if self._clear_since is None:
                    self._clear_since = now
                if now - self._clear_since >= cfg.hold_recover_s:
                    target, reason = MissionState.SURVEY, "观测恢复且障碍解除"
            else:
                self._clear_since = None
            if elapsed > cfg.survey_timeout_s:
                target, reason = MissionState.RECOVERY, "保持状态超时"
        elif state == MissionState.EXIT_WATER:
            if obs.z_m > cfg.surface_band_m:
                target, reason = MissionState.RETURN, "离开水面"
            elif elapsed > cfg.exit_timeout_s:
                target, reason = MissionState.RECOVERY, "出水阶段超时"
        elif state == MissionState.RETURN and obs.at_home and (
                obs.z_m >= cfg.complete_altitude_m):
            target, reason = MissionState.COMPLETE, "返回指定回收位置"
        if target is not None:
            self._transition(target, now, reason)
        state = self.state
        mode = "air"
        z, speed = cfg.air_altitude_m, 0.0
        if state in (MissionState.ENTER_WATER, MissionState.SURVEY,
                     MissionState.CONFIRM, MissionState.HOLD):
            mode, z = "water", -cfg.survey_depth_m
            if state == MissionState.SURVEY:
                speed = cfg.survey_speed_m_s
        elif state in (MissionState.EXIT_WATER, MissionState.RECOVERY):
            mode, z = "transition", 0.0
        elif state == MissionState.RETURN:
            speed = cfg.return_speed_m_s
        elif state in (MissionState.ABORT, MissionState.COMPLETE, MissionState.PREFLIGHT):
            mode, z = "standby", obs.z_m
        if state == MissionState.HOLD:
            z = obs.z_m
        return BrainCommand(state.value, mode, z, speed,
                            state == MissionState.CONFIRM, reason)
