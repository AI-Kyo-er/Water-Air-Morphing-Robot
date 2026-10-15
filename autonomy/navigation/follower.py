"""动态窗口局部规划：速度、转向与静态地图碰撞检查。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import OccupancyGrid


def wrap_angle(angle: float) -> float:
    """将角度归一到 [-pi, pi)。"""
    return float((angle + np.pi) % (2*np.pi) - np.pi)


@dataclass(frozen=True)
class FollowerConfig:
    """局部速度候选、滚动预测与代价参数。"""

    max_speed_m_s: float = 0.35
    max_yaw_rate_rad_s: float = 0.9
    acceleration_m_s2: float = 0.3
    yaw_acceleration_rad_s2: float = 1.8
    horizon_s: float = 1.5
    rollout_dt_s: float = 0.1
    speed_samples: int = 7
    yaw_samples: int = 15
    goal_weight: float = 3
    heading_weight: float = 0.4
    clearance_weight: float = 0.3
    speed_weight: float = 0.2


@dataclass
class VelocityCommand:
    """前向速度与偏航角速度目标，包含阻塞原因和预测轨迹。"""

    speed_m_s: float
    yaw_rate_rad_s: float
    blocked: bool
    trajectory: np.ndarray
    reason: str


class DynamicWindowFollower:
    """在加速度可达窗口内选速度；依赖外部位姿与测量速度。"""

    def __init__(self, config: FollowerConfig | None = None):
        self.config = config or FollowerConfig()
        if min(vars(self.config).values()) <= 0:
            raise ValueError("局部规划参数必须为正")

    def command(self, grid: OccupancyGrid, pose: np.ndarray, target_xy: np.ndarray,
                measured_speed: float, measured_yaw_rate: float,
                dt_s: float, speed_limit: float | None = None) -> VelocityCommand:
        """采样常速度弧线与减速尾段，检验膨胀地图内的可行性。"""
        pose, target_xy = np.asarray(pose, float), np.asarray(target_xy, float)
        if pose.shape != (3,) or target_xy.shape != (2,) or not np.all(np.isfinite(
                [*pose, *target_xy, measured_speed, measured_yaw_rate, dt_s])) or dt_s <= 0:
            raise ValueError("位姿、目标或周期无效")
        cfg = self.config
        maximum = cfg.max_speed_m_s if speed_limit is None else min(cfg.max_speed_m_s, speed_limit)
        if not np.isfinite(maximum) or maximum < 0:
            raise ValueError("速度上限无效")
        if maximum == 0:
            return VelocityCommand(0, 0, False, pose[None].copy(), "任务要求保持")
        low = max(0, measured_speed-cfg.acceleration_m_s2*dt_s)
        high = min(maximum, max(0, measured_speed)+cfg.acceleration_m_s2*dt_s)
        if low > high:
            low = high
        yaw_low = max(-cfg.max_yaw_rate_rad_s, measured_yaw_rate-cfg.yaw_acceleration_rad_s2*dt_s)
        yaw_high = min(cfg.max_yaw_rate_rad_s, measured_yaw_rate+cfg.yaw_acceleration_rad_s2*dt_s)
        if yaw_low > yaw_high:
            yaw_low = yaw_high = float(np.clip(measured_yaw_rate, -cfg.max_yaw_rate_rad_s,
                                               cfg.max_yaw_rate_rad_s))
        velocities = np.unique(np.r_[np.linspace(low, high, cfg.speed_samples)])
        yaw_rates = np.unique(np.r_[np.linspace(yaw_low, yaw_high, cfg.yaw_samples),
                                    np.clip(0, yaw_low, yaw_high)])
        blocked, clearance = grid.blocked(), grid.clearance()
        resolution = grid.config.resolution_m
        origin = np.array(grid.config.origin_xy_m)
        step_s = min(cfg.rollout_dt_s, resolution / (4*cfg.max_speed_m_s))

        def check(point: np.ndarray) -> tuple[bool, float]:
            col, row = np.floor((point[:2]-origin)/resolution).astype(int)
            if not 0 <= row < blocked.shape[0] or not 0 <= col < blocked.shape[1]:
                return False, 0
            return not blocked[row, col], float(clearance[row, col])

        best, best_cost = None, np.inf
        for speed in velocities:
            for yaw_rate in yaw_rates:
                state, trace, minimum_clearance = pose.copy(), [pose.copy()], np.inf
                valid, current_clearance = check(state)
                minimum_clearance = min(minimum_clearance, current_clearance)
                elapsed = 0.0
                while valid and elapsed < cfg.horizon_s + speed/cfg.acceleration_m_s2:
                    step = min(step_s, cfg.horizon_s+speed/cfg.acceleration_m_s2-elapsed)
                    velocity = speed if elapsed < cfg.horizon_s else max(
                        0, speed-cfg.acceleration_m_s2*(elapsed-cfg.horizon_s))
                    midpoint_yaw = state[2]+0.5*yaw_rate*step
                    state[:2] += velocity*step*np.array([np.cos(midpoint_yaw), np.sin(midpoint_yaw)])
                    state[2] = wrap_angle(state[2]+yaw_rate*step)
                    valid, distance = check(state)
                    minimum_clearance = min(minimum_clearance, distance)
                    trace.append(state.copy())
                    elapsed += step
                if not valid:
                    continue
                endpoint = trace[min(len(trace)-1, int(np.ceil(cfg.horizon_s/step_s)))]
                direction = np.arctan2(target_xy[1]-endpoint[1], target_xy[0]-endpoint[0])
                cost = (cfg.goal_weight*np.linalg.norm(endpoint[:2]-target_xy)
                        + cfg.heading_weight*abs(wrap_angle(direction-endpoint[2]))
                        + cfg.clearance_weight*np.exp(-minimum_clearance / resolution)
                        + cfg.speed_weight*(maximum-speed))
                if cost < best_cost:
                    best_cost = cost
                    best = VelocityCommand(float(speed), float(yaw_rate), False,
                                           np.asarray(trace), "局部预测轨迹可行")
        if best is None:
            return VelocityCommand(0, 0, True, pose[None].copy(), "无可行局部轨迹，请求减速保持")
        return best
