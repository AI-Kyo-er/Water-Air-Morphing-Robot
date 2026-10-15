"""全局规划、路径局部目标和地图变化触发重规划。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .grid import GridConfig, OccupancyGrid
from .planner import AStarPlanner, PlannerConfig, PathResult
from .follower import DynamicWindowFollower, FollowerConfig, VelocityCommand


@dataclass(frozen=True)
class NavigatorConfig:
    """局部目标和任务到达参数。"""

    lookahead_m: float = 0.9
    goal_tolerance_m: float = 0.25
    replan_interval_s: float = 2


@dataclass
class NavigationResult:
    """路径、局部控制目标与可解释导航状态。"""

    status: str
    command: VelocityCommand
    path: np.ndarray
    local_target_xy_m: np.ndarray
    replanned: bool
    reason: str


class Navigator:
    """固定深度/高度层中的二维导航；位置与航向由外部估计器提供。"""

    def __init__(self, grid: OccupancyGrid, planner: AStarPlanner,
                 follower: DynamicWindowFollower, config: NavigatorConfig | None = None):
        self.grid, self.planner, self.follower = grid, planner, follower
        self.config = config or NavigatorConfig()
        if min(vars(self.config).values()) <= 0:
            raise ValueError("导航参数必须为正")
        self.goal: np.ndarray | None = None
        self.plan: PathResult | None = None
        self.replan_count = 0
        self._last_plan = -np.inf
        self._map_version = -1
        self._last_time = -np.inf

    @classmethod
    def from_yaml(cls, path: str | Path) -> Navigator:
        """从集中配置构造地图、规划器和跟随器。"""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw["grid"]["origin_xy_m"] = tuple(raw["grid"]["origin_xy_m"])
        return cls(OccupancyGrid(GridConfig(**raw["grid"])),
                   AStarPlanner(PlannerConfig(**raw["planner"])),
                   DynamicWindowFollower(FollowerConfig(**raw["follower"])),
                   NavigatorConfig(**raw["navigator"]))

    def set_goal(self, goal_xy_m: np.ndarray) -> None:
        """设置世界坐标导航目标，目标变化后重新规划。"""
        goal = np.asarray(goal_xy_m, float)
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("导航目标应为有限二维坐标")
        if self.goal is None or not np.allclose(goal, self.goal):
            self.goal = goal.copy()
            self.plan = None

    def _local_target(self, position: np.ndarray) -> np.ndarray:
        points = self.plan.points_xy_m
        if len(points) == 1:
            return points[0]
        distances, projections = [], []
        for a, b in zip(points[:-1], points[1:]):
            delta = b-a
            fraction = np.clip(np.dot(position-a, delta)/max(delta@delta, 1e-12), 0, 1)
            projection = a+fraction*delta
            distances.append(np.linalg.norm(position-projection))
            projections.append(projection)
        index = int(np.argmin(distances))
        cursor, remaining = projections[index], self.config.lookahead_m
        target = points[-1]
        for endpoint in points[index+1:]:
            length = np.linalg.norm(endpoint-cursor)
            if length >= remaining:
                target = cursor+(endpoint-cursor)*remaining/max(length, 1e-12)
                break
            remaining -= length
            cursor = endpoint
        if self.grid.segment_is_free(position, target):
            return target
        # 前视点不能跨过不可见拐角，否则局部代价可能把机器人锁在墙边。
        for endpoint in points[index+1:]:
            if self.grid.segment_is_free(position, endpoint):
                return endpoint.copy()
        return projections[index]

    def step(self, pose_xy_yaw: np.ndarray, speed_m_s: float, yaw_rate_rad_s: float,
             timestamp_s: float, dt_s: float, enabled: bool = True,
             speed_limit: float | None = None) -> NavigationResult:
        """根据当前地图与位姿给出一周期速度目标，disabled 时保留任务路径。"""
        pose = np.asarray(pose_xy_yaw, float)
        if pose.shape != (3,) or not np.all(np.isfinite([*pose, timestamp_s, dt_s])) or (
                timestamp_s < self._last_time or dt_s <= 0):
            raise ValueError("导航位姿、时间或周期无效")
        self._last_time = timestamp_s
        stop = VelocityCommand(0, 0, False, pose[None].copy(), "保持")
        path = self.plan.points_xy_m if self.plan is not None else np.empty((0, 2))
        if not enabled or self.goal is None:
            return NavigationResult("paused", stop, path, pose[:2].copy(), False, "任务暂停或尚未设置目标")
        if np.linalg.norm(pose[:2]-self.goal) <= self.config.goal_tolerance_m:
            return NavigationResult("arrived", stop, path, self.goal.copy(), False, "到达导航目标")
        replanned = False
        if self.plan is None or self.grid.version != self._map_version or (
                timestamp_s-self._last_plan >= self.config.replan_interval_s):
            self.plan = self.planner.plan(self.grid, pose[:2], self.goal)
            self._last_plan, self._map_version = timestamp_s, self.grid.version
            self.replan_count += 1
            replanned = True
        if not self.plan.success:
            stop.blocked = True
            return NavigationResult("unreachable", stop, self.plan.points_xy_m,
                                    self.goal.copy(), replanned, self.plan.reason)
        target = self._local_target(pose[:2])
        command = self.follower.command(self.grid, pose, target, speed_m_s, yaw_rate_rad_s,
                                        dt_s, speed_limit)
        return NavigationResult("blocked" if command.blocked else "following", command,
                                self.plan.points_xy_m.copy(), target, replanned, command.reason)
