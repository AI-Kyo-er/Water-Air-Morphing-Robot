"""带障碍余量代价与禁止切角的 A* 全局路径规划。"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np

from .grid import OccupancyGrid


@dataclass(frozen=True)
class PlannerConfig:
    """障碍距离代价与搜索预算。"""

    clearance_weight: float = 1.5
    clearance_decay_m: float = 0.7
    max_expansions: int = 20000
    shortcut_cost_tolerance: float = 1.05


@dataclass
class PathResult:
    """路径结果；不可达时明确返回原因，不生成穿障路径。"""

    success: bool
    points_xy_m: np.ndarray
    cost: float
    expanded: int
    reason: str


class AStarPlanner:
    """8 邻域 A*，欧氏启发式与非负距离代价。"""

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        c = self.config
        if c.clearance_weight < 0 or c.clearance_decay_m <= 0 or c.max_expansions < 1 or c.shortcut_cost_tolerance < 1:
            raise ValueError("规划参数无效")

    def plan(self, grid: OccupancyGrid, start_xy: np.ndarray,
             goal_xy: np.ndarray) -> PathResult:
        """接收世界米制起终点，返回连续可检查的路径点。"""
        start, goal = grid.world_to_cell(start_xy), grid.world_to_cell(goal_xy)
        blocked = grid.blocked()
        empty = np.empty((0, 2))
        if not grid.contains(start) or not grid.contains(goal):
            return PathResult(False, empty, np.inf, 0, "起终点超出地图")
        if blocked[start] or blocked[goal]:
            return PathResult(False, empty, np.inf, 0, "起终点进入障碍或机体膨胀区")
        resolution = grid.config.resolution_m
        clearance = grid.clearance()
        risk = self.config.clearance_weight * np.exp(-clearance / self.config.clearance_decay_m)
        counter = itertools.count()
        queue = [(np.linalg.norm(np.array(start)-goal)*resolution, next(counter), 0.0, start)]
        scores, parent = {start: 0.0}, {}
        expanded = 0
        while queue and expanded < self.config.max_expansions:
            _, _, cost, current = heapq.heappop(queue)
            if cost > scores[current] + 1e-12:
                continue
            expanded += 1
            if current == goal:
                cells = [current]
                while current in parent:
                    current = parent[current]
                    cells.append(current)
                points = np.array([grid.cell_to_world(cell) for cell in reversed(cells)])
                # 保留真实起终点并逐段验证，避免栅格中心偏移改变任务目标。
                points = np.vstack([start_xy, points, goal_xy])
                return PathResult(True, self._simplify(grid, points), cost, expanded, "找到可行路径")
            r, c = current
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                neighbor = (r+dr, c+dc)
                if not grid.contains(neighbor) or blocked[neighbor]:
                    continue
                if dr and dc and (blocked[r+dr, c] or blocked[r, c+dc]):
                    continue
                step = np.hypot(dr, dc) * resolution * (1+0.5*(risk[current]+risk[neighbor]))
                candidate = cost + step
                if candidate + 1e-12 < scores.get(neighbor, np.inf):
                    scores[neighbor], parent[neighbor] = candidate, current
                    heuristic = np.linalg.norm(np.array(neighbor)-goal)*resolution
                    heapq.heappush(queue, (candidate+heuristic, next(counter), candidate, neighbor))
        reason = "搜索预算耗尽" if queue else "目标不可达"
        return PathResult(False, empty, np.inf, expanded, reason)

    def _simplify(self, grid: OccupancyGrid, points: np.ndarray) -> np.ndarray:
        """保守碰撞和风险代价同时通过时简化，避免抹去余量优化。"""
        clearance = grid.clearance()
        risk = self.config.clearance_weight * np.exp(-clearance/self.config.clearance_decay_m)

        def segment_cost(a, b):
            length = np.linalg.norm(b-a)
            count = max(2, int(np.ceil(length/grid.config.resolution_m))+1)
            samples = np.linspace(a, b, count)
            values = np.array([1+risk[grid.world_to_cell(p)] for p in samples])
            return length*float(np.trapezoid(values, dx=1/(count-1)))

        cumulative = np.r_[0.0, np.cumsum([segment_cost(a, b) for a, b in zip(points[:-1], points[1:])])]
        result, i = [points[0]], 0
        while i < len(points)-1:
            j = len(points)-1
            while j > i+1 and (not grid.segment_is_free(points[i], points[j]) or
                segment_cost(points[i], points[j]) > (cumulative[j]-cumulative[i])*self.config.shortcut_cost_tolerance+1e-9):
                j -= 1
            if not grid.segment_is_free(points[i], points[j]):
                raise RuntimeError("搜索路径未通过连续空间检查")
            if np.linalg.norm(points[j] - result[-1]) > 1e-9:
                result.append(points[j])
            i = j
        return np.asarray(result)
