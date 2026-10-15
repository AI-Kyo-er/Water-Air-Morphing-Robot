"""米制二维占据栅格与测距射线更新，不包含定位或 SLAM。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class GridConfig:
    """栅格、观测概率和机体膨胀参数。"""

    width_cells: int = 80
    height_cells: int = 60
    resolution_m: float = 0.15
    origin_xy_m: tuple[float, float] = (0, 0)
    unknown_is_obstacle: bool = True
    occupied_threshold: float = 0.5
    free_log_odds: float = -0.5
    hit_log_odds: float = 0.9
    log_odds_limit: float = 4
    robot_radius_m: float = 0.32


class OccupancyGrid:
    """行对应世界 y，列对应世界 x，输出已膨胀的可通行空间。"""

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        c = self.config
        if min(c.width_cells, c.height_cells) < 2 or c.resolution_m <= 0 or c.robot_radius_m < 0:
            raise ValueError("地图尺寸、分辨率或机体半径无效")
        if not np.all(np.isfinite([*c.origin_xy_m, c.resolution_m, c.robot_radius_m,
                                   c.occupied_threshold, c.free_log_odds, c.hit_log_odds,
                                   c.log_odds_limit])) or not (
            c.free_log_odds < 0 < c.occupied_threshold < c.hit_log_odds <= c.log_odds_limit):
            raise ValueError("地图概率参数无效")
        self.log_odds = np.zeros((c.height_cells, c.width_cells))
        self.observed = np.zeros_like(self.log_odds, dtype=bool)
        self.version = 0
        self._cached_version = -1
        self._clearance: np.ndarray | None = None
        self._last_scan_s = -np.inf

    def world_to_cell(self, xy: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        """世界坐标 m 转为 (row, column)。"""
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError("位置应为有限二维坐标")
        column, row = np.floor((xy - self.config.origin_xy_m) / self.config.resolution_m).astype(int)
        return int(row), int(column)

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        """返回栅格中心世界坐标。"""
        row, column = cell
        return np.array(self.config.origin_xy_m) + np.array([column+0.5, row+0.5]) * self.config.resolution_m

    def contains(self, cell: tuple[int, int]) -> bool:
        """判断栅格是否在地图内。"""
        r, c = cell
        return 0 <= r < self.config.height_cells and 0 <= c < self.config.width_cells

    def occupied(self) -> np.ndarray:
        """返回原始占据掩码，未知处理由配置决定。"""
        result = self.log_odds >= self.config.occupied_threshold
        if self.config.unknown_is_obstacle:
            result |= ~self.observed
        return result

    def clearance(self) -> np.ndarray:
        """距障碍和地图边缘的保守距离，单位 m。"""
        if self._cached_version != self.version:
            free = np.pad(~self.occupied(), 1, constant_values=False)
            distance = distance_transform_edt(free)[1:-1, 1:-1] * self.config.resolution_m
            yy, xx = np.indices(distance.shape)
            border = np.minimum.reduce([xx+0.5, yy+0.5,
                                        self.config.width_cells-xx-0.5,
                                        self.config.height_cells-yy-0.5]) * self.config.resolution_m
            self._clearance = np.minimum(distance, border)
            self._cached_version = self.version
        return self._clearance.copy()

    def blocked(self) -> np.ndarray:
        """加入机体半径及半格对角离散余量的碰撞掩码。"""
        margin = self.config.robot_radius_m + self.config.resolution_m / np.sqrt(2)
        return self.clearance() <= margin

    def is_free(self, xy: np.ndarray | tuple[float, float]) -> bool:
        """判断中心位置是否属于保守可行空间。"""
        cell = self.world_to_cell(xy)
        return self.contains(cell) and not self.blocked()[cell]

    def set_known_free(self) -> None:
        """载入已知空闲地图；真实未知区域不能调用此函数冒充观测。"""
        self.log_odds.fill(-self.config.log_odds_limit)
        self.observed.fill(True)
        self.version += 1

    def set_rectangle(self, rectangle: tuple[float, ...] | list[float], occupied: bool = True) -> None:
        """载入已知矩形区域，坐标为 xmin、ymin、xmax、ymax。"""
        x0, y0, x1, y1 = rectangle
        if not np.all(np.isfinite(rectangle)) or x0 >= x1 or y0 >= y1:
            raise ValueError("矩形边界无效")
        yy, xx = np.indices(self.log_odds.shape)
        wx = self.config.origin_xy_m[0] + (xx+0.5)*self.config.resolution_m
        wy = self.config.origin_xy_m[1] + (yy+0.5)*self.config.resolution_m
        mask = (wx >= x0) & (wx < x1) & (wy >= y0) & (wy < y1)
        self.log_odds[mask] = self.config.log_odds_limit * (1 if occupied else -1)
        self.observed[mask] = True
        self.version += 1

    def segment_is_free(self, start: np.ndarray, end: np.ndarray) -> bool:
        """密采样线段并检查对角跨格，不允许穿过占据角点。"""
        distance = np.linalg.norm(np.asarray(end) - start)
        count = max(2, int(np.ceil(distance / (self.config.resolution_m / 4))) + 1)
        points = np.linspace(start, end, count)
        blocked = self.blocked()
        previous = None
        for point in points:
            cell = self.world_to_cell(point)
            if not self.contains(cell) or blocked[cell]:
                return False
            if previous is not None and cell[0] != previous[0] and cell[1] != previous[1]:
                if blocked[cell[0], previous[1]] or blocked[previous[0], cell[1]]:
                    return False
            previous = cell
        return True

    def update_scan(self, pose_xy_yaw: np.ndarray, angles_rad: np.ndarray,
                    ranges_m: np.ndarray, max_range_m: float, timestamp_s: float) -> None:
        """用已标定声呐/测距射线更新；正无穷表示量程内无回波。"""
        pose = np.asarray(pose_xy_yaw, float)
        angles, ranges = np.asarray(angles_rad, float), np.asarray(ranges_m, float)
        if pose.shape != (3,) or angles.shape != ranges.shape or angles.ndim != 1:
            raise ValueError("位姿/射线形状无效")
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(angles)) or (
                not np.isfinite(max_range_m) or max_range_m <= 0 or
                not np.isfinite(timestamp_s) or timestamp_s <= self._last_scan_s):
            raise ValueError("扫描时间、位姿或量程无效")
        if not self.contains(self.world_to_cell(pose[:2])):
            raise ValueError("扫描原点必须在地图内")
        self._last_scan_s = timestamp_s
        cfg = self.config
        before = self.occupied()
        for angle, distance in zip(angles, ranges):
            if np.isnan(distance) or distance <= 0:
                continue
            hit = np.isfinite(distance) and distance < max_range_m
            length = min(distance, max_range_m)
            end = pose[:2] + length * np.array([np.cos(pose[2]+angle), np.sin(pose[2]+angle)])
            n = max(2, int(np.ceil(length / (cfg.resolution_m/2)))+1)
            cells = list(dict.fromkeys(self.world_to_cell(p) for p in np.linspace(pose[:2], end, n)))
            endpoint = self.world_to_cell(end)
            for cell in cells:
                if not self.contains(cell):
                    continue
                self.observed[cell] = True
                increment = cfg.hit_log_odds if hit and cell == endpoint else cfg.free_log_odds
                self.log_odds[cell] = np.clip(self.log_odds[cell]+increment,
                                              -cfg.log_odds_limit, cfg.log_odds_limit)
        if not np.array_equal(before, self.occupied()):
            self.version += 1
