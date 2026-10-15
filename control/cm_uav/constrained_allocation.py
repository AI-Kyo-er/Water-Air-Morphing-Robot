"""面向介质与执行器约束的逐次线性控制分配。单位采用 SI。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import lsq_linear

from .geometry import body_wrench, effectiveness_matrix
from .params import Medium, VehicleParams


@dataclass(frozen=True)
class AllocationConfig:
    """任务轴权重、单位归一化尺度与执行器变化率约束。"""

    force_scale_n: float = 12.0
    torque_scale_nm: float = 1.0
    axis_priority: tuple[float, ...] = (1, 0, 3, 3, 3, 1.5)
    thrust_rate_n_s: float = 80.0
    smoothness: float = 0.002
    max_iterations: int = 5
    independent_tilt: bool = True
    enforce_rates: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> AllocationConfig:
        """读取集中配置中的 allocation 节点。"""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["allocation"]
        raw["axis_priority"] = tuple(raw["axis_priority"])
        return cls(**raw)


@dataclass
class AllocationResult:
    """执行指令、实际模型广义力、残差与约束诊断。"""

    command: np.ndarray
    achieved_wrench: np.ndarray
    residual: np.ndarray
    normalized_error: float
    local_rank: int
    active_bounds: int
    iterations: int
    forced_projection: bool


class ConstrainedAllocator:
    """独立倾转、变化率约束和已知效能损失下的控制分配器。

    health 是已知/外部估计的推力效能比例，不是本类估计的故障概率。
    当前平行倾转构型的 Fy 恒为零，局部秩至多为 5。
    """

    def __init__(self, params: VehicleParams, config: AllocationConfig | None = None):
        self.params = params
        self.config = config or AllocationConfig()
        cfg = self.config
        scales = np.array([cfg.force_scale_n] * 3 + [cfg.torque_scale_nm] * 3)
        if np.any(scales <= 0) or cfg.thrust_rate_n_s <= 0:
            raise ValueError("尺度和推力变化率必须为正")
        if len(cfg.axis_priority) != 6 or np.any(np.array(cfg.axis_priority) < 0):
            raise ValueError("axis_priority 必须包含六个非负权重")
        if cfg.smoothness < 0 or cfg.max_iterations < 1:
            raise ValueError("正则权重非负，迭代次数至少为一")
        self.weights = np.array(cfg.axis_priority) / scales

    def wrench(self, command: np.ndarray, health: np.ndarray) -> np.ndarray:
        """由命令推力和实际效能计算机体系力/力矩 [N, N·m]。"""
        return body_wrench(command[:4] * health, command[4:], self.params.rotor)

    def bounds(
        self, previous: np.ndarray, medium: Medium, dt: float, air_fraction: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求本周期可达区间；物理边界变化时先投影旧指令。"""
        rotor = self.params.rotor
        physical_low = np.r_[np.zeros(4), np.full(4, -rotor.tilt_limit)]
        physical_high = np.r_[np.full(4, rotor.thrust_max(medium)),
                              np.full(4, rotor.tilt_limit)]
        previous = np.clip(previous, physical_low, physical_high)
        if self.config.enforce_rates:
            step = np.r_[np.full(4, self.config.thrust_rate_n_s * dt),
                         np.full(4, rotor.tilt_rate_limit * dt)]
            low = np.maximum(physical_low, previous - step)
            high = np.minimum(physical_high, previous + step)
        else:
            low, high = physical_low.copy(), physical_high.copy()
        # 空中倾转回中位仍遵守变化率，不能瞬间锁死非零舵角。
        if air_fraction >= 0.999:
            target = np.clip(np.zeros(4), low[4:], high[4:])
            low[4:] = target
            high[4:] = target
        return low, high, previous

    @staticmethod
    def _bounded_step(a: np.ndarray, b: np.ndarray, low: np.ndarray,
                      high: np.ndarray) -> np.ndarray:
        """处理锁定变量，避免上下界相等时求解器拒绝问题。"""
        free = high - low > 1e-10
        solution = low.copy()
        if np.any(free):
            rhs = b - a[:, ~free] @ solution[~free]
            fit = lsq_linear(a[:, free], rhs, bounds=(low[free], high[free]),
                             method="bvls", tol=1e-8, max_iter=100)
            if not fit.success:
                raise RuntimeError(f"约束分配求解失败：{fit.message}")
            solution[free] = fit.x
        return solution

    def allocate(self, target: np.ndarray, previous: np.ndarray, medium: Medium,
                 dt: float, health: np.ndarray | None = None,
                 air_fraction: float = 0.0) -> AllocationResult:
        """返回本周期指令；target/previous 形状为 (6,)/(8,)，dt 单位 s。"""
        target = np.asarray(target, dtype=float)
        previous = np.asarray(previous, dtype=float)
        health = np.ones(4) if health is None else np.asarray(health, dtype=float)
        if target.shape != (6,) or previous.shape != (8,) or health.shape != (4,):
            raise ValueError("target、previous、health 形状应为 (6)、(8)、(4)")
        if not np.all(np.isfinite(np.r_[target, previous, health, dt, air_fraction])):
            raise ValueError("输入必须为有限数值")
        if dt <= 0 or np.any((health < 0) | (health > 1)) or not 0 <= air_fraction <= 1:
            raise ValueError("dt 必须为正，health 与 air_fraction 必须位于 [0,1]")
        low, high, anchor = self.bounds(previous, medium, dt, air_fraction)
        u = np.clip(anchor, low, high)
        transform = np.eye(8)
        if not self.config.independent_tilt:
            if np.ptp(anchor[4:]) > 1e-8:
                raise ValueError("共倾转消融要求前一周期倾转角相同")
            transform = np.zeros((8, 5))
            transform[:4, :4] = np.eye(4)
            transform[4:, 4] = 1.0
        scale = np.r_[np.full(4, self.params.rotor.thrust_max(medium)),
                      np.full(4, self.params.rotor.tilt_limit)]
        penalty = np.sqrt(self.config.smoothness) / scale

        def cost(candidate: np.ndarray) -> float:
            error = self.weights * (self.wrench(candidate, health) - target)
            change = penalty * (candidate - anchor)
            return float(error @ error + change @ change)

        iterations = 0
        for iterations in range(1, self.config.max_iterations + 1):
            jac = effectiveness_matrix(u[:4] * health, u[4:], self.params.rotor)
            jac[:, :4] *= health[None, :]
            a = np.vstack([self.weights[:, None] * (jac @ transform),
                           penalty[:, None] * transform])
            b = np.r_[self.weights * (target - self.wrench(u, health)),
                      -penalty * (u - anchor)]
            dl, dh = low - u, high - u
            if not self.config.independent_tilt:
                dl = np.r_[dl[:4], np.max(dl[4:])]
                dh = np.r_[dh[:4], np.min(dh[4:])]
            step = transform @ self._bounded_step(a, b, dl, dh)
            old_cost = cost(u)
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = np.clip(u + fraction * step, low, high)
                if cost(candidate) < old_cost - 1e-12:
                    u = candidate
                    accepted = True
                    break
            if not accepted or np.linalg.norm(step) < 1e-7:
                break
        achieved = self.wrench(u, health)
        jac = effectiveness_matrix(u[:4] * health, u[4:], self.params.rotor)
        jac[:, :4] *= health[None, :]
        return AllocationResult(
            command=u, achieved_wrench=achieved, residual=target - achieved,
            normalized_error=float(np.linalg.norm(self.weights * (target - achieved))),
            local_rank=int(np.linalg.matrix_rank(jac @ transform)),
            active_bounds=int(np.sum(np.isclose(u, low) | np.isclose(u, high))),
            iterations=iterations,
            forced_projection=bool(np.any(np.abs(anchor - previous) > 1e-9)),
        )
