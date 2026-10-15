"""共倾转与独立倾转的控制分配参考实现。

本机型共有 8 个执行量（4 个推力 + 4 个倾转角）。当前平行倾转几何的
直接横向力为零，六维广义力并非任意可达。控制分配解决的是：

    给定期望合力 / 合力矩 w_ref ∈ R^6，
    求执行量 u = [T; theta]，使映射结果尽可能等于 w_ref，
    同时满足推力上下限、倾转角限幅，并尽量降低执行代价。

采用两级结构化分配（与"工"字机架的真实作动方式一致）：

1. **共倾转角**：四套推进单元绕同一横向轴同步倾转，形成统一的前推/后推
   矢量；倾转角由期望水平/垂直力的方向确定，力矩耦合由几何映射计算。
2. **差动推力**：在共倾转的公共推力基底上，用四桨差动推力产生滚转 / 俯仰 /
   偏航力矩（标准四旋翼混控）。

实现上以当前执行量 u0 为工作点一阶展开，构成**加权最小二乘 + 显式上下界**
的增量问题（含倾转基线正则），并用 scipy 求解；同时提供面向实时嵌入式
的加权伪逆快速实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .geometry import body_wrench, effectiveness_matrix
from .params import Medium, VehicleParams


@dataclass
class AllocationWeights:
    """控制分配权重配置。"""

    force: float = 1.0
    torque: float = 1.0
    thrust_effort: float = 1e-2
    tilt_effort: float = 5e-2
    tikhonov: float = 1e-3
    tilt_regularization: float = 0.05  # 拉向倾转基线的强度


class TiltRotorAllocator:
    """倾转旋翼过驱动控制分配器。"""

    def __init__(self, params: VehicleParams, weights: AllocationWeights | None = None):
        self.p = params
        self.w = weights or AllocationWeights()

    # ------------------------------------------------------------------
    def _tilt_baseline(self, w_ref: np.ndarray) -> float:
        """由期望前向 / 垂直力确定共倾转基线角 theta。"""
        fx, fz = float(w_ref[0]), float(w_ref[2])
        theta = np.arctan2(fx, max(fz, 1e-3))
        return float(np.clip(theta, -self.p.rotor.tilt_limit, self.p.rotor.tilt_limit))

    # ------------------------------------------------------------------
    def allocate(
        self,
        w_ref: np.ndarray,
        u_prev: np.ndarray,
        medium: Medium,
        ignore_tilt: bool = False,
    ) -> np.ndarray:
        """结构化控制分配（共倾转角 + 差动推力）。

        Args:
            w_ref: 期望力 / 力矩 (6,)。
            u_prev: 上一步执行量 (8,)。
            medium: 当前介质。
            ignore_tilt: 为 True 时锁定倾转角（空中模式仅分配推力）。

        Returns:
            新的执行量 u (8,)，其中四个倾转角相同。
        """
        u0 = np.asarray(u_prev, dtype=float).reshape(-1).copy()
        T0 = u0[0:4]
        theta0 = float(u0[4:8].mean())

        T_all = np.full(4, theta0)
        B_full = effectiveness_matrix(T0, T_all, self.p.rotor)
        wrench0 = body_wrench(T0, T_all, self.p.rotor)
        residual = np.asarray(w_ref, dtype=float).reshape(-1) - wrench0

        # --- 组装变量：dT(4) [+ dtheta(1)] ---
        cols = [B_full[:, i] for i in range(4)]
        nv = 4
        if not ignore_tilt:
            cols.append(B_full[:, 4:8].sum(axis=1))  # 共倾转角对合力矩的贡献
            nv = 5
        B = np.column_stack(cols)

        base = self._tilt_baseline(w_ref)
        target = np.zeros(nv)
        if not ignore_tilt:
            target[4] = base - theta0

        W = np.diag([self.w.force] * 3 + [self.w.torque] * 3)
        eff = np.array([self.w.thrust_effort] * 4 + ([self.w.tilt_effort] if not ignore_tilt else []))
        reg = self.w.tikhonov
        reg_base = self.w.tilt_regularization
        A = np.vstack(
            [W @ B, np.sqrt(reg) * np.diag(eff), np.sqrt(reg_base) * np.eye(nv)]
        )
        b = np.concatenate([W @ residual, np.zeros(nv), np.sqrt(reg_base) * target])

        t_max = self.p.rotor.thrust_max(medium)
        tl = self.p.rotor.tilt_limit
        lb = list(-T0)
        ub = list(t_max - T0)
        if not ignore_tilt:
            lb.append(-tl - theta0)
            ub.append(tl - theta0)
        lb = np.asarray(lb)
        ub = np.maximum(np.asarray(ub), lb + 1e-9)

        sol = lsq_linear(A, b, bounds=(lb, ub), method="trf")
        du = sol.x

        T = np.clip(T0 + du[0:4], 0.0, t_max)
        theta = theta0 if ignore_tilt else float(np.clip(theta0 + du[4], -tl, tl))
        return np.concatenate([T, np.full(4, theta)])

    # ------------------------------------------------------------------
    def allocate_full(
        self, w_ref: np.ndarray, u_prev: np.ndarray, medium: Medium
    ) -> np.ndarray:
        """独立倾转角分配（8 自由度 QP），用于需要差动倾转的高级机动。

        一般机动建议使用 `allocate`；本方法在极端差动倾转场景下有效，但对
        执行量权重敏感，需谨慎使用。
        """
        u0 = np.asarray(u_prev, dtype=float).reshape(-1).copy()
        B = effectiveness_matrix(u0[0:4], u0[4:8], self.p.rotor)
        residual = np.asarray(w_ref, dtype=float).reshape(-1) - body_wrench(
            u0[0:4], u0[4:8], self.p.rotor
        )
        W = np.diag([self.w.force] * 3 + [self.w.torque] * 3)
        reg = self.w.tikhonov
        eff = np.array([self.w.thrust_effort] * 4 + [self.w.tilt_effort] * 4)
        A = np.vstack([W @ B, np.sqrt(reg) * np.diag(eff)])
        b = np.concatenate([W @ residual, np.zeros(8)])
        t_max = self.p.rotor.thrust_max(medium)
        tl = self.p.rotor.tilt_limit
        lb = np.concatenate([-u0[0:4], -tl - u0[4:8]])
        ub = np.concatenate([t_max - u0[0:4], tl - u0[4:8]])
        ub = np.maximum(ub, lb + 1e-9)
        sol = lsq_linear(A, b, bounds=(lb, ub), method="trf")
        u = u0 + sol.x
        u[0:4] = np.clip(u[0:4], 0.0, t_max)
        u[4:8] = np.clip(u[4:8], -tl, tl)
        return u

    # ------------------------------------------------------------------
    def allocate_pinv(
        self, w_ref: np.ndarray, u_prev: np.ndarray, medium: Medium
    ) -> np.ndarray:
        """加权伪逆快速分配（无约束，可实时），供嵌入式固件参考实现。

        u = u0 + Wt B^T (B Wt B^T + eps I)^{-1} (w_ref - wrench(u0))，再做硬限幅。
        """
        u0 = np.asarray(u_prev, dtype=float).reshape(-1).copy()
        B = effectiveness_matrix(u0[0:4], u0[4:8], self.p.rotor)
        residual = np.asarray(w_ref, dtype=float).reshape(-1) - body_wrench(
            u0[0:4], u0[4:8], self.p.rotor
        )
        Wt = np.diag(1.0 / np.array([self.w.thrust_effort] * 4 + [self.w.tilt_effort] * 4))
        eps = 1e-6
        M = B @ Wt @ B.T + eps * np.eye(6)
        du = Wt @ B.T @ np.linalg.solve(M, residual)
        u = u0 + du
        t_max = self.p.rotor.thrust_max(medium)
        u[0:4] = np.clip(u[0:4], 0.0, t_max)
        u[4:8] = np.clip(u[4:8], -self.p.rotor.tilt_limit, self.p.rotor.tilt_limit)
        return u
