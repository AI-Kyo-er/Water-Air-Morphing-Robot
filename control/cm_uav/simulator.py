"""闭环仿真环境。

把动力学、控制分配、级联控制器串成闭环，按固定步长推进并记录完整状态、
执行量与能耗估算，供数值实验与图表生成使用。所有物理量统一使用 SI 单位。

支持两级时间尺度：动力学积分步长 dt 与飞控控制步长 control_dt，后者更接近
真实机载飞控（约 100 Hz）的控制频率。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .controller import CascadedController, ModeManager, Setpoint
from .dynamics import CrossMediumDynamics, VehicleState, euler_from_R
from .params import VehicleParams


@dataclass
class SimLog:
    """仿真记录。"""

    t: list[float] = field(default_factory=list)
    position: list[np.ndarray] = field(default_factory=list)
    velocity: list[np.ndarray] = field(default_factory=list)
    euler: list[np.ndarray] = field(default_factory=list)
    omega: list[np.ndarray] = field(default_factory=list)
    thrust: list[np.ndarray] = field(default_factory=list)
    tilt: list[np.ndarray] = field(default_factory=list)
    power: list[float] = field(default_factory=list)
    medium: list[str] = field(default_factory=list)
    setpoint: list[np.ndarray] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "t": np.asarray(self.t),
            "position": np.asarray(self.position),
            "velocity": np.asarray(self.velocity),
            "euler": np.asarray(self.euler),
            "omega": np.asarray(self.omega),
            "thrust": np.asarray(self.thrust),
            "tilt": np.asarray(self.tilt),
            "power": np.asarray(self.power),
            "setpoint": np.asarray(self.setpoint),
            "medium": np.asarray(self.medium),
        }


class Simulator:
    """跨介质闭环仿真器。"""

    def __init__(self, params: VehicleParams, initial_medium: str = "air"):
        self.p = params
        self.dynamics = CrossMediumDynamics(params)
        self.mode = ModeManager(initial_medium)
        self.controller = CascadedController(params, mode=self.mode)

    # ------------------------------------------------------------------
    def induced_power(self, thrust: np.ndarray, medium) -> float:
        """基于动量理论的诱导功率估算：P = Σ T_i * sqrt(T_i / (2 rho A))。"""
        A = self.p.rotor.disk_area
        rho = medium.density
        t = np.clip(thrust, 0.0, None)
        vi = np.sqrt(t / (2.0 * rho * A))
        return float(np.sum(t * vi))

    # ------------------------------------------------------------------
    def _limit_tilt_rate(
        self, u_new: np.ndarray, u_prev: np.ndarray, dt: float
    ) -> np.ndarray:
        lim = self.p.rotor.tilt_rate_limit * dt
        u_new[4:8] = np.clip(u_new[4:8], u_prev[4:8] - lim, u_prev[4:8] + lim)
        return u_new

    # ------------------------------------------------------------------
    def run(
        self,
        x0: np.ndarray,
        setpoint_fn: Callable[[float], Setpoint],
        medium_fn,
        t_end: float,
        dt: float = 0.002,
        control_dt: float = 0.01,
    ) -> dict[str, np.ndarray]:
        """运行闭环仿真。

        Args:
            x0: 初始状态（18 维）。
            setpoint_fn: 时间 -> 设定点。
            medium_fn: 时间与位置 -> 介质名，或常量字符串。
            t_end: 仿真时长，s。
            dt: 动力学积分步长，s。
            control_dt: 控制更新步长，s。

        Returns:
            记录字典（见 SimLog.arrays）。
        """
        if isinstance(medium_fn, str):
            const = medium_fn
            medium_fn = lambda t, pos: const  # noqa: E731

        ctrl_every = max(1, int(round(control_dt / dt)))
        log = SimLog()
        x = np.asarray(x0, dtype=float).copy()
        medium0 = self.p.medium(medium_fn(0.0, x[0:3]))
        u = np.zeros(8)
        u[0:4] = self.dynamics.hover_thrust_per_rotor(medium0)
        self.mode.command(medium0.name)

        n = int(round(t_end / dt))
        for k in range(n):
            t = k * dt
            st = VehicleState.from_vector(x)
            medium_name = medium_fn(t, st.position)
            medium = self.p.medium(medium_name)
            self.mode.command(medium_name)

            if k % ctrl_every == 0:
                sp = setpoint_fn(t)
                u_prev = u.copy()
                u = self.controller.compute(st, sp, medium, u_prev, control_dt)
                u = self._limit_tilt_rate(u, u_prev, control_dt)

            power = self.induced_power(u[0:4], medium)

            log.t.append(t)
            log.position.append(st.position.copy())
            log.velocity.append(st.velocity.copy())
            log.euler.append(euler_from_R(st.rotation))
            log.omega.append(st.omega.copy())
            log.thrust.append(u[0:4].copy())
            log.tilt.append(u[4:8].copy())
            log.power.append(power)
            log.medium.append(medium_name)
            log.setpoint.append(setpoint_fn(t).position.copy())

            x = self.dynamics.step(x, u, medium, dt)

        return log.arrays()
