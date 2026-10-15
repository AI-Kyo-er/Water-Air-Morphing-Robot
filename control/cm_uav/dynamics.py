"""跨介质统一六自由度动力学模型。

模型以机体系转动矩阵描述姿态，避免欧拉角奇异；介质差异通过统一的
参数集体现（密度、浮力、附加质量、阻尼），从而在空中与水中共用同一套
状态方程，仅切换介质参数——这是跨介质控制的物理基础。

状态向量 state = [p(3), v(3), R(9), omega(3)]，共 18 维：
    p     : 世界系位置（z 向上），m
    v     : 世界系速度，m/s
    R     : 机体系 -> 世界系旋转矩阵，行优先展平
    omega : 机体系角速度，rad/s
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import body_wrench
from .params import Medium, VehicleParams


def hat(w: np.ndarray) -> np.ndarray:
    """向量 -> 反对称矩阵（叉乘算子）。"""
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=float,
    )


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """对旋转矩阵做极分解正交化，抑制数值漂移。"""
    u, _, vt = np.linalg.svd(R)
    Rn = u @ vt
    if np.linalg.det(Rn) < 0:
        u[:, -1] *= -1
        Rn = u @ vt
    return Rn


def euler_from_R(R: np.ndarray) -> np.ndarray:
    """从旋转矩阵提取 ZYX 欧拉角 [roll, pitch, yaw]，rad。"""
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


@dataclass
class VehicleState:
    """飞行器状态容器（便于人读与日志）。"""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def euler(self) -> np.ndarray:
        return euler_from_R(self.rotation)

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.position, self.velocity, self.rotation.reshape(-1), self.omega]
        )

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "VehicleState":
        return cls(
            position=x[0:3].copy(),
            velocity=x[3:6].copy(),
            rotation=x[6:15].reshape(3, 3).copy(),
            omega=x[15:18].copy(),
        )


class CrossMediumDynamics:
    """跨介质六自由度动力学。"""

    def __init__(self, params: VehicleParams):
        self.p = params
        self._inertia = np.diag(params.inertia)

    # ------------------------------------------------------------------
    def derivative(self, x: np.ndarray, u: np.ndarray, medium: Medium) -> np.ndarray:
        """状态导数 dx/dt。u = [T1..T4, theta1..theta4]。"""
        st = VehicleState.from_vector(x)
        R = st.rotation
        omega = st.omega
        v_body = R.T @ st.velocity

        # --- 推进系统力 / 力矩（机体系）---
        wrench = body_wrench(u[0:4], u[4:8], self.p.rotor)
        force_body = wrench[0:3]
        torque_body = wrench[3:6]

        # --- 流体阻尼（机体系，含线性 + 二次项）---
        drag_body = -(
            medium.linear_drag * v_body
            + medium.quadratic_drag * v_body * np.abs(v_body)
        )
        angular_drag = -medium.angular_drag * omega

        # --- 附加质量（广义惯性等效）---
        m_eff = self.p.mass * (1.0 + medium.added_mass_factor)

        # --- 世界系外力：重力 + 浮力 + 阻力 ---
        gravity_w = np.array([0.0, 0.0, -self.p.weight])
        buoyancy_w = np.array([0.0, 0.0, self.p.buoyancy(medium)])
        f_body_total = force_body + drag_body
        accel_world = (R @ f_body_total + gravity_w + buoyancy_w) / m_eff

        # --- 姿态动力学（欧拉方程）---
        I = self._inertia
        gyro = np.cross(omega, I @ omega)
        omega_dot = np.linalg.solve(I, torque_body + angular_drag - gyro)

        # --- 组装导数 ---
        R_dot = R @ hat(omega)
        dx = np.zeros(18)
        dx[0:3] = st.velocity
        dx[3:6] = accel_world
        dx[6:15] = R_dot.reshape(-1)
        dx[15:18] = omega_dot
        return dx

    # ------------------------------------------------------------------
    def step(
        self, x: np.ndarray, u: np.ndarray, medium: Medium, dt: float
    ) -> np.ndarray:
        """四阶龙格-库塔积分一步，并对旋转矩阵正交化。"""
        k1 = self.derivative(x, u, medium)
        k2 = self.derivative(x + 0.5 * dt * k1, u, medium)
        k3 = self.derivative(x + 0.5 * dt * k2, u, medium)
        k4 = self.derivative(x + dt * k3, u, medium)
        x_new = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        R = orthonormalize(x_new[6:15].reshape(3, 3))
        x_new[6:15] = R.reshape(-1)
        return x_new

    # ------------------------------------------------------------------
    def hover_thrust_per_rotor(self, medium: Medium, tilt: float = 0.0) -> float:
        """悬停所需单桨推力（含浮力补偿），用于配平与初始化。"""
        required = self.p.weight - self.p.buoyancy(medium)
        vertical_component = 4.0 * np.cos(tilt)
        return max(required, 0.0) / max(vertical_component, 1e-6)
