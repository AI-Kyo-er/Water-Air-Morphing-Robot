"""跨介质自适应级联控制器与模态管理（核心算法模块之一）。

设计要点（创新点）：
1. **统一控制律、介质自适应增益**：位置—速度—姿态级联结构在空中与水中
   完全一致，仅通过模态管理器切换增益与力分解策略，避免两套割裂的控制器。
2. **介质自适应力分配 (Medium-Adaptive Force Decomposition)**：
   水平方向所需的推进力在「改变机体姿态（气动惯用）」与「改变旋翼倾转
   （水动惯用）」之间按授权系数 beta 连续分配——
       beta = 1 → 全部由机体姿态提供（空中、高效）；
       beta = 0 → 全部由旋翼倾转提供（水中、保持机体平直降低阻力）。
   该系数随介质与过渡进度平滑变化，从数学上统一了两种工作模态。
3. **连续授权调度**：跨介质切换时逐步调整增益与姿态授权。
   当前控制器不使用积分作用，不能据此宣称完成积分状态的无扰转移。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .allocation import TiltRotorAllocator
from .dynamics import VehicleState, euler_from_R, hat
from .params import Medium, VehicleParams


def rotation_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def align_z_to(d: np.ndarray, yaw: float) -> np.ndarray:
    """构造使机体 z 轴对齐方向 d、且偏航为 yaw 的旋转矩阵。"""
    z_b = d / np.linalg.norm(d)
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    if abs(np.dot(x_c, z_b)) > 0.99:
        x_c = np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2), 0.0])
    y_b = np.cross(z_b, x_c)
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    return np.column_stack([x_b, y_b, z_b])


@dataclass
class Setpoint:
    """轨迹设定点。"""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0


@dataclass
class ControllerGains:
    """控制增益（按介质调度）。"""

    pos: np.ndarray
    vel: np.ndarray
    att: np.ndarray
    rate: np.ndarray


AIR_GAINS = ControllerGains(
    pos=np.array([1.6, 1.6, 2.0]),
    vel=np.array([2.6, 2.6, 3.0]),
    att=np.array([14.0, 14.0, 6.0]),
    rate=np.array([4.0, 4.0, 2.6]),
)

WATER_GAINS = ControllerGains(
    pos=np.array([1.4, 1.4, 2.4]),
    vel=np.array([2.4, 2.4, 3.0]),
    att=np.array([12.0, 12.0, 5.0]),
    rate=np.array([3.6, 3.6, 2.4]),
)


class ModeManager:
    """介质模态管理：增益调度与连续力分配授权。"""

    def __init__(self, initial_medium: str = "air"):
        self.medium_name = initial_medium
        self.blend = 1.0 if initial_medium == "air" else 0.0  # 1=空中, 0=水中
        self._reset_requests = 0

    def command(self, medium: str) -> None:
        if medium != self.medium_name:
            self.medium_name = medium
            self._reset_requests += 1

    def update(self, dt: float, rate: float = 1.5) -> None:
        """将授权系数向目标平滑过渡。"""
        target = 1.0 if self.medium_name == "air" else 0.0
        step = rate * dt
        if abs(target - self.blend) <= step:
            self.blend = target
        else:
            self.blend += np.sign(target - self.blend) * step

    def gains(self) -> ControllerGains:
        """按 blend 对空中/水中增益做线性插值。"""
        b = self.blend
        return ControllerGains(
            pos=b * AIR_GAINS.pos + (1 - b) * WATER_GAINS.pos,
            vel=b * AIR_GAINS.vel + (1 - b) * WATER_GAINS.vel,
            att=b * AIR_GAINS.att + (1 - b) * WATER_GAINS.att,
            rate=b * AIR_GAINS.rate + (1 - b) * WATER_GAINS.rate,
        )

    @property
    def attitude_authority(self) -> float:
        """水平力中由机体姿态承担的比例 beta。"""
        return self.blend

    def consume_reset(self) -> bool:
        if self._reset_requests > 0:
            self._reset_requests -= 1
            return True
        return False


class CascadedController:
    """位置—速度—姿态级联控制器。"""

    def __init__(self, params: VehicleParams, mode: ModeManager | None = None):
        self.p = params
        self.mode = mode or ModeManager()
        self.allocator = TiltRotorAllocator(params)
        self._integral = np.zeros(3)

    # ------------------------------------------------------------------
    def desired_wrench(
        self, state: VehicleState, sp: Setpoint, medium: Medium, dt: float
    ) -> np.ndarray:
        """计算期望机体力 / 力矩 (6,)。"""
        g = self.mode.gains()
        m_eff = self.p.mass * (1.0 + medium.added_mass_factor)
        R = state.rotation

        # ---------- 位置 / 速度环 ----------
        e_p = sp.position - state.position
        e_v = sp.velocity - state.velocity
        a_cmd = sp.accel + g.pos * e_p + g.vel * e_v
        # 限幅，避免大误差下指令加速度发散（同时限制期望倾转幅值）
        a_cmd = np.clip(a_cmd, -self.p.a_cmd_limit, self.p.a_cmd_limit)

        # 推进力需克服的重力 / 浮力（世界系）
        gravity_w = np.array([0.0, 0.0, -self.p.weight])
        buoyancy_w = np.array([0.0, 0.0, self.p.buoyancy(medium)])
        f_w = m_eff * a_cmd - gravity_w - buoyancy_w

        # ---------- 介质自适应力分解 ----------
        # beta = 1：机体 z 轴完全对齐期望推力方向（全靠姿态产生水平力，空中）
        # beta = 0：机体 z 轴竖直（全靠旋翼倾转产生水平力，水中）
        beta = self.mode.attitude_authority
        d = np.array([beta * f_w[0], beta * f_w[1], f_w[2]])
        norm = np.linalg.norm(d)
        d = np.array([0.0, 0.0, 1.0]) if norm < 1e-6 else d / norm
        R_ref = align_z_to(d, sp.yaw)

        # 机体系期望力 = 将该世界系力旋回期望姿态的机体系
        f_body_des = R_ref.T @ f_w

        # ---------- 姿态环（旋转矩阵误差）----------
        # e_rot 表示 R 相对 R_ref 的误差；控制律 tau = I(-kp e_rot - kd omega)
        e_rot = 0.5 * vee(R_ref.T @ R - R.T @ R_ref)
        tau_body = self.p.inertia * (-g.att * e_rot - g.rate * state.omega)

        return np.concatenate([f_body_des, tau_body])

    # ------------------------------------------------------------------
    def compute(
        self,
        state: VehicleState,
        sp: Setpoint,
        medium: Medium,
        u_prev: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """一步闭环：返回新的执行量 u (8,)。"""
        self.mode.update(dt)
        w_ref = self.desired_wrench(state, sp, medium, dt)
        # 空中模式锁定倾转角（旋翼保持水平），水中释放
        ignore_tilt = self.mode.blend > 0.98
        u = self.allocator.allocate(w_ref, u_prev, medium, ignore_tilt=ignore_tilt)
        return u
