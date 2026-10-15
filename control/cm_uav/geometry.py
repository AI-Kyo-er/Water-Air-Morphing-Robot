"""推进系统的几何—力学映射（全项目唯一真源）。

本模块把「推进单元的推力与倾转角」映射为作用在机体质心上的合力与合力矩，
并提供该映射对执行量的雅可比（控制效率矩阵）。动力学、控制分配、固件
均必须复用这里的定义，避免出现两套不一致的模型。

执行量向量 u = [T1, T2, T3, T4, theta1, theta2, theta3, theta4]：
    T_i    : 第 i 套推进单元推力，N，>= 0
    theta_i: 第 i 套推进单元倾转角，rad，绕机体 y 轴、由竖直指向 tilt_direction

每个推进单元的推力方向：
    n_i(theta) = cos(theta) * z_hat + sin(theta) * d_i
其中 d_i 为安装时设定的倾转趋向方向（本机型取机体 x 轴正向，"工"字布局使
全部螺旋桨在水中可 100% 参与前向推进）。
"""

from __future__ import annotations

import numpy as np

from .params import RotorArray

Z_HAT = np.array([0.0, 0.0, 1.0])


def thrust_directions(theta: np.ndarray, rotor: RotorArray) -> np.ndarray:
    """给定倾转角，返回各推进单元的推力方向 (4, 3)。"""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    d = rotor.tilt_directions
    return np.cos(theta)[:, None] * Z_HAT[None, :] + np.sin(theta)[:, None] * d


def thrust_direction_derivatives(theta: np.ndarray, rotor: RotorArray) -> np.ndarray:
    """推力方向对倾转角的导数 dn_i/dtheta (4, 3)。"""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    d = rotor.tilt_directions
    return -np.sin(theta)[:, None] * Z_HAT[None, :] + np.cos(theta)[:, None] * d


def body_wrench(T: np.ndarray, theta: np.ndarray, rotor: RotorArray) -> np.ndarray:
    """由执行量计算机体系合力 / 合力矩，返回 (6,) = [Fx, Fy, Fz, tx, ty, tz]。"""
    T = np.asarray(T, dtype=float).reshape(-1)
    n = thrust_directions(theta, rotor)
    forces = T[:, None] * n  # (4, 3)

    force = forces.sum(axis=0)
    torque = np.zeros(3)
    for i in range(len(T)):
        torque += np.cross(rotor.positions[i], forces[i])
        # 旋翼反扭矩：沿自转轴（即推力方向），符号由自转方向决定
        torque += rotor.spin_signs[i] * rotor.reaction_torque_coeff * T[i] * n[i]
    return np.concatenate([force, torque])


def effectiveness_matrix(T: np.ndarray, theta: np.ndarray, rotor: RotorArray) -> np.ndarray:
    """控制效率矩阵 B = d(wrench)/d(u)，形状 (6, 8)。"""
    T = np.asarray(T, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n = thrust_directions(theta, rotor)
    dn = thrust_direction_derivatives(theta, rotor)

    B = np.zeros((6, 8))
    for i in range(len(T)):
        # --- 对推力 T_i 的偏导 ---
        B[0:3, i] = n[i]
        B[3:6, i] = np.cross(rotor.positions[i], n[i]) + rotor.spin_signs[i] * rotor.reaction_torque_coeff * n[i]
        # --- 对倾转角 theta_i 的偏导 ---
        B[0:3, 4 + i] = T[i] * dn[i]
        B[3:6, 4 + i] = np.cross(rotor.positions[i], T[i] * dn[i]) + \
            rotor.spin_signs[i] * rotor.reaction_torque_coeff * T[i] * dn[i]
    return B
