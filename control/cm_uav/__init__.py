"""cm_uav —— 跨介质矢量推进无人机控制算法包。

模块划分：
    params      : 整机与介质参数
    geometry    : 推进系统几何—力学映射（唯一真源）
    dynamics    : 跨介质六自由度动力学
    allocation  : 过驱动控制分配（核心）
    controller  : 介质自适应级联控制与模态管理（核心）
    simulator   : 闭环数值仿真环境
"""

from .allocation import AllocationWeights, TiltRotorAllocator
from .controller import CascadedController, ModeManager, Setpoint
from .dynamics import CrossMediumDynamics, VehicleState
from .params import Medium, VehicleParams, load_default
from .simulator import Simulator

__all__ = [
    "AllocationWeights",
    "TiltRotorAllocator",
    "CascadedController",
    "ModeManager",
    "Setpoint",
    "CrossMediumDynamics",
    "VehicleState",
    "Medium",
    "VehicleParams",
    "load_default",
    "Simulator",
]
