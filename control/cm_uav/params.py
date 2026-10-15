"""跨介质矢量推进无人机的参数定义与配置加载。

本模块集中管理飞行器的质量、惯量、几何布局、推进单元参数，以及空中 /
水两种介质的物性参数。所有数值均来自配置文件（configs/*.yaml），
禁止在算法代码中硬编码。

参考量纲：长度 m，质量 kg，时间 s，角度 rad，力 N，力矩 N·m。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class RotorArray:
    """四套倾转推进单元的几何与性能参数。

    Attributes:
        positions: (4, 3) 各推进单元在机体系下的安装位置，单位 m。
        spin_signs: (4,) 各旋翼自转方向符号 (+1 / -1)。
        tilt_directions: (4, 3) 各推进单元倾转时推力所趋向的机体方向单位矢量。
        thrust_max_air: 单套推进单元在空气中的最大推力，N。
        thrust_max_water: 单套推进单元在水中的最大推力，N。
        reaction_torque_coeff: 反扭矩系数 k_Q（反扭矩 = k_Q * 推力）。
        tilt_limit: 倾转角幅值上限，rad。
        tilt_rate_limit: 倾转角速率上限，rad/s。
    """

    positions: np.ndarray
    spin_signs: np.ndarray
    tilt_directions: np.ndarray
    thrust_max_air: float
    thrust_max_water: float
    reaction_torque_coeff: float
    tilt_limit: float = np.deg2rad(90.0)
    tilt_rate_limit: float = np.deg2rad(180.0)
    prop_diameter: float = 0.20

    def thrust_max(self, medium: "Medium") -> float:
        return self.thrust_max_air if medium.is_air else self.thrust_max_water

    @property
    def disk_area(self) -> float:
        return np.pi * (self.prop_diameter / 2.0) ** 2


@dataclass
class Medium:
    """介质物性参数。"""

    name: str
    density: float
    is_air: bool
    linear_drag: np.ndarray  # (3,) 机体轴系线性阻尼系数
    quadratic_drag: np.ndarray  # (3,) 机体轴系二次阻尼系数
    angular_drag: np.ndarray  # (3,) 角阻尼系数
    added_mass_factor: float = 0.0  # 附加质量比例（水下显著）


@dataclass
class VehicleParams:
    """整机参数。"""

    name: str
    mass: float
    inertia: np.ndarray  # (3,) 主惯量 Ixx, Iyy, Izz
    gravity: float
    volume: float  # 排水体积 m^3
    rotor: RotorArray
    air: Medium
    water: Medium
    max_speed: float = 15.0
    a_cmd_limit: float = 12.0  # 期望加速度指令限幅，m/s^2
    notes: str = ""

    # ---- 便捷属性 -------------------------------------------------
    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    def buoyancy(self, medium: Medium) -> float:
        return medium.density * self.volume * self.gravity

    def buoyancy_ratio(self, medium: Medium) -> float:
        """浮重比：>1 表示正浮力。"""
        return self.buoyancy(medium) / self.weight

    def medium(self, name: str) -> Medium:
        key = name.lower()
        if key in ("air", "aerial"):
            return self.air
        if key in ("water", "underwater", "aquatic"):
            return self.water
        raise ValueError(f"未知介质：{name}")

    # ---- 加载 -----------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleParams":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VehicleParams":
        rot = raw["rotor"]
        rotor = RotorArray(
            positions=np.asarray(rot["positions"], dtype=float),
            spin_signs=np.asarray(rot["spin_signs"], dtype=float),
            tilt_directions=np.asarray(rot["tilt_directions"], dtype=float),
            thrust_max_air=float(rot["thrust_max_air"]),
            thrust_max_water=float(rot["thrust_max_water"]),
            reaction_torque_coeff=float(rot["reaction_torque_coeff"]),
            tilt_limit=float(np.deg2rad(rot.get("tilt_limit_deg", 90.0))),
            tilt_rate_limit=float(np.deg2rad(rot.get("tilt_rate_limit_deg_s", 180.0))),
            prop_diameter=float(rot.get("prop_diameter", 0.20)),
        )
        # 归一化倾转方向
        norms = np.linalg.norm(rotor.tilt_directions, axis=1, keepdims=True)
        rotor.tilt_directions = rotor.tilt_directions / np.clip(norms, 1e-9, None)

        def _medium(d: dict[str, Any], is_air: bool) -> Medium:
            return Medium(
                name=d["name"],
                density=float(d["density"]),
                is_air=is_air,
                linear_drag=np.asarray(d["linear_drag"], dtype=float),
                quadratic_drag=np.asarray(d["quadratic_drag"], dtype=float),
                angular_drag=np.asarray(d["angular_drag"], dtype=float),
                added_mass_factor=float(d.get("added_mass_factor", 0.0)),
            )

        return cls(
            name=raw.get("name", "cm-uav"),
            mass=float(raw["mass"]),
            inertia=np.asarray(raw["inertia"], dtype=float),
            gravity=float(raw.get("gravity", 9.80665)),
            volume=float(raw["volume"]),
            rotor=rotor,
            air=_medium(raw["air"], True),
            water=_medium(raw["water"], False),
            max_speed=float(raw.get("max_speed", 15.0)),
            a_cmd_limit=float(raw.get("a_cmd_limit", 12.0)),
            notes=raw.get("notes", ""),
        )


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "vehicle_default.yaml"


def load_default() -> VehicleParams:
    """加载默认整机配置。"""
    return VehicleParams.from_yaml(DEFAULT_CONFIG)
