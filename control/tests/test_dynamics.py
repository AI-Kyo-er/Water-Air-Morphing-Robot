"""跨介质动力学的单元测试。"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cm_uav import CrossMediumDynamics, VehicleState, load_default


def _hover_inputs(p, dyn, medium):
    T = dyn.hover_thrust_per_rotor(medium)
    return np.concatenate([np.full(4, T), np.zeros(4)])


def test_air_hover_holds_position():
    p = load_default()
    dyn = CrossMediumDynamics(p)
    u = _hover_inputs(p, dyn, p.air)
    x = VehicleState().to_vector()
    for _ in range(200):
        x = dyn.step(x, u, p.air, 0.005)
    st = VehicleState.from_vector(x)
    assert abs(st.position[2]) < 5e-3
    assert np.linalg.norm(st.velocity) < 5e-3


def test_water_buoyancy_reduces_thrust_need():
    p = load_default()
    dyn = CrossMediumDynamics(p)
    t_air = dyn.hover_thrust_per_rotor(p.air)
    t_water = dyn.hover_thrust_per_rotor(p.water)
    assert t_water < t_air  # 水中靠浮力大幅减小推力需求


def test_forward_tilt_produces_forward_force():
    p = load_default()
    dyn = CrossMediumDynamics(p)
    tilt = np.deg2rad(45.0)
    T = 2.0
    u = np.concatenate([np.full(4, T), np.full(4, tilt)])
    x = VehicleState().to_vector()
    for _ in range(40):
        x = dyn.step(x, u, p.water, 0.005)
    st = VehicleState.from_vector(x)
    assert st.velocity[0] > 0.0  # 前向加速


def test_rotation_stays_orthonormal():
    p = load_default()
    dyn = CrossMediumDynamics(p)
    u = _hover_inputs(p, dyn, p.air)
    u[4:8] = np.deg2rad(10.0)
    x = VehicleState().to_vector()
    for _ in range(500):
        x = dyn.step(x, u, p.air, 0.005)
    R = x[6:15].reshape(3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
