"""控制分配与几何映射的单元测试。"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cm_uav import TiltRotorAllocator, load_default
from cm_uav.geometry import body_wrench, effectiveness_matrix


def test_effectiveness_shape():
    p = load_default()
    B = effectiveness_matrix(np.full(4, 3.0), np.zeros(4), p.rotor)
    assert B.shape == (6, 8)


def test_jacobian_consistency():
    """控制效率矩阵应与有限差分一致。"""
    p = load_default()
    rng = np.random.default_rng(0)
    T = rng.uniform(1.0, 5.0, 4)
    th = rng.uniform(-0.5, 0.5, 4)
    w0 = body_wrench(T, th, p.rotor)
    B = effectiveness_matrix(T, th, p.rotor)
    du = rng.uniform(-1e-7, 1e-7, 8)
    w1 = body_wrench(T + du[0:4], th + du[4:8], p.rotor)
    assert np.allclose(B @ du, w1 - w0, atol=1e-6)


def test_hover_allocation_air():
    """空中锁定倾转时，悬停指令应给出等分推力、零倾角。"""
    p = load_default()
    alloc = TiltRotorAllocator(p)
    w_ref = np.array([0.0, 0.0, p.weight, 0.0, 0.0, 0.0])
    u = alloc.allocate(w_ref, np.zeros(8), p.air, ignore_tilt=True)
    assert np.allclose(u[0:4], p.weight / 4.0, rtol=0.15)
    assert np.max(np.abs(u[4:8])) < 1e-6


def test_water_forward_uses_tilt():
    """水中前向力应主要由倾转角承担，机体不倾斜。"""
    p = load_default()
    alloc = TiltRotorAllocator(p)
    hover = p.weight - p.buoyancy(p.water)
    w_ref = np.array([8.0, 0.0, hover, 0.0, 0.0, 0.0])
    u = alloc.allocate(w_ref, np.zeros(8), p.water, ignore_tilt=False)
    assert np.mean(u[4:8]) > np.deg2rad(10.0)  # 明显倾转
    assert np.all(u[0:4] >= 0.0)


def test_pinv_matches_constrained_without_saturation():
    """小指令下伪逆分配与约束分配结果接近。"""
    p = load_default()
    alloc = TiltRotorAllocator(p)
    w_ref = np.array([2.0, 0.0, 8.0, 0.0, 0.0, 0.0])
    u_c = alloc.allocate(w_ref, np.zeros(8), p.air, ignore_tilt=False)
    u_p = alloc.allocate_pinv(w_ref, np.zeros(8), p.air)
    assert np.allclose(u_c[0:4], u_p[0:4], atol=0.5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
