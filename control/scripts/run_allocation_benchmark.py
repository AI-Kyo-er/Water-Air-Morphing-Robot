"""统一输入下比较约束分配、基线与消融；输出 CSV 和来源记录。"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from control.cm_uav.constrained_allocation import AllocationConfig, ConstrainedAllocator
from control.cm_uav.geometry import body_wrench, effectiveness_matrix
from control.cm_uav.params import load_default
from scripts.research_common import write_csv, write_metadata

CONFIG = ROOT / "control/configs/research.yaml"
OUT = ROOT / "experiments/control/allocation_benchmark"
METHODS = ("pinv_clip", "common_tilt", "proposed", "no_rate", "no_health")


def make_case(case: str, t: np.ndarray, rng: np.random.Generator
              ) -> tuple[np.ndarray, np.ndarray]:
    """构造具有前向力、深度力和姿态力矩的模型输入，所有方法共用。"""
    target = np.column_stack([
        5 + 2 * np.sin(0.8 * t), np.zeros_like(t), 12 + np.sin(0.5 * t),
        0.35 * np.sin(0.7 * t), 0.35 * np.cos(0.6 * t),
        0.6 * np.sin(0.9 * t),
    ])
    target += rng.normal(0, 0.015, target.shape)
    target[:, 1] = 0
    health = np.ones((len(t), 4))
    if case == "yaw_maneuver":
        target[:, 5] = 1.6 * np.sin(0.9 * t)
    elif case == "thrust_loss":
        health[t >= 5, 0] = 0.55
        health[t >= 7, 1] = 0.85
    elif case == "fast_command":
        target[:, 0] = np.where(np.sin(4 * t) >= 0, 10.0, -6.0)
        target[:, 5] = 1.2 * np.sin(3 * t)
    elif case == "medium_transition":
        airborne = (t < 4.0) | (t >= 8.0)
        target[airborne, 0] = 0.0
        target[airborne, 2] = 35.8
        target[airborne, 3:6] *= 0.1
    return target, health


def main() -> None:
    """执行所有场景与种子，禁止根据结果挑选样本。"""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["benchmark"]
    cfg = AllocationConfig.from_yaml(CONFIG)
    params = load_default()
    dt = raw["dt_s"]
    t = np.arange(0, raw["duration_s"], dt)
    metrics, trace = [], []
    for case in raw["cases"]:
        for seed in raw["seeds"]:
            targets, health = make_case(case, t, np.random.default_rng(seed))
            for method in METHODS:
                setting = replace(cfg, independent_tilt=method != "common_tilt",
                                  enforce_rates=method != "no_rate")
                allocator = ConstrainedAllocator(params, setting)
                previous = np.r_[np.full(4, raw["initial_thrust_n"]), np.zeros(4)]
                errors, rate_hits, changes, durations = [], [], [], []
                for k, now in enumerate(t):
                    in_air = case == "medium_transition" and (now < 4.0 or now >= 8.0)
                    medium = params.air if in_air else params.water
                    air_fraction = float(in_air)
                    begin = time.perf_counter()
                    if method == "pinv_clip":
                        jac = effectiveness_matrix(previous[:4] * health[k],
                                                   previous[4:], params.rotor)
                        jac[:, :4] *= health[k][None, :]
                        residual = targets[k] - allocator.wrench(previous, health[k])
                        delta = np.linalg.pinv(allocator.weights[:, None] * jac,
                                              rcond=1e-6) @ (allocator.weights * residual)
                        low, high, _ = allocator.bounds(previous, medium, dt, air_fraction)
                        command = np.clip(previous + delta, low, high)
                    else:
                        knowledge = np.ones(4) if method == "no_health" else health[k]
                        command = allocator.allocate(targets[k], previous, medium,
                                                     dt, knowledge, air_fraction).command
                    durations.append((time.perf_counter() - begin) * 1000)
                    actual = body_wrench(command[:4] * health[k], command[4:], params.rotor)
                    error = float(np.linalg.norm(allocator.weights * (targets[k] - actual)))
                    rate_hit = bool(np.any(abs(command[:4] - previous[:4]) >
                                           cfg.thrust_rate_n_s * dt + 1e-7) or np.any(
                        abs(command[4:] - previous[4:]) > params.rotor.tilt_rate_limit * dt + 1e-7))
                    errors.append(error)
                    rate_hits.append(rate_hit)
                    changes.append(float(np.linalg.norm((command - previous) /
                        np.r_[np.full(4, params.rotor.thrust_max_water),
                              np.full(4, params.rotor.tilt_limit)])))
                    if seed == raw["seeds"][0]:
                        row = {"case": case, "method": method, "time_s": round(now, 4),
                               "source_kind": "simulation", "normalized_error": error,
                               "rate_violation": int(rate_hit), "medium": medium.name}
                        for i, axis in enumerate(("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm")):
                            row[f"target_{axis}"] = targets[k, i]
                            row[f"achieved_{axis}"] = actual[i]
                        for i in range(4):
                            row[f"thrust_{i+1}_n"] = command[i]
                            row[f"tilt_{i+1}_rad"] = command[i+4]
                            row[f"health_{i+1}"] = health[k, i]
                        trace.append(row)
                    previous = command
                metrics.append({"case": case, "seed": seed, "method": method,
                                "normalized_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                                "p95_normalized_error": float(np.percentile(errors, 95)),
                                "rate_violation_fraction": float(np.mean(rate_hits)),
                                "mean_command_change": float(np.mean(changes)),
                                "solve_p95_ms": float(np.percentile(durations, 95))})
        print(f"完成控制分配场景：{case}", flush=True)
    write_csv(OUT / "data/metrics.csv", metrics)
    write_csv(OUT / "data/traces.csv", trace)
    write_metadata(OUT / "run_metadata.json", [CONFIG, ROOT / "control/configs/vehicle_default.yaml",
                   ROOT / "control/cm_uav/constrained_allocation.py", Path(__file__)],
                   raw["seeds"], "同输入、同几何、同边界的约束分配对比；no_rate 为去除速率约束的消融。")
    print(f"结果已保存：{OUT}")


if __name__ == "__main__":
    main()
