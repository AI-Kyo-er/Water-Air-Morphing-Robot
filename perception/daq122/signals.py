"""水声与磁场信号模型（用于感知算法的可复现验证）。

在水池 / 外场实测数据接入前，用物理模型合成符合实际特征的传感器信号：
    - 水听器：目标辐射噪声的螺旋桨叶频线谱 + 水声背景噪声；
    - 磁力计：铁磁目标通过时的偶极子磁异常（双极子脉冲）。
合成信号用于验证检测与融合算法，实测数据接入同一接口后即可直接替换。
"""

from __future__ import annotations

import numpy as np


def synth_hydrophone(
    fs: float = 200e3,
    duration: float = 1.0,
    tones: tuple[float, ...] = (350.0, 700.0),
    tone_amp: float = 1.0,
    noise_std: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """合成水听器信号：叶频线谱 + 宽带噪声。

    Args:
        fs: 采样率，Hz。
        duration: 时长，s。
        tones: 线谱频率，Hz（如螺旋桨叶频及其谐波）。
        tone_amp: 线谱幅值。
        noise_std: 背景噪声标准差。
        seed: 随机种子。

    Returns:
        (t, x, truth)：时间、信号、真值字典。
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for k, f in enumerate(tones):
        phase = rng.uniform(0, 2 * np.pi)
        x += tone_amp * (0.9 ** k) * np.sin(2 * np.pi * f * t + phase)
    x += rng.normal(0.0, noise_std, n)
    truth = {"tones": tones, "noise_std": noise_std, "fs": fs}
    return t, x, truth


def synth_magnetometer(
    fs: float = 1000.0,
    duration: float = 6.0,
    t0: float = 3.0,
    amplitude: float = 50.0,
    depth: float = 1.0,
    noise_std: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """合成磁力计信号：铁磁目标的偶极子磁异常。

    垂直分量近似为双极子脉冲：
        B(t) = A * (t0 - t) / ((t0 - t)^2 + d^2)^{3/2}

    Args:
        fs: 采样率，Hz。
        duration: 时长，s。
        t0: 目标最近通过时刻，s。
        amplitude: 异常幅值，nT。
        depth: 最近通过距离，m。
        noise_std: 背景噪声标准差，nT。
        seed: 随机种子。

    Returns:
        (t, b, truth)。
    """
    rng = np.random.default_rng(seed + 1)
    n = int(fs * duration)
    t = np.arange(n) / fs
    dt = t0 - t
    b = amplitude * dt / np.power(dt ** 2 + depth ** 2, 1.5)
    b += rng.normal(0.0, noise_std, n)
    truth = {"t0": t0, "amplitude": amplitude, "depth": depth, "noise_std": noise_std}
    return t, b, truth
