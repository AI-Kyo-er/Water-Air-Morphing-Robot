"""水声窄带检测与磁异常检测。

- 水声：基于 Welch 功率谱，在目标频率近邻搜索线谱峰值，估计信噪比（dB）；
- 磁异常：以偶极子脉冲为模板做匹配滤波，峰值位置即为目标最近通过时刻。
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def detect_tonal(
    x: np.ndarray, fs: float, f0: float, rel_band: float = 0.05
) -> dict:
    """检测频率 f0 附近的窄带线谱。

    Args:
        x: 水听器信号。
        fs: 采样率，Hz。
        f0: 目标频率，Hz。
        rel_band: 搜索相对带宽。

    Returns:
        {"snr_db", "f_peak", "score"}，score 为归一化检测量（snr_db/常数）。
    """
    f, psd = signal.welch(x, fs=fs, nperseg=min(len(x), 8192))
    mask = (f >= f0 * (1 - rel_band)) & (f <= f0 * (1 + rel_band))
    if not np.any(mask):
        return {"snr_db": -np.inf, "f_peak": f0, "score": 0.0}
    band = psd[mask]
    f_band = f[mask]
    peak_idx = int(np.argmax(band))
    peak = band[peak_idx]
    # 噪声底：整段谱的中位数
    floor = np.median(psd) + 1e-18
    snr_db = 10.0 * np.log10(peak / floor)
    return {
        "snr_db": float(snr_db),
        "f_peak": float(f_band[peak_idx]),
        "score": float(max(snr_db, 0.0)),
    }


def _dipole_template(n: int, fs: float, depth: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs - (n / fs) / 2.0
    h = -t / np.power(t ** 2 + depth ** 2, 1.5)
    return h / (np.linalg.norm(h) + 1e-18)


def detect_magnetic_anomaly(
    b: np.ndarray, t: np.ndarray, depth: float = 1.0
) -> dict:
    """磁异常匹配滤波检测。

    Args:
        b: 磁力计信号（已去基线）。
        t: 时间向量。
        depth: 模板特征深度，m。

    Returns:
        {"t0", "amplitude", "snr", "score"}。
    """
    fs = 1.0 / (t[1] - t[0])
    baseline = np.median(b)
    sig = b - baseline
    sigma = np.std(sig) + 1e-12
    h = _dipole_template(len(sig) // 2, fs, depth)
    corr = signal.correlate(sig, h, mode="same")
    corr /= sigma
    idx = int(np.argmax(np.abs(corr)))
    snr = float(np.max(np.abs(corr)))
    return {
        "t0": float(t[idx]),
        "amplitude": float(corr[idx] * sigma),
        "snr": snr,
        "score": float(max(snr, 0.0)),
    }
