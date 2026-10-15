"""DAQ122 采集数据读取。

DAQ122 为基于 FPGA 的多通道高速采集器（单通道最高 200 kHz），水听器（声压）
与磁力计（磁场）均经其数字化后记录为 CSV / 二进制文件。本模块提供统一的读取
接口，供后续信号处理与融合使用。
"""

from __future__ import annotations

import pathlib

import numpy as np


def load_csv(path: str | pathlib.Path, time_col: int = 0,
             value_cols: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """读取 DAQ122 CSV 记录。

    Args:
        path: CSV 文件路径。
        time_col: 时间列索引。
        value_cols: 数值列索引；None 表示除时间列外的所有列。

    Returns:
        (t, values)：时间向量 (N,) 与数据 (N,) 或 (N, C)。
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    t = raw[:, time_col]
    cols = value_cols if value_cols is not None else [c for c in range(raw.shape[1]) if c != time_col]
    values = raw[:, cols]
    if values.shape[1] == 1:
        values = values[:, 0]
    return t, values


def load_binary(path: str | pathlib.Path, dtype: str = "<i2",
                sample_rate: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """读取 DAQ122 二进制记录（ADC 原始码）。

    Args:
        path: 二进制文件路径。
        dtype: 数据类型（默认 16 位有符号小端）。
        sample_rate: 采样率，Hz；给定时返回时间轴，否则返回采样序号。

    Returns:
        (t, values)。
    """
    values = np.fromfile(path, dtype=np.dtype(dtype))
    if sample_rate is None:
        t = np.arange(values.size, dtype=float)
    else:
        t = np.arange(values.size, dtype=float) / sample_rate
    return t, values.astype(float)
