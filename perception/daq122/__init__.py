"""感知层：水听器 / 磁力计数据读取与信号处理。

提供 DAQ122 采集文件的读取工具，以及水声窄带检测与磁异常检测的实现。
"""

from .daq_reader import load_csv, load_binary
from .signals import synth_hydrophone, synth_magnetometer
from .detectors import detect_tonal, detect_magnetic_anomaly

__all__ = [
    "load_csv",
    "load_binary",
    "synth_hydrophone",
    "synth_magnetometer",
    "detect_tonal",
    "detect_magnetic_anomaly",
]
