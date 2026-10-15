"""水声 / 磁异常检测与融合的单元测试。"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from perception.daq122.detectors import detect_magnetic_anomaly, detect_tonal
from perception.daq122.signals import synth_hydrophone, synth_magnetometer
from perception.fusion.fusion import Evidence, MultiModalFusion


def test_tonal_detects_present():
    _, x, _ = synth_hydrophone(fs=200e3, duration=0.1, tones=(350.0,),
                               tone_amp=1.0, noise_std=1.0, seed=0)
    s = detect_tonal(x, 200e3, 350.0)["snr_db"]
    assert s > 10.0


def test_tonal_lower_without_target():
    _, x, _ = synth_hydrophone(fs=200e3, duration=0.1, tones=(350.0,),
                               tone_amp=0.0, noise_std=1.0, seed=0)
    s = detect_tonal(x, 200e3, 350.0)["snr_db"]
    assert s < 10.0


def test_magnetic_detects_present():
    fs = 1000.0
    _, b, truth = synth_magnetometer(fs=fs, duration=6.0, t0=3.0,
                                     amplitude=20.0, depth=1.0, noise_std=1.0, seed=1)
    r = detect_magnetic_anomaly(b, np.arange(b.size) / fs)
    assert r["snr"] > 10.0
    assert abs(r["t0"] - truth["t0"]) < 0.5


def test_fusion_separates_present_and_absent():
    fuse = MultiModalFusion(prior=0.1)
    strong = [
        Evidence("acoustic", 19.0, sensitivity=0.30, reference=8.0),
        Evidence("magnetic", 60.0, sensitivity=0.25, reference=4.0),
    ]
    weak = [
        Evidence("acoustic", 1.0, sensitivity=0.30, reference=8.0),
        Evidence("magnetic", 2.0, sensitivity=0.25, reference=4.0),
    ]
    assert fuse.fuse(strong)["posterior"] > 0.9
    assert fuse.fuse(weak)["posterior"] < 0.1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
