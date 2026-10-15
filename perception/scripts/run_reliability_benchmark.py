"""独立校准集与多种子评估集下的传感器退化融合基准。"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception.fusion.reliability import (
    FusionConfig, GaussianCalibration, ReliabilityFusion, SensorEvidence,
)
from scripts.research_common import classification_metrics, write_csv, write_metadata

CONFIG = ROOT / "perception/configs/reliability.yaml"
OUT = ROOT / "experiments/perception/reliability_benchmark"
METHODS = ("single_acoustic", "fixed_fusion", "no_quality", "no_temporal", "proposed")


def calibrate(raw: dict) -> dict[str, GaussianCalibration]:
    """使用专用种子拟合检测分数模型，不读取评估集。"""
    rng = np.random.default_rng(raw["calibration_seed"])
    n = raw["calibration_samples"]
    labels = np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
    models = {}
    for name in raw["modalities"]:
        scores = rng.normal(raw["score_background_mean"] + labels * (
            raw["score_target_mean"] - raw["score_background_mean"]), raw["score_std"])
        models[name] = GaussianCalibration.fit(scores, labels)
    return models


def make_sequence(raw: dict, seed: int) -> tuple[np.ndarray, np.ndarray, list[list[SensorEvidence]]]:
    """生成检测层开发记录；quality 为外部信号质量输入，不使用目标真值推断。"""
    rng = np.random.default_rng(seed)
    t = np.arange(0, raw["duration_s"], raw["dt_s"])
    truth = np.zeros(len(t), bool)
    for start, end in raw["target_intervals_s"]:
        truth |= (t >= start) & (t < end)
    frames = []
    cached_sonar = 0.0
    cached_timestamp = 0.0
    for k, now in enumerate(t):
        shared = rng.normal(0, raw["common_noise_std"])
        frame = []
        for name in raw["modalities"]:
            score = rng.normal(raw["score_target_mean"] if truth[k] else
                               raw["score_background_mean"], raw["score_std"])
            quality, available, timestamp = 1.0, True, now
            group = "acoustic_family" if name in ("acoustic", "sonar") else name
            if name in ("acoustic", "sonar"):
                score += shared
            if name == "acoustic" and raw["acoustic_interference_s"][0] <= now < raw[
                    "acoustic_interference_s"][1]:
                score = rng.normal(raw["interference_score"], raw["interference_std"])
                quality = raw["degraded_quality"]
            if name == "vision" and raw["missing_vision_s"][0] <= now < raw["missing_vision_s"][1]:
                available = False
            if name == "sonar":
                stale = raw["stale_sonar_s"][0] <= now < raw["stale_sonar_s"][1]
                if stale:
                    score, timestamp = cached_sonar, cached_timestamp
                else:
                    cached_sonar, cached_timestamp = score, now
            frame.append(SensorEvidence(name, float(score), quality, float(timestamp), available, group))
        frames.append(frame)
    return t, truth, frames


def main() -> None:
    """输出全部种子指标、逐帧证据、校准参数和第一种子的决策轨迹。"""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["benchmark"]
    cfg = FusionConfig.from_yaml(CONFIG)
    calibration = calibrate(raw)
    metrics, traces, observations = [], [], []
    for seed in raw["evaluation_seeds"]:
        t, truth, frames = make_sequence(raw, seed)
        for method in METHODS:
            setting = replace(cfg, dwell_s=0) if method == "no_temporal" else cfg
            fusion = ReliabilityFusion(calibration, setting)
            probabilities, predictions = [], []
            for k, frame in enumerate(frames):
                if method in ("fixed_fusion", "single_acoustic"):
                    eligible = [ev for ev in frame if ev.available and (
                        method != "single_acoustic" or ev.name == "acoustic")]
                    llr = sum(np.clip(calibration[ev.name].log_likelihood_ratio(ev.score),
                                      -cfg.llr_clip, cfg.llr_clip) for ev in eligible)
                    probability = float(expit(logit(cfg.prior) + llr))
                    declared = probability >= cfg.declare_on
                    support = float(len(eligible))
                else:
                    current = [replace(ev, quality=1.0) for ev in frame] if method == "no_quality" else frame
                    result = fusion.update(current, t[k])
                    probability, declared, support = result.posterior, result.declared, result.support
                probabilities.append(probability)
                predictions.append(declared)
                if seed == raw["evaluation_seeds"][0]:
                    traces.append({"time_s": round(t[k], 4), "method": method,
                                   "target_present": int(truth[k]), "posterior": probability,
                                   "declared": int(declared), "support": support,
                                   "source_kind": "simulation"})
            row = {"seed": seed, "method": method}
            row.update(classification_metrics(truth, np.array(probabilities), np.array(predictions)))
            metrics.append(row)
        for k, frame in enumerate(frames):
            for ev in frame:
                observations.append({"seed": seed, "time_s": round(t[k], 4),
                                     "target_present": int(truth[k]),
                                     **asdict(ev), "source_kind": "simulation"})
    write_csv(OUT / "data/metrics.csv", metrics)
    write_csv(OUT / "data/traces.csv", traces)
    write_csv(OUT / "data/evidence.csv", observations)
    write_csv(OUT / "data/calibration.csv", [{"modality": name, **asdict(model)}
                                           for name, model in calibration.items()])
    write_metadata(OUT / "run_metadata.json", [CONFIG, Path(__file__),
                   ROOT / "perception/fusion/reliability.py"],
                   [raw["calibration_seed"], *raw["evaluation_seeds"]],
                   "校准与评估种子分离；包含质量下降、视觉缺失与声呐过期的时序场景。")
    print(f"完成可信融合基准：{len(raw['evaluation_seeds'])} 个种子，{len(metrics)} 组结果")


if __name__ == "__main__":
    main()
