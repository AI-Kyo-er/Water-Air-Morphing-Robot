"""传感器质量、新鲜度和相关性约束下的目标存在性融合。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.special import expit, logit


@dataclass(frozen=True)
class FusionConfig:
    """单周期证据融合与时间判定参数，时间单位为秒。"""

    prior: float = 0.25
    declare_on: float = 0.78
    declare_off: float = 0.35
    dwell_s: float = 0.4
    max_age_s: float = 0.6
    llr_clip: float = 8.0
    correlation_discount: float = 0.5
    min_std: float = 0.1

    @classmethod
    def from_yaml(cls, path: str | Path) -> FusionConfig:
        """读取 fusion 配置段。"""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["fusion"]
        return cls(**raw)


@dataclass(frozen=True)
class GaussianCalibration:
    """由独立校准样本估计的条件高斯分数模型。"""

    background_mean: float
    background_std: float
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray,
            min_std: float = 0.1) -> GaussianCalibration:
        """labels 为 0/1，校准集必须同时含两类且每类至少两条。"""
        scores, labels = np.asarray(scores), np.asarray(labels)
        if scores.shape != labels.shape or not np.all(np.isfinite(scores)):
            raise ValueError("分数和标签应同形且分数有限")
        if not np.all(np.isin(labels, [0, 1])) or min_std <= 0:
            raise ValueError("标签应为 0/1，最小标准差应为正")
        groups = [scores[labels == value] for value in (0, 1)]
        if min(map(len, groups)) < 2:
            raise ValueError("每类至少需要两条校准记录")
        return cls(float(groups[0].mean()), max(float(groups[0].std()), min_std),
                   float(groups[1].mean()), max(float(groups[1].std()), min_std))

    def log_likelihood_ratio(self, score: float) -> float:
        """计算 log p(score|目标) / p(score|背景)，不混用不同量纲分数。"""
        background = -np.log(self.background_std) - 0.5 * (
            (score - self.background_mean) / self.background_std) ** 2
        target = -np.log(self.target_std) - 0.5 * (
            (score - self.target_mean) / self.target_std) ** 2
        return float(target - background)


@dataclass(frozen=True)
class SensorEvidence:
    """一帧检测证据；quality 为上游信号质量，timestamp_s 为采集时刻。"""

    name: str
    score: float
    quality: float
    timestamp_s: float
    available: bool = True
    correlation_group: str = ""


@dataclass
class FusionResult:
    """目标概率、稳定判定、有效支持度和各通道贡献。"""

    posterior: float
    declared: bool
    support: float
    contributions: dict[str, float]
    weights: dict[str, float]
    rejected: tuple[str, ...]


class ReliabilityFusion:
    """按帧重新计算后验，避免重复累加相邻帧证据造成虚假高置信度。

    质量加权和分组折扣得到的是 tempered posterior，需另行评估概率校准。
    本模块不提供目标方位与定位，不把目标存在性扩写成定位能力。
    """

    def __init__(self, calibration: dict[str, GaussianCalibration],
                 config: FusionConfig | None = None):
        self.calibration = calibration
        self.config = config or FusionConfig()
        cfg = self.config
        if not 0 < cfg.prior < 1 or not 0 < cfg.declare_off < cfg.declare_on < 1:
            raise ValueError("先验和迟滞阈值无效")
        if cfg.max_age_s <= 0 or cfg.dwell_s < 0 or cfg.llr_clip <= 0:
            raise ValueError("新鲜度、驻留时间与截断阈值无效")
        if not 0 <= cfg.correlation_discount <= 1:
            raise ValueError("相关性折扣必须位于 [0,1]")
        self.declared = False
        self._pending_since: float | None = None
        self._last_timestamp = -np.inf

    def update(self, evidence: list[SensorEvidence], now_s: float) -> FusionResult:
        """融合当前帧；now_s 必须单调不减，重名通道禁止重复计算。"""
        if not np.isfinite(now_s) or now_s < self._last_timestamp:
            raise ValueError("当前时间应有限且单调不减")
        names = [ev.name for ev in evidence]
        if len(names) != len(set(names)):
            raise ValueError("同一帧不能包含重名通道")
        self._last_timestamp = now_s
        weights, raw, groups, rejected = {}, {}, {}, []
        cfg = self.config
        for ev in evidence:
            age = now_s - ev.timestamp_s
            valid = (ev.available and ev.name in self.calibration
                     and np.all(np.isfinite([ev.score, ev.quality, age]))
                     and 0 <= ev.quality <= 1 and 0 <= age < cfg.max_age_s)
            if not valid:
                rejected.append(ev.name)
                continue
            weight = ev.quality * (1.0 - age / cfg.max_age_s)
            weights[ev.name] = weight
            llr = self.calibration[ev.name].log_likelihood_ratio(ev.score)
            raw[ev.name] = weight * float(np.clip(llr, -cfg.llr_clip, cfg.llr_clip))
            group = ev.correlation_group or ev.name
            groups.setdefault(group, []).append(ev.name)
        contributions = {}
        for members in groups.values():
            factor = 1.0 + cfg.correlation_discount * (len(members) - 1)
            for name in members:
                contributions[name] = raw[name] / factor
        posterior = float(expit(logit(cfg.prior) + sum(contributions.values())))
        support = float(sum(weights.values()))
        change = ((not self.declared and posterior >= cfg.declare_on and support > 0)
                  or (self.declared and (posterior <= cfg.declare_off or support == 0)))
        if change:
            if self._pending_since is None:
                self._pending_since = now_s
            if now_s - self._pending_since + 1e-9 >= cfg.dwell_s:
                self.declared = not self.declared
                self._pending_since = None
        else:
            self._pending_since = None
        return FusionResult(posterior, self.declared, support, contributions,
                            weights, tuple(rejected))
