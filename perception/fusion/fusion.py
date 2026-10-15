"""多模态目标探测融合。

将水声、磁异常（以及可扩展的视觉、声呐）的检测证据在**对数似然比**框架下
融合，输出目标存在后验概率与置信度。各模态以校准后的灵敏度参数把检测量映射
为似然比，从而把不同物理量纲的证据统一到同一概率空间。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Evidence:
    """单一模态的检测证据。"""

    name: str
    score: float          # 检测量（如 SNR，越大越可能是目标）
    sensitivity: float    # 灵敏度：score 对对数似然比的换算系数 (1/dB)
    reference: float      # 判定参考值（score 等于该值似然比为 1）


def evidence_log_lr(ev: Evidence) -> float:
    """将检测量映射为对数似然比。"""
    return ev.sensitivity * (ev.score - ev.reference)


class MultiModalFusion:
    """多模态贝叶斯融合器。"""

    def __init__(self, prior: float = 0.1):
        if not 0.0 < prior < 1.0:
            raise ValueError("先验概率必须在 (0,1) 内")
        self.prior = prior

    def fuse(self, evidence: list[Evidence]) -> dict:
        """融合多模态证据，返回目标存在后验概率与判定。

        Returns:
            {"posterior", "log_odds", "per_evidence": {name: log_lr}, "declared"}
        """
        prior_log_odds = np.log(self.prior / (1.0 - self.prior))
        per = {}
        total = prior_log_odds
        for ev in evidence:
            llr = evidence_log_lr(ev)
            per[ev.name] = llr
            total += llr
        posterior = 1.0 / (1.0 + np.exp(-total))
        return {
            "posterior": float(posterior),
            "log_odds": float(total),
            "per_evidence": per,
            "declared": bool(posterior >= 0.5),
        }

    @staticmethod
    def estimate_range(amplitude: float, ref_amplitude: float,
                       ref_range: float) -> float:
        """由磁异常幅值估计目标距离：A ∝ 1 r^3。"""
        amplitude = max(amplitude, 1e-9)
        return ref_range * (ref_amplitude / amplitude) ** (1.0 / 3.0)
