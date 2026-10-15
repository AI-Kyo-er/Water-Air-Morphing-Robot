"""验证缺失、过期、相关证据和概率判定的边界情况。"""

import unittest
from dataclasses import replace

from perception.fusion.reliability import (
    FusionConfig, GaussianCalibration, ReliabilityFusion, SensorEvidence,
)


class ReliabilityFusionTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {name: GaussianCalibration(0, 0.8, 2.4, 0.8)
                            for name in ("acoustic", "sonar", "magnetic")}

    def test_missing_and_stale_channels_have_no_evidence(self):
        fusion = ReliabilityFusion(self.calibration)
        result = fusion.update([
            SensorEvidence("acoustic", 20, 1, 0, available=False),
            SensorEvidence("sonar", 20, 1, 0),
        ], 1.0)
        self.assertAlmostEqual(result.posterior, fusion.config.prior)
        self.assertEqual(result.support, 0)
        self.assertEqual(len(result.rejected), 2)

    def test_low_quality_contamination_is_attenuated(self):
        config = replace(FusionConfig(), dwell_s=0)
        high = ReliabilityFusion(self.calibration, config).update([
            SensorEvidence("acoustic", 4, 1, 0)], 0)
        low = ReliabilityFusion(self.calibration, config).update([
            SensorEvidence("acoustic", 4, 0.01, 0)], 0)
        self.assertGreater(high.posterior, low.posterior)
        self.assertFalse(low.declared)

    def test_dwell_prevents_single_frame_declaration(self):
        fusion = ReliabilityFusion(self.calibration)
        for t in (0, 0.1, 0.2):
            result = fusion.update([SensorEvidence("magnetic", 3, 1, t)], t)
            self.assertFalse(result.declared)
        self.assertTrue(fusion.update([SensorEvidence("magnetic", 3, 1, 0.5)], 0.5).declared)

    def test_correlated_channels_are_discounted(self):
        grouped = [SensorEvidence(name, 2, 1, 0, correlation_group="acoustic")
                   for name in ("acoustic", "sonar")]
        cfg = replace(FusionConfig(), dwell_s=0)
        grouped_result = ReliabilityFusion(self.calibration, cfg).update(grouped, 0)
        separate_result = ReliabilityFusion(self.calibration, cfg).update(
            [replace(ev, correlation_group="") for ev in grouped], 0)
        self.assertLess(grouped_result.posterior, separate_result.posterior)

    def test_duplicate_and_future_evidence(self):
        fusion = ReliabilityFusion(self.calibration)
        ev = SensorEvidence("acoustic", 2, 1, 0)
        with self.assertRaises(ValueError):
            fusion.update([ev, ev], 0)
        self.assertEqual(fusion.update([replace(ev, timestamp_s=2)], 1).support, 0)

    def test_probability_is_finite_for_extreme_scores(self):
        fusion = ReliabilityFusion(self.calibration)
        result = fusion.update([SensorEvidence("acoustic", 1e6, 1, 0)], 0)
        self.assertTrue(0 < result.posterior < 1)


if __name__ == "__main__":
    unittest.main()
