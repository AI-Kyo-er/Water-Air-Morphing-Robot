"""验证可达约束、真实几何、故障效能和限速行为。"""

import unittest
from dataclasses import replace

import numpy as np

from control.cm_uav.constrained_allocation import AllocationConfig, ConstrainedAllocator
from control.cm_uav.geometry import body_wrench
from control.cm_uav.params import load_default
from control.cm_uav.simulator import Simulator


class ConstrainedAllocationTests(unittest.TestCase):
    def setUp(self):
        self.params = load_default()
        self.allocator = ConstrainedAllocator(self.params)
        self.previous = np.r_[np.full(4, 3.0), np.full(4, 0.2)]

    def test_rate_and_box_limits_under_unreachable_command(self):
        result = self.allocator.allocate(np.array([100, 8, 90, 20, 10, 5]),
                                         self.previous, self.params.water, 0.02)
        self.assertLessEqual(np.max(abs(result.command[:4] - self.previous[:4])),
                             self.allocator.config.thrust_rate_n_s * 0.02 + 1e-8)
        self.assertLessEqual(np.max(abs(result.command[4:] - self.previous[4:])),
                             self.params.rotor.tilt_rate_limit * 0.02 + 1e-8)
        self.assertGreater(result.normalized_error, 0)
        self.assertTrue(np.all(result.command[:4] >= 0))

    def test_no_lateral_force_or_false_full_rank_claim(self):
        result = self.allocator.allocate(np.array([2, 4, 8, 0, 0, 0]),
                                         self.previous, self.params.water, 0.04)
        self.assertEqual(result.achieved_wrench[1], 0)
        self.assertEqual(result.residual[1], 4)
        self.assertLessEqual(result.local_rank, 5)

    def test_known_health_is_in_achieved_wrench(self):
        health = np.array([0.0, 0.7, 1.0, 1.0])
        result = self.allocator.allocate(np.array([2, 0, 9, 0, 0, 0]),
                                         self.previous, self.params.water, 0.04, health)
        actual = body_wrench(result.command[:4] * health, result.command[4:],
                             self.params.rotor)
        np.testing.assert_allclose(result.achieved_wrench, actual)

    def test_air_recenters_gradually(self):
        result = self.allocator.allocate(np.array([0, 0, 35, 0, 0, 0]),
                                         self.previous, self.params.air, 0.02,
                                         air_fraction=1)
        expected = 0.2 - self.params.rotor.tilt_rate_limit * 0.02
        np.testing.assert_allclose(result.command[4:], expected)

    def test_medium_capacity_change_is_diagnosed(self):
        old = np.r_[np.full(4, 16.0), np.zeros(4)]
        result = self.allocator.allocate(np.zeros(6), old, self.params.air, 0.02)
        self.assertTrue(result.forced_projection)
        self.assertLessEqual(result.command[:4].max(), self.params.rotor.thrust_max_air)

    def test_air_recenter_even_when_old_wrench_was_exact(self):
        target = self.allocator.wrench(self.previous, np.ones(4))
        result = self.allocator.allocate(target, self.previous, self.params.air,
                                         0.01, air_fraction=1)
        self.assertLessEqual(np.max(result.command[4:]),
                             0.2 - self.params.rotor.tilt_rate_limit * 0.01 + 1e-8)

    def test_common_tilt_ablation_keeps_angles_equal(self):
        allocator = ConstrainedAllocator(self.params,
                                       replace(AllocationConfig(), independent_tilt=False))
        result = allocator.allocate(np.array([5, 0, 8, 0, 0, 1]), self.previous,
                                    self.params.water, 0.04)
        self.assertLess(np.ptp(result.command[4:]), 1e-9)

    def test_invalid_inputs(self):
        for dt in (0, -1):
            with self.assertRaises(ValueError):
                self.allocator.allocate(np.zeros(6), self.previous, self.params.air, dt)
        with self.assertRaises(ValueError):
            self.allocator.allocate(np.full(6, np.nan), self.previous, self.params.air, 0.1)

    def test_original_simulator_enforces_previous_command_rate(self):
        sim = Simulator(self.params)
        candidate = self.previous.copy()
        candidate[4:] = 1.0
        limited = sim._limit_tilt_rate(candidate, self.previous, 0.01)
        self.assertLessEqual(np.max(limited[4:] - self.previous[4:]),
                             self.params.rotor.tilt_rate_limit * 0.01 + 1e-8)


if __name__ == "__main__":
    unittest.main()
