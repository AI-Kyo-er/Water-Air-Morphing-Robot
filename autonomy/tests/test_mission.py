"""任务主路径、保护优先级和状态恢复的正式验证。"""

import unittest
from dataclasses import replace

import numpy as np

from autonomy.mission import MissionConfig, MissionManager, MissionState, Observation
from autonomy.runtime import LayeredRuntime
from control.cm_uav.constrained_allocation import ConstrainedAllocator
from control.cm_uav.params import load_default
from perception.fusion.reliability import (
    FusionConfig, GaussianCalibration, ReliabilityFusion, SensorEvidence,
)


class MissionTests(unittest.TestCase):
    def observation(self, t=0.0, **kwargs):
        return Observation(timestamp_s=t, z_m=-0.3, depth_m=0.3,
                           sensor_support=2, **kwargs)

    def test_complete_route(self):
        manager = MissionManager()
        route = [
            Observation(0, 0, 0, flight_ready=True),
            Observation(1, 2, 0, at_search_area=True),
            self.observation(2), self.observation(3, target_declared=True),
            self.observation(3.1, target_declared=True),
            self.observation(4, target_declared=True),
            Observation(5, 0.1, 0), Observation(6, 2, 0, at_home=True),
        ]
        states = [manager.update(obs).state for obs in route]
        self.assertIn("confirm", states)
        self.assertEqual(states[-1], "complete")

    def test_failure_precedes_target_confirmation(self):
        manager = MissionManager(replace(MissionConfig(), initial_state="survey"))
        command = manager.update(self.observation(0, actuator_health=0.3,
                                                  target_declared=True))
        self.assertEqual(command.state, "recovery")
        self.assertEqual(command.forward_speed_m_s, 0)

    def test_lost_link_recovery_remains_latched(self):
        manager = MissionManager(replace(MissionConfig(), initial_state="survey"))
        self.assertEqual(manager.update(self.observation(0, link_age_s=2)).state, "recovery")
        self.assertEqual(manager.update(self.observation(1)).state, "recovery")
        self.assertEqual(manager.update(Observation(2, 0.1, 0)).state, "abort")

    def test_hold_requires_continuous_recovery(self):
        manager = MissionManager(replace(MissionConfig(), initial_state="survey"))
        self.assertEqual(manager.update(self.observation(0, obstacle_distance_m=0.2)).state,
                         "hold")
        self.assertEqual(manager.update(self.observation(0.5)).state, "hold")
        self.assertEqual(manager.update(self.observation(1.6)).state, "survey")

    def test_low_battery_exits_underwater_task(self):
        manager = MissionManager(replace(MissionConfig(), initial_state="survey"))
        self.assertEqual(manager.update(self.observation(0, battery_fraction=0.1)).state,
                         "exit_water")

    def test_invalid_observation_does_not_enter_task(self):
        manager = MissionManager()
        with self.assertRaises(ValueError):
            manager.update(Observation(0, float("nan"), 0))
        self.assertEqual(manager.state, MissionState.PREFLIGHT)

    def test_runtime_uses_observed_medium_and_fusion(self):
        calibration = {"acoustic": GaussianCalibration(0, 0.8, 2.4, 0.8)}
        fusion = ReliabilityFusion(calibration, replace(FusionConfig(), dwell_s=0))
        manager = MissionManager(replace(MissionConfig(), initial_state="survey"))
        runtime = LayeredRuntime(fusion, manager, ConstrainedAllocator(load_default()))
        result = runtime.step(self.observation(0), [SensorEvidence("acoustic", 3, 1, 0)],
                              np.array([2, 0, 4, 0, 0, 0]), np.r_[np.ones(4), np.zeros(4)],
                              0.04)
        self.assertEqual(result.brain.state, "confirm")
        self.assertEqual(runtime.medium_name, "water")
        self.assertIsNotNone(result.allocation)

    def test_new_brain_state_changes_same_cycle_wrench(self):
        calibration = {"acoustic": GaussianCalibration(0, 0.8, 2.4, 0.8)}
        fusion = ReliabilityFusion(calibration, replace(FusionConfig(), dwell_s=0))
        manager = MissionManager(replace(MissionConfig(), initial_state="survey"))
        runtime = LayeredRuntime(fusion, manager, ConstrainedAllocator(load_default()))
        captured = []

        def small_brain(command, observation):
            captured.append(command)
            return np.array([command.forward_speed_m_s * 8, 0, 4, 0, 0, 0])

        result = runtime.step(self.observation(0), [SensorEvidence("acoustic", 3, 1, 0)],
                              small_brain, np.r_[np.ones(4), np.zeros(4)], 0.04)
        self.assertEqual(captured[0].state, "confirm")
        self.assertEqual(captured[0].forward_speed_m_s, 0)
        target = result.allocation.achieved_wrench + result.allocation.residual
        self.assertAlmostEqual(target[0], 0)


if __name__ == "__main__":
    unittest.main()
