"""视觉/融合/任务与导航的统一大脑行为验证。"""

import unittest
from dataclasses import replace

import numpy as np

from autonomy.brain import PerceptionNavigationBrain
from autonomy.mission import MissionConfig, MissionManager, Observation
from autonomy.navigation.grid import GridConfig, OccupancyGrid
from autonomy.navigation.planner import AStarPlanner
from autonomy.navigation.follower import DynamicWindowFollower
from autonomy.navigation.navigator import Navigator
from perception.fusion.reliability import FusionConfig, GaussianCalibration, ReliabilityFusion, SensorEvidence


class BrainTests(unittest.TestCase):
    def brain(self):
        grid = OccupancyGrid(GridConfig(width_cells=40, height_cells=30, resolution_m=0.2))
        grid.set_known_free()
        fusion = ReliabilityFusion({"acoustic": GaussianCalibration(0, 0.8, 2.4, 0.8)},
                                   replace(FusionConfig(), dwell_s=0))
        return PerceptionNavigationBrain(None, fusion,
            MissionManager(replace(MissionConfig(), initial_state="survey")),
            Navigator(grid, AStarPlanner(), DynamicWindowFollower()),
            {"survey": np.array([4, 1]), "return": np.array([1, 1])})

    def test_target_confirmation_pauses_navigation(self):
        brain = self.brain()
        result = brain.step(Observation(0, -0.3, 0.3), np.array([1, 1, 0]), 0, 0, 0.2,
                            evidence=[SensorEvidence("acoustic", 3, 1, 0)])
        self.assertEqual(result.mission.state, "confirm")
        self.assertEqual(result.navigation.status, "paused")
        self.assertEqual(result.navigation.command.speed_m_s, 0)

    def test_background_continues_path_following(self):
        brain = self.brain()
        result = brain.step(Observation(0, -0.3, 0.3), np.array([1, 1, 0]), 0, 0, 0.2,
                            evidence=[SensorEvidence("acoustic", 0, 1, 0)])
        self.assertEqual(result.mission.state, "survey")
        self.assertGreater(result.navigation.command.speed_m_s, 0)
        self.assertEqual(result.mission.forward_speed_m_s, result.navigation.command.speed_m_s)

    def test_path_arrival_finishes_survey(self):
        brain = self.brain()
        result = brain.step(Observation(0, -0.3, 0.3), np.array([4, 1, 0]), 0, 0, 0.2,
                            evidence=[SensorEvidence("acoustic", 0, 1, 0)])
        self.assertEqual(result.mission.state, "exit_water")

    def test_no_fresh_evidence_requests_hold(self):
        brain = self.brain()
        result = brain.step(Observation(1, -0.3, 0.3), np.array([1, 1, 0]), 0, 0, 0.2,
                            evidence=[SensorEvidence("acoustic", 3, 1, 0)])
        self.assertEqual(result.mission.state, "hold")
        self.assertEqual(result.navigation.command.speed_m_s, 0)


if __name__ == "__main__":
    unittest.main()
