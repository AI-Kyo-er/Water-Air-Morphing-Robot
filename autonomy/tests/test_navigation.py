"""机体余量、不可达、切角、扫描与局部导航的正式测试。"""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from autonomy.navigation.grid import GridConfig, OccupancyGrid
from autonomy.navigation.planner import AStarPlanner
from autonomy.navigation.follower import DynamicWindowFollower, FollowerConfig
from autonomy.navigation.navigator import Navigator, NavigatorConfig


class NavigationTests(unittest.TestCase):
    def grid(self, radius=0.1):
        grid = OccupancyGrid(GridConfig(width_cells=30, height_cells=25,
                                       resolution_m=0.2, robot_radius_m=radius))
        grid.set_known_free()
        return grid

    def test_world_cell_roundtrip(self):
        grid = self.grid()
        cell = (8, 5)
        self.assertEqual(grid.world_to_cell(grid.cell_to_world(cell)), cell)

    def test_unknown_is_blocked(self):
        grid = OccupancyGrid()
        self.assertFalse(grid.is_free((1, 1)))

    def test_footprint_inflation(self):
        grid = self.grid(radius=0.4)
        grid.set_rectangle((2, 1, 2.4, 3))
        self.assertFalse(grid.is_free((1.7, 2)))
        self.assertTrue(grid.is_free((1.1, 2)))

    def test_obstacle_route_and_continuous_simplification(self):
        grid = self.grid()
        grid.set_rectangle((2, 0, 2.6, 3.2))
        result = AStarPlanner().plan(grid, np.array([0.7, 0.7]), np.array([4.5, 1]))
        self.assertTrue(result.success)
        self.assertGreater(result.expanded, 0)
        for a, b in zip(result.points_xy_m[:-1], result.points_xy_m[1:]):
            self.assertTrue(grid.segment_is_free(a, b))

    def test_full_wall_is_unreachable(self):
        grid = self.grid()
        grid.set_rectangle((2, 0, 2.6, 5))
        result = AStarPlanner().plan(grid, np.array([0.7, 0.7]), np.array([4.5, 1]))
        self.assertFalse(result.success)
        self.assertEqual(len(result.points_xy_m), 0)

    def test_cost_aware_shortcut_preserves_clearance_objective(self):
        grid = self.grid()
        grid.set_rectangle((2, 0.8, 2.4, 1.2))
        planner = AStarPlanner()
        result = planner.plan(grid, np.array([0.7, 0.7]), np.array([4.5, 1.5]))
        self.assertTrue(result.success)
        for a, b in zip(result.points_xy_m[:-1], result.points_xy_m[1:]):
            self.assertTrue(grid.segment_is_free(a, b))

    def test_no_diagonal_corner_cutting(self):
        grid = OccupancyGrid(GridConfig(width_cells=5, height_cells=5, resolution_m=1,
                                       robot_radius_m=0))
        grid.set_known_free()
        grid.set_rectangle((2, 1, 3, 2))
        grid.set_rectangle((1, 2, 2, 3))
        self.assertFalse(grid.segment_is_free(np.array([1.5, 1.5]), np.array([2.5, 2.5])))

    def test_scan_marks_hit_free_space_and_rejects_old_time(self):
        grid = OccupancyGrid(GridConfig(width_cells=20, height_cells=20,
                                       resolution_m=0.2, robot_radius_m=0))
        pose = np.array([1, 1, 0])
        grid.update_scan(pose, np.array([0]), np.array([1.0]), 2, 1)
        self.assertTrue(grid.occupied()[grid.world_to_cell((2, 1))])
        self.assertFalse(grid.occupied()[grid.world_to_cell((1.4, 1))])
        with self.assertRaises(ValueError):
            grid.update_scan(pose, np.array([0]), np.array([1]), 2, 0)

    def test_local_speed_window(self):
        grid = self.grid()
        command = DynamicWindowFollower().command(grid, np.array([1, 1, 0]),
                                                  np.array([2, 1]), 0, 0, 0.2)
        self.assertFalse(command.blocked)
        self.assertLessEqual(command.speed_m_s, 0.3*0.2+1e-9)
        self.assertLessEqual(abs(command.yaw_rate_rad_s), 1.8*0.2+1e-9)

    def test_obstacle_inside_rollout_is_rejected(self):
        grid = self.grid()
        grid.set_rectangle((1.4, 0.4, 2, 1.6))
        command = DynamicWindowFollower().command(grid, np.array([1, 1, 0]),
                                                  np.array([2, 1]), 0.35, 0, 0.2)
        for point in command.trajectory:
            self.assertTrue(grid.is_free(point[:2]))

    def test_map_change_triggers_replan(self):
        grid = self.grid()
        navigator = Navigator(grid, AStarPlanner(), DynamicWindowFollower())
        navigator.set_goal(np.array([4, 1]))
        first = navigator.step(np.array([1, 1, 0]), 0, 0, 0, 0.2)
        grid.set_rectangle((2, 0.7, 2.4, 1.3))
        second = navigator.step(np.array([1, 1, 0]), 0, 0, 0.2, 0.2)
        self.assertTrue(first.replanned and second.replanned)
        self.assertEqual(navigator.replan_count, 2)

    def test_task_pause_stops_navigation(self):
        navigator = Navigator(self.grid(), AStarPlanner(), DynamicWindowFollower())
        navigator.set_goal(np.array([4, 1]))
        result = navigator.step(np.array([1, 1, 0]), 0.2, 0, 0, 0.2, enabled=False)
        self.assertEqual(result.command.speed_m_s, 0)
        self.assertEqual(result.status, "paused")

    def test_local_target_does_not_cut_invisible_corner(self):
        navigator = Navigator.from_yaml(Path(__file__).resolve().parents[1]/"configs/navigation.yaml")
        navigator.grid.set_known_free()
        for rectangle in ((3, 0, 3.8, 5.6), (6, 3.4, 6.8, 9), (4.3, 5.8, 4.8, 6.5)):
            navigator.grid.set_rectangle(rectangle)
        navigator.set_goal(np.array([10.5, 7.4]))
        pose = np.array([4.038397, 6.520559, 0.540464])
        result = navigator.step(pose, 0, 0, 0, 0.2)
        self.assertTrue(navigator.grid.segment_is_free(pose[:2], result.local_target_xy_m))

    def test_short_route_reaches_goal(self):
        navigator = Navigator(self.grid(), AStarPlanner(), DynamicWindowFollower())
        goal = np.array([2.5, 1.5])
        navigator.set_goal(goal)
        pose, speed, rate = np.array([0.7, 0.7, 0]), 0.0, 0.0
        for index in range(200):
            result = navigator.step(pose, speed, rate, index*0.2, 0.2)
            if result.status == "arrived":
                break
            speed, rate = result.command.speed_m_s, result.command.yaw_rate_rad_s
            pose[:2] += speed*0.2*np.array([np.cos(pose[2]+0.1*rate), np.sin(pose[2]+0.1*rate)])
            pose[2] += rate*0.2
        self.assertEqual(result.status, "arrived")


if __name__ == "__main__":
    unittest.main()
