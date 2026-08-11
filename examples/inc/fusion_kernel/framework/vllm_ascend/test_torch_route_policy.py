import unittest

from inc_moe_runtime import select_route_lanes, use_parallel_route


class RoutePolicyTests(unittest.TestCase):
    def test_topk_one_stays_scalar(self):
        self.assertFalse(use_parallel_route(1 << 20, 1, 48))

    def test_small_launch_bound_work_stays_scalar(self):
        self.assertFalse(use_parallel_route(32, 8, 48))
        self.assertFalse(use_parallel_route(512, 2, 48))

    def test_protocol_work_selects_parallel(self):
        self.assertTrue(use_parallel_route(128, 8, 48))
        self.assertTrue(use_parallel_route(512, 3, 48))
        self.assertTrue(use_parallel_route(2048, 2, 48))

    def test_lane_count_is_bounded_by_work_and_histogram(self):
        self.assertEqual(select_route_lanes(128, 128, 48), 8)
        self.assertEqual(select_route_lanes(8192, 128, 48), 16)
        self.assertEqual(select_route_lanes(8192, 64, 48), 32)
        self.assertEqual(select_route_lanes(32, 64, 48), 2)
        self.assertEqual(select_route_lanes(0, 64, 48), 0)


if __name__ == "__main__":
    unittest.main()
