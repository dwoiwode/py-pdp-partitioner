import unittest

from src.blackbox_functions.synthetic_functions import Square
from src.sampler.grid_sampler import GridSampler


class TestGridSampler(unittest.TestCase):
    def test_sample_2D_single_point(self):
        f = Square.for_n_dimensions(2)
        cs = f.config_space

        sampler = GridSampler(f, cs)

        sampler.sample(1)

        self.assertEqual(1, len(sampler))
        self.assertEqual(0, len(sampler._grid))

        # ConfigSpace.generate_grid does not yield expected behavior
        # self.assertEqual(0, sampler.config_list[0]['x1'])
        # self.assertEqual(0, sampler.config_list[0]['x2'])

    def test_sample_2D_3x3_grid(self):
        f = Square.for_n_dimensions(2)
        cs = f.config_space

        sampler = GridSampler(f, cs)

        sampler.sample(9)

        self.assertEqual(9, len(sampler))
        self.assertEqual(0, len(sampler._grid))

        expected_points = {
            # 4 Corners
            (-5, -5),
            (-5, 5),
            (5, 5),
            (5, -5),
            # Middle points
            (0, 0),
            (-5, 0),
            (5, 0),
            (0, -5),
            (0, 5),
        }
        resulting_points = {(config["x1"], config["x2"]) for config in sampler.config_list}
        self.assertSetEqual(expected_points, resulting_points)

    def test_sample_3D_big_grid(self):
        f = Square.for_n_dimensions(3)
        cs = f.config_space

        sampler = GridSampler(f, cs)

        sampler.sample(240)  # 7^3 = 343

        self.assertEqual(240, len(sampler))
        self.assertEqual(103, len(sampler._grid))
