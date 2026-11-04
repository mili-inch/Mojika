import math
import unittest
from unittest.mock import patch

import torch

from mojika.server.noise import NoiseTargets, mix_noise


class NoiseTargetsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(42)
        self.source = torch.randn(65536, generator=generator, dtype=torch.float64)
        self.fresh = torch.randn(65536, generator=generator, dtype=torch.float64)

    def test_mixing_preserves_variance_and_controls_distance(self):
        previous_distance = -1
        for mix in (0, 0.05, 0.3, 0.7, 1):
            point = mix_noise(self.source, self.fresh, mix)
            self.assertLess(abs(point.mean().item()), 0.02)
            self.assertLess(abs(point.var().item() - 1), 0.02)
            correlation = (point * self.source).mean().item()
            self.assertAlmostEqual(correlation, math.sqrt(1 - mix * mix), delta=0.02)
            distance = (point - self.source).square().mean().item()
            self.assertGreater(distance, previous_distance)
            previous_distance = distance
        self.assertTrue(torch.equal(mix_noise(self.source, self.fresh, 0), self.source))
        self.assertTrue(torch.equal(mix_noise(self.source, self.fresh, 1), self.fresh))

    def test_bridge_innovations_have_unit_variance_and_are_independent(self):
        for mix in (0.05, 0.4, 0.9):
            targets = NoiseTargets(self.source, 9, mix)
            previous, innovation = self.source, self.fresh
            correlation = math.sqrt(1 - mix * mix)
            for index in range(1, 18):
                point = targets.point(index)
                fresh = (point - correlation * previous) / mix
                self.assertAlmostEqual(fresh.var().item(), 1, delta=0.025)
                self.assertAlmostEqual(fresh.mean().item(), 0, delta=0.02)
                self.assertAlmostEqual((fresh * previous).mean().item(), 0, delta=0.02)
                self.assertAlmostEqual(
                    (fresh * innovation).mean().item(), 0, delta=0.02
                )
                self.assertAlmostEqual(point.var().item(), 1, delta=0.025)
                self.assertAlmostEqual(
                    (point * self.source).mean().item(), correlation**index, delta=0.02
                )
                previous, innovation = point, fresh

    def test_points_are_deterministic_for_any_access_order(self):
        for mix in (0, 0.2, 1):
            dense = NoiseTargets(self.source, 17, mix)
            expected = {i: dense.point(i) for i in range(34)}
            sparse = NoiseTargets(self.source, 17, mix)
            for i in (33, 1, 20, 3, 19, 2, 0, 4, 31, 17, 7):
                self.assertTrue(torch.equal(sparse.point(i), expected[i]))
            self.assertLessEqual(len(sparse.recent), 4)

    def test_zero_mix_keeps_targets_and_unit_mix_draws_independent_noise(self):
        frozen = NoiseTargets(self.source, 17, 0)
        with patch.object(frozen, "normal", side_effect=AssertionError("Unneeded RNG")):
            for index in (0, 1, 10, 100000):
                self.assertTrue(torch.equal(frozen.point(index), self.source))
        independent = NoiseTargets(self.source, 17, 1)
        point = independent.point(1)
        self.assertTrue(torch.equal(point, independent.normal(2)))
        self.assertAlmostEqual((point * self.source).mean().item(), 0, delta=0.02)

    def test_invalid_mix_is_rejected(self):
        for mix in (-0.01, 1.01, math.nan, math.inf):
            with self.assertRaises(ValueError):
                mix_noise(self.source, self.fresh, mix)
            with self.assertRaises(ValueError):
                NoiseTargets(self.source, 42, mix)
