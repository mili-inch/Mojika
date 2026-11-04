import math
import unittest
from dataclasses import asdict

import torch

from mojika.utils.timestep_distribution import (
    TimeSamplingParameters,
    sample_times,
    time_cdf,
    time_density,
)


class TimestepDistributionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.parameters = TimeSamplingParameters(-0.8, 0.8, 0.15, 0.25)

    def test_zero_endpoint_densities_follow_logit_normal_cdf(self):
        z = torch.linspace(-4, 4, 1001, dtype=torch.float64)
        for location, scale in ((-0.8, 0.8), (0.0, 1.0), (0.5, 0.6)):
            params = TimeSamplingParameters(location, scale, 0.0, 0.0)
            times = torch.sigmoid(location + scale * z)
            self.assertTrue((times.diff() > 0).all())
            actual = time_cdf(times, **asdict(params))
            torch.testing.assert_close(actual, 0.5 * torch.erfc(-z / math.sqrt(2)))
        endpoints = torch.tensor([0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            time_cdf(endpoints, **asdict(self.parameters)), endpoints
        )

    def test_generated_times_match_cdf_and_restore_generator_state(self):
        generator = torch.Generator().manual_seed(123)
        state = generator.get_state()
        sample = sample_times(
            self.parameters, (200000,), device="cpu", generator=generator
        )
        self.assertEqual(sample.dtype, torch.float32)
        self.assertTrue(torch.isfinite(sample).all())
        self.assertTrue(((sample >= 0) & (sample <= 1)).all())
        thresholds = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], dtype=torch.float64)
        probabilities = time_cdf(thresholds, **asdict(self.parameters))
        for threshold, probability in zip(thresholds, probabilities, strict=True):
            observed = (sample < threshold).float().mean()
            tolerance = (
                6 * math.sqrt(probability * (1 - probability) / sample.numel()) + 1e-5
            )
            self.assertAlmostEqual(observed.item(), probability.item(), delta=tolerance)
        generator.set_state(state)
        repeated = sample_times(
            self.parameters, (200000,), device="cpu", generator=generator
        )
        torch.testing.assert_close(sample, repeated, atol=0, rtol=0)

    def test_invalid_parameters_fail_before_sampling(self):
        valid = asdict(self.parameters)
        for name in valid:
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        TimeSamplingParameters(**(valid | {name: value}))
        for changes in (
            {"scale": 0.0},
            {"scale": -1.0},
            {"noise_density": -0.1},
            {"image_density": -0.1},
            {"noise_density": 1.1, "image_density": 1.0},
        ):
            with self.assertRaises(ValueError):
                TimeSamplingParameters(**(valid | changes))

    def test_endpoint_density_values_floor_and_normalization(self):
        t = torch.linspace(0, 1, 100001, dtype=torch.float64)
        params = asdict(self.parameters)
        density = time_density(t, **params)
        self.assertEqual(density[0].item(), self.parameters.noise_density)
        self.assertEqual(density[-1].item(), self.parameters.image_density)
        floor = (
            self.parameters.noise_density * (1 - t) + self.parameters.image_density * t
        )
        self.assertTrue((density >= floor).all())
        self.assertAlmostEqual(torch.trapezoid(density, t).item(), 1.0, delta=1e-6)
        interior = t[1:-1:100].clone().requires_grad_()
        derivative = torch.autograd.grad(time_cdf(interior, **params).sum(), interior)[
            0
        ]
        torch.testing.assert_close(derivative, time_density(interior, **params))

    def test_equal_endpoint_densities_reproduce_target_mixture(self):
        t = torch.linspace(0, 1, 10001, dtype=torch.float64)
        params = TimeSamplingParameters(-0.8, 0.8, 0.2, 0.2)
        torch.testing.assert_close(
            time_cdf(t, **asdict(params)),
            0.2 * t + 0.8 * torch.special.ndtr((torch.logit(t) + 0.8) / 0.8),
            rtol=0,
            atol=3e-16,
        )

    def test_boundary_component_weights_sample_triangles_and_uniform(self):
        t = torch.linspace(0, 1, 101, dtype=torch.float64)
        for noise_density, image_density, expected_cdf, mean, variance in (
            (2.0, 0.0, 2 * t - t.square(), 1 / 3, 1 / 18),
            (0.0, 2.0, t.square(), 2 / 3, 1 / 18),
            (1.0, 1.0, t, 0.5, 1 / 12),
        ):
            params = TimeSamplingParameters(-0.8, 0.8, noise_density, image_density)
            torch.testing.assert_close(time_cdf(t, **asdict(params)), expected_cdf)
            samples = sample_times(
                params,
                (200000,),
                device="cpu",
                generator=torch.Generator().manual_seed(93),
            )
            self.assertAlmostEqual(samples.mean().item(), mean, delta=0.002)
            self.assertAlmostEqual(samples.var().item(), variance, delta=0.002)





if __name__ == "__main__":
    unittest.main()
