import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mojika.server.noise import NoiseSegment, NoiseSpline, mix_noise


class NoiseSplineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.generator = torch.Generator().manual_seed(817)
        self.start = self.normal()
        self.fresh = self.normal()
        self.zero = torch.zeros_like(self.start)

    def normal(self):
        return torch.randn(65536, generator=self.generator, dtype=torch.float64)

    def path(self, mix=0.4, automatic=True, interval=1.25, now=0):
        rho = math.sqrt(1 - mix * mix)
        return NoiseSpline(
            self.start,
            self.zero,
            self.zero,
            mix_noise(self.start, self.fresh, mix),
            now,
            interval,
            automatic,
            42,
            mix,
            [[1, 0, 0, rho], [0, 0, 0, 0], [0, 0, 0, 0], [rho, 0, 0, 1]],
        )

    def assert_gaussian(self, value):
        self.assertAlmostEqual(value.mean().item(), 0, delta=0.018)
        self.assertAlmostEqual(value.square().mean().item(), 1, delta=0.025)
        self.assertAlmostEqual(value.pow(3).mean().item(), 0, delta=0.06)
        self.assertAlmostEqual(value.pow(4).mean().item(), 3, delta=0.13)
        self.assertAlmostEqual(
            (value.abs() > 2).double().mean().item(), 0.0455, delta=0.004
        )
        # Independent spatial coordinates retain the chi-square radial law.
        radii = value.reshape(-1, 4).square().sum(dim=1)
        self.assertAlmostEqual(radii.var().item(), 8, delta=0.4)

    def test_unit_gaussian_marginals_through_correlated_segments(self):
        for mix in (0, 0.01, 0.4, 1):
            path = self.path(mix)
            for now in (0, 0.2, 0.7, 1.25, 1.8, 4.1, 20.3):
                self.assert_gaussian(path.sample(now)[0])

    def test_c1_knots_and_analytic_velocity(self):
        for mix in (0, 0.1, 1):
            path = self.path(mix)
            for index in range(1, 6):
                now = index * path.interval
                left, lv = path.sample(now - 1e-7)
                value, velocity = path.sample(now)
                right, rv = path.sample(now + 1e-7)
                torch.testing.assert_close(value, path.point(index), atol=1e-12, rtol=0)
                torch.testing.assert_close(left, right, atol=2e-6, rtol=0)
                torch.testing.assert_close(lv, rv, atol=4e-6, rtol=0)
                if mix > 0:
                    self.assertGreater(velocity.square().mean().item(), 0.001)
            for now in (0.2, 0.7, 1.8, 4.1):
                _, velocity = path.sample(now)
                before, _ = path.sample(now - 1e-5)
                after, _ = path.sample(now + 1e-5)
                torch.testing.assert_close(
                    velocity, (after - before) / 2e-5, atol=1e-8, rtol=1e-7
                )

    def test_predicted_state_covariance_matches_samples(self):
        path = self.path()
        for now in (0.2, 0.7, 1.25, 1.8, 8.1):
            value, velocity = path.sample(now)
            samples = torch.stack(
                (value, velocity, path.acceleration(now), path.segment.end)
            )
            empirical = (samples @ samples.T / value.numel()).numpy()
            np.testing.assert_allclose(
                path.moments(now), empirical, atol=0.025, rtol=0.025
            )

    def test_analytic_acceleration_matches_velocity_derivative(self):
        for mix in (0, 0.5, 1):
            path = self.path(mix)
            for now in (0.02, 0.2, 0.7, 1.8, 4.1):
                acceleration = path.acceleration(now)
                before = path.sample(now - 1e-5)[1]
                after = path.sample(now + 1e-5)[1]
                torch.testing.assert_close(
                    acceleration, (after - before) / 2e-5, atol=1e-7, rtol=1e-7
                )

    def test_normalization_is_independent_of_realized_values(self):
        path = self.path()
        path.sample(1.6)
        segment = path.segment
        # Scaling a realization with the same supplied law must scale the output;
        # measured RMS normalization would erase this factor.
        from dataclasses import replace

        scaled = replace(
            segment,
            coefficients=tuple(
                tuple(value * 2 for value in direction)
                for direction in segment.coefficients
            ),
            velocity_coefficients=tuple(
                tuple(value * 2 for value in direction)
                for direction in segment.velocity_coefficients
            ),
            acceleration_coefficients=tuple(
                tuple(value * 2 for value in direction)
                for direction in segment.acceleration_coefficients
            ),
        )
        for u in (0.1, 0.5, 0.9):
            for actual, expected in zip(scaled.sample(u), segment.sample(u)):
                torch.testing.assert_close(actual, expected * 2, atol=0, rtol=0)
            torch.testing.assert_close(
                scaled.acceleration(u), segment.acceleration(u) * 2, atol=0, rtol=0
            )
        self.assertIsInstance(segment, NoiseSegment)

    def test_decimal_knots_and_adjacent_floats_keep_position_velocity_and_moments(self):
        for epoch, interval, index in (
            (0, 0.29, 7),
            (100000.123456789, 0.27, 3),
            (1000000.123456789, 0.31, 31),
            (1000000.123456789, 37.91, 1000000),
        ):
            with self.subTest(epoch=epoch, interval=interval, index=index):
                path = self.path(interval=interval, now=epoch)
                knot = epoch + index * interval
                positions, velocities = [], []
                for now in (
                    math.nextafter(knot, -math.inf),
                    knot,
                    math.nextafter(knot, math.inf),
                ):
                    value, velocity = path.sample(now)
                    positions.append(value)
                    velocities.append(velocity)
                    torch.testing.assert_close(
                        value, path.point(index), atol=2e-7, rtol=0
                    )
                    covariance = path.moments(now)
                    self.assertTrue(np.isfinite(covariance).all())
                    self.assertAlmostEqual(covariance[0, 0], 1)
                torch.testing.assert_close(
                    positions[0], positions[-1], atol=2e-7, rtol=0
                )
                torch.testing.assert_close(
                    velocities[0], velocities[-1], atol=2e-7, rtol=0
                )

    def test_replan_at_decimal_knot_then_stop(self):
        epoch, interval = 100000.123456789, 0.27
        path = self.path(interval=interval, now=epoch)
        now = epoch + 3 * interval
        position, velocity = path.sample(now)
        covariance, target = path.moments(now), path.segment.end
        stopped = NoiseSpline(
            position,
            velocity,
            path.acceleration(now),
            target,
            now,
            0.29,
            False,
            17,
            0.4,
            covariance,
        )
        value, speed = stopped.sample(now)
        torch.testing.assert_close(value, position, atol=0, rtol=0)
        torch.testing.assert_close(speed, velocity, atol=0, rtol=0)
        for after in (math.nextafter(now + 0.29, math.inf), now + 3600):
            value, speed = stopped.sample(after)
            torch.testing.assert_close(value, target, atol=1e-8, rtol=0)
            torch.testing.assert_close(speed, self.zero, atol=1e-8, rtol=0)
            np.testing.assert_allclose(
                stopped.moments(after),
                [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
                atol=1e-8,
            )

    def test_replans_keep_joint_covariance_and_unit_marginals(self):
        path = self.path()
        for now, duration, mix, automatic, refresh in (
            (0.3, 0.5, 0.4, True, False),
            (0.6, 2, 0.8, True, True),
            (1.1, 0.3, 0.8, False, False),
            (2, 0.25, 0.1, True, True),
            (2.1, 1, 0, True, True),
        ):
            value, velocity = path.sample(now)
            acceleration = path.acceleration(now)
            covariance = path.moments(now)
            target = path.segment.end
            if refresh:
                rho = math.sqrt(1 - mix * mix)
                target = mix_noise(value, self.normal(), mix)
                covariance[:3, 3] = covariance[3, :3] = rho * covariance[:3, 0]
                covariance[3, 3] = 1
            path = NoiseSpline(
                value,
                velocity,
                acceleration,
                target,
                now,
                duration,
                automatic,
                17,
                mix,
                covariance,
            )
            actual, speed = path.sample(now)
            torch.testing.assert_close(actual, value, atol=0, rtol=0)
            torch.testing.assert_close(speed, velocity, atol=0, rtol=0)
            after, after_speed = path.sample(now + 1e-7)
            torch.testing.assert_close(
                after, value + velocity * 1e-7, atol=1e-10, rtol=1e-10
            )
            torch.testing.assert_close(after_speed, velocity, atol=1e-4, rtol=1e-4)
            self.assert_gaussian(path.sample(now + duration * 0.4)[0])
        stopped, speed = path.sample(100)
        torch.testing.assert_close(stopped, target)
        torch.testing.assert_close(speed, self.zero, atol=1e-12, rtol=0)

    def test_long_gap_is_bounded_and_frame_rate_independent(self):
        dense, sparse = self.path(), self.path()
        for now in np.arange(0, 10, 0.033):
            dense.sample(now)
        for actual, expected in zip(sparse.sample(10.127), dense.sample(10.127)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with patch.object(
            sparse.targets, "normal", wraps=sparse.targets.normal
        ) as normal:
            value, velocity = sparse.sample(86400 * 365 + 0.1)
        self.assertLess(normal.call_count, 160)
        self.assertLess(len(sparse.targets.anchors), 30)
        self.assertTrue(torch.isfinite(velocity).all())
        self.assert_gaussian(value)

    def test_invalid_covariance_and_times_fail_explicitly(self):
        for interval in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                self.path(interval=interval)
        for covariance in (
            np.eye(2),
            np.ones((4, 4)),
            np.diag([2, 0, 0, 1]),
            np.diag([1, 1, 0, 1]),  # Cov(z, a) must equal -Var(v).
            [[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 1]],
            [[1, 0, 0, math.nan], [0, 0, 0, 0], [0, 0, 0, 0], [math.nan, 0, 0, 1]],
        ):
            with self.assertRaises(ValueError):
                NoiseSpline(
                    self.start,
                    self.zero,
                    self.zero,
                    self.fresh,
                    0,
                    1,
                    True,
                    1,
                    0.4,
                    covariance,
                )
        path = self.path()
        for now in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                path.sample(now)
            with self.assertRaises(ValueError):
                path.moments(now)
        for u in (-0.01, 1.01, math.inf, math.nan):
            with self.assertRaises(ValueError):
                path.segment.sample(u)
            with self.assertRaises(ValueError):
                path.segment.moments(u)

    def test_float32_accuracy_with_long_duration_and_nonzero_acceleration(self):
        basis = torch.stack(tuple(self.normal() for _ in range(5)))
        e = np.eye(5)
        endpoints = (
            e[0],
            2 * e[1],
            -4 * e[0] + 3 * e[2],
            e[3],
            2 * e[4],
            -4 * e[3] + 3 * e[2],
        )
        for duration in (0.1, 0.25, 60):
            reference = NoiseSegment.create(
                duration, basis, np.eye(5), endpoints
            )
            actual = NoiseSegment.create(
                duration, basis.float(), np.eye(5), endpoints
            )
            for u in (1e-6, 0.01, 0.5 - 1e-8, 0.5, 0.5 + 1e-8, 0.99, 0.9999, 1 - 1e-6):
                for observed, expected in zip(actual.sample(u), reference.sample(u)):
                    torch.testing.assert_close(
                        observed.double(), expected, atol=1e-4, rtol=1e-5
                    )
                torch.testing.assert_close(
                    actual.acceleration(u).double(),
                    reference.acceleration(u),
                    # Acceleration roundoff scales with the inverse duration squared.
                    atol=0.002 * max(1, (0.25 / duration) ** 2),
                    rtol=1e-5,
                )

    def test_c2_knots_share_acceleration_and_state_covariance(self):
        for mix in (0, 0.01, 0.5, 1):
            for interval in (0.1, 0.25, 1.25, 60):
                path = self.path(mix, interval=interval)
                for index in range(1, 5):
                    left, right = path._segment(index - 1), path._segment(index)
                    torch.testing.assert_close(
                        left.acceleration(1),
                        right.acceleration(0),
                        atol=1e-11,
                        rtol=1e-11,
                    )
                    np.testing.assert_allclose(
                        left.moments(1)[:3, :3],
                        right.moments(0)[:3, :3],
                        atol=1e-9,
                        rtol=1e-10,
                    )
                    # Approach from both sides, including the normalization derivatives.
                    torch.testing.assert_close(
                        left.acceleration(1 - 1e-8),
                        right.acceleration(1e-8),
                        # At fixed progress offsets, acceleration differences scale as 1/T².
                        atol=1e-4 * max(1, (0.25 / interval) ** 2),
                        rtol=1e-5,
                    )

    def test_replanning_preserves_full_normalized_state_and_covariance(self):
        path = self.path()
        for now, duration, automatic in (
            (0.2, 0.25, True),
            (0.4, 60, True),
            (0.5, 0.27, False),
        ):
            position, velocity = path.sample(now)
            acceleration, covariance = path.acceleration(now), path.moments(now)
            target = path.segment.end
            path = NoiseSpline(
                position,
                velocity,
                acceleration,
                target,
                now,
                duration,
                automatic,
                42,
                0.4,
                covariance,
            )
            torch.testing.assert_close(
                path.acceleration(now), acceleration, atol=0, rtol=0
            )
            np.testing.assert_allclose(
                path.moments(now), covariance, atol=1e-10, rtol=1e-10
            )
            epsilon = 1e-7
            torch.testing.assert_close(
                path.sample(now + epsilon)[1],
                velocity + epsilon * acceleration,
                atol=1e-10,
                rtol=1e-9,
            )
            torch.testing.assert_close(
                path.acceleration(now + epsilon), acceleration, atol=1e-3, rtol=1e-5
            )
            self.assert_gaussian(path.sample(now + duration * 0.3)[0])
        torch.testing.assert_close(
            path.acceleration(now + duration - 1e-8), self.zero, atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            path.acceleration(now + duration), self.zero, atol=0, rtol=0
        )
        torch.testing.assert_close(
            path.acceleration(now + duration + 1), self.zero, atol=0, rtol=0
        )
