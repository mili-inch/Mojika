import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mojika.conditional.conditioner import Conditioner
from mojika.server.engine import Engine
from mojika.server.interpolation import Interpolation
from mojika.server.schema import Action
from mojika.server.spline import HermitePath


def unit(value):
    return value / value.square().mean().sqrt()


class HermitePathTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(71)
        self.start = torch.randn(1, 1280, generator=generator, dtype=torch.float64)
        self.target = torch.randn(1, 1280, generator=generator, dtype=torch.float64)
        self.velocity = self.target - self.start * 0.3
        self.acceleration = self.start * 0.7 + self.target * 0.2
        self.zero = torch.zeros_like(self.start)

    def test_zero_to_condition_preserves_magnitude_and_endpoint_derivatives(self):
        path = HermitePath.create(self.zero, self.zero, self.zero, self.target, 0, 2)
        position, _, _ = path.sample(1)
        torch.testing.assert_close(position, self.target * 0.5)
        path = HermitePath.create(
            self.start, self.velocity, self.acceleration, self.target, 10, 1.5
        )
        for when, expected in (
            (10, (self.start, self.velocity, self.acceleration)),
            (11.5, (self.target, self.zero, self.zero)),
            (1000, (self.target, self.zero, self.zero)),
        ):
            for actual, value in zip(path.sample(when), expected, strict=True):
                torch.testing.assert_close(actual, value, atol=0, rtol=0)

    def test_analytic_velocity_and_acceleration_match_finite_differences(self):
        path = HermitePath.create(
            self.start * 0.01,
            self.velocity * 0.01,
            self.acceleration * 0.01,
            self.target * 0.012,
            0,
            1.5,
        )
        for now in (0.0001, 0.01, 0.2, 0.75, 1.49):
            epsilon = 1e-6
            _, velocity, acceleration = path.sample(now)
            before, before_velocity, _ = path.sample(now - epsilon)
            after, after_velocity, _ = path.sample(now + epsilon)
            torch.testing.assert_close(
                velocity, (after - before) / (2 * epsilon), atol=2e-9, rtol=1e-6
            )
            torch.testing.assert_close(
                acceleration,
                (after_velocity - before_velocity) / (2 * epsilon),
                atol=2e-9,
                rtol=1e-6,
            )

    def test_midflight_retarget_and_duration_change_are_c2(self):
        path = HermitePath.create(self.start, self.zero, self.zero, self.target, 0, 2)
        for now, target, duration in (
            (0.3, self.start + self.target, 0.4),
            (0.5, self.start, 3),
            (1.2, self.zero, 1),
        ):
            state = path.sample(now)
            self.assertGreater(state[2].abs().sum().item(), 0)
            path = HermitePath.create(*state, target, now, duration)
            for actual, value in zip(path.sample(now), state, strict=True):
                torch.testing.assert_close(actual, value, atol=0, rtol=0)
            epsilon = 1e-7
            after, velocity, acceleration = path.sample(now + epsilon)
            torch.testing.assert_close(
                after, state[0] + state[1] * epsilon, atol=1e-10, rtol=1e-10
            )
            torch.testing.assert_close(
                velocity, state[1] + state[2] * epsilon, atol=1e-9, rtol=1e-9
            )
            torch.testing.assert_close(acceleration, state[2], atol=1e-3, rtol=1e-4)

    def test_clear_reaches_zero_with_zero_velocity_and_acceleration(self):
        path = HermitePath.create(self.start, self.zero, self.zero, self.zero, 0, 2)
        for u in (0, 0.25, 0.5, 0.75, 1):
            position, _, _ = path.sample(2 * u)
            expected = (1 - 10 * u**3 + 15 * u**4 - 6 * u**5) * self.start
            torch.testing.assert_close(position, expected)
        for now in (2 - 1e-7, 2, 2 + 1e-7):
            position, velocity, acceleration = path.sample(now)
            torch.testing.assert_close(position, self.zero, atol=1e-12, rtol=0)
            torch.testing.assert_close(velocity, self.zero, atol=1e-11, rtol=0)
            torch.testing.assert_close(acceleration, self.zero, atol=1e-5, rtol=0)

    def test_float32_endpoint_after_short_to_long_retarget(self):
        short = HermitePath.create(
            self.start.float(),
            self.zero.float(),
            self.zero.float(),
            self.target.float(),
            0,
            0.1,
        )
        state = short.sample(0.02)
        path = HermitePath.create(*state, self.zero.float(), 0.02, 60)
        reference = HermitePath.create(
            *(value.double() for value in state), self.zero, 0.02, 60
        )
        for u in (0.999, 0.999999, 1):
            now = 0.02 + 60 * u
            for actual, expected in zip(
                path.sample(now), reference.sample(now), strict=True
            ):
                torch.testing.assert_close(
                    actual.double(), expected, atol=1e-8, rtol=2e-5
                )

    def test_stationary_short_and_returning_paths_are_finite(self):
        for dtype in (torch.float32, torch.float64):
            for target, velocity, acceleration in (
                (self.start, self.zero, self.zero),
                (self.start * 0.9, self.zero, self.zero),
                (self.start + self.target * 1e-7, self.zero, self.zero),
                (self.start, self.velocity, self.acceleration),
                (-self.start, self.velocity, self.acceleration),
                (self.zero, self.velocity, self.acceleration),
            ):
                path = HermitePath.create(
                    self.start.to(dtype),
                    velocity.to(dtype),
                    acceleration.to(dtype),
                    target.to(dtype),
                    0,
                    1,
                )
                for now in (0, 1e-8, 0.2, 0.5, 0.999999, 1, 100):
                    for value in path.sample(now):
                        self.assertTrue(torch.isfinite(value).all())
                if torch.equal(velocity, self.zero) and torch.equal(target, self.start):
                    position, speed, acceleration = path.sample(0.3)
                    torch.testing.assert_close(position, self.start.to(dtype))
                    torch.testing.assert_close(
                        speed, self.zero.to(dtype), atol=1e-6, rtol=0
                    )
                    torch.testing.assert_close(
                        acceleration, self.zero.to(dtype), atol=1e-6, rtol=0
                    )

    def test_invalid_inputs_are_rejected_explicitly(self):
        for bad in (self.start * float("nan"), self.start * float("inf")):
            with self.assertRaises(ValueError):
                HermitePath.create(self.start, self.zero, self.zero, bad, 0, 1)
            with self.assertRaises(ValueError):
                HermitePath.create(self.start, self.zero, bad, self.target, 0, 1)
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                HermitePath.create(
                    self.start, self.zero, self.zero, self.target, 0, duration
                )
        path = HermitePath.create(self.start, self.zero, self.zero, self.target, 0, 1)
        for now in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                path.sample(now)

    def test_hint_hermite_can_pass_through_zero_without_normalization(self):
        one, zero = torch.ones(1, 16, 16), torch.zeros(1, 16, 16)
        path = HermitePath.create(one, zero, zero, -one, 0, 2)
        value, velocity, acceleration = path.sample(1)
        torch.testing.assert_close(value, zero)
        torch.testing.assert_close(velocity, torch.full_like(one, -1.875))
        torch.testing.assert_close(acceleration, zero)
        value, velocity, acceleration = path.sample(2)
        torch.testing.assert_close(value, -one)
        torch.testing.assert_close(velocity, zero)
        torch.testing.assert_close(acceleration, zero)


class InterpolationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.initial = {
            "gpt": torch.zeros(1, 1280),
            "clip": torch.zeros(1, 512),
            "hint": torch.ones(1, 16, 16),
        }
        self.state = Interpolation(self.initial, torch.device("cpu"), 42, 0)

    def test_noise_moves_automatically_on_start(self):
        self.assertEqual(
            self.state.controls.model_dump(),
            {
                "transition_seconds": 1.0,
                "noise_auto": True,
                "noise_mix": 0.5,
            },
        )
        initial_noise = self.state.values["noise"].clone()
        self.state.advance(0.4)
        self.assertFalse(torch.equal(self.state.values["noise"], initial_noise))
        self.state.advance(10)
        self.assertFalse(torch.equal(self.state.values["noise"], initial_noise))
        for key, value in self.initial.items():
            torch.testing.assert_close(self.state.values[key], value)

    def test_retarget_interval_and_unconditional_restore(self):
        target = unit(torch.randn(1, 1280))
        self.state.update("gpt", target, 0.1)
        position, velocity, acceleration = self.state.condition_paths["gpt"].sample(0.4)
        self.state.update("transition_seconds", 0.5, 0.4)
        torch.testing.assert_close(self.state.values["gpt"], position)
        torch.testing.assert_close(self.state.velocities["gpt"], velocity)
        torch.testing.assert_close(self.state.accelerations["gpt"], acceleration)
        position, velocity, acceleration = self.state.condition_paths["gpt"].sample(0.6)
        self.state.update("clear_conditions", None, 0.6)
        torch.testing.assert_close(self.state.values["gpt"], position)
        torch.testing.assert_close(self.state.velocities["gpt"], velocity)
        torch.testing.assert_close(self.state.accelerations["gpt"], acceleration)
        self.state.advance(1.2)
        self.assertTrue(torch.equal(self.state.values["gpt"], self.initial["gpt"]))
        self.assertTrue(
            torch.equal(self.state.velocities["gpt"], torch.zeros_like(target))
        )

    def test_initial_noise_uses_correlated_gaussian_path(self):
        generator = torch.Generator().manual_seed(42)
        start = torch.randn(1, 1, 256, 256, generator=generator)
        fresh = torch.randn(1, 1, 256, 256, generator=generator)
        rho = 0.75**0.5
        torch.testing.assert_close(self.state.path.start, start)
        torch.testing.assert_close(self.state.path.target, rho * start + 0.5 * fresh)
        self.assertEqual(self.state.path.epoch, 0)
        self.assertEqual(
            self.state.path.moments(0)[[0, 1, 3]][:, [0, 1, 3]].tolist(),
            [[1, 0, rho], [0, 0, 0], [rho, 0, 1]],
        )
        for now in (0.1, 0.4, 0.8, 1.2, 2.1):
            self.state.advance(now)
            value, velocity = self.state.values["noise"], self.state.velocities["noise"]
            samples = torch.stack(
                (
                    value,
                    velocity,
                    self.state.path.acceleration(now),
                    self.state.targets["noise"],
                )
            )
            samples = samples.flatten(start_dim=1).double()
            empirical = samples @ samples.T / value.numel()
            predicted = torch.from_numpy(self.state.path.moments(now))
            torch.testing.assert_close(empirical, predicted, atol=0.025, rtol=0.025)

    def test_clear_all_conditions_is_c2_and_keeps_noise_path_and_controls(self):
        self.state.update("transition_seconds", 2.5, 0.1)
        self.state.update("noise_auto", False, 0.1)
        for key, initial in self.initial.items():
            self.state.update(
                key, -initial if key == "hint" else unit(torch.randn_like(initial)), 0.2
            )
        expected = {
            key: path.sample(0.6) for key, path in self.state.condition_paths.items()
        }
        noise = self.state.path
        controls = self.state.controls
        self.state.update("clear_conditions", None, 0.6)
        self.assertIs(self.state.path, noise)
        self.assertEqual(self.state.controls, controls)
        for key, (value, velocity, acceleration) in expected.items():
            actual, speed, actual_acceleration = self.state.condition_paths[key].sample(
                0.6
            )
            torch.testing.assert_close(actual, value)
            torch.testing.assert_close(speed, velocity)
            torch.testing.assert_close(actual_acceleration, acceleration)
        self.state.advance(3.2)
        for key, initial in self.initial.items():
            torch.testing.assert_close(self.state.values[key], initial)
            torch.testing.assert_close(
                self.state.velocities[key], torch.zeros_like(initial)
            )

    def test_clear_failure_does_not_partially_replace_targets(self):
        self.state.update("gpt", unit(torch.randn(1, 1280)), 0.1)
        self.state.values["clip"] = torch.full_like(self.initial["clip"], float("nan"))
        paths, targets = self.state.condition_paths.copy(), self.state.targets.copy()
        with self.assertRaises(ValueError):
            self.state.update("clear_conditions", None, 0.3)
        self.assertEqual(self.state.condition_paths, paths)
        for key in self.initial:
            self.assertIs(self.state.targets[key], targets[key])

    def test_individual_clear_keeps_other_targets_paths_noise_and_controls(self):
        for key, initial in self.initial.items():
            self.state.update(
                key, -initial if key == "hint" else unit(torch.randn_like(initial)), 0.1
            )
        noise, controls = self.state.path, self.state.controls
        for now, selected in ((0.2, "gpt"), (0.3, "clip"), (0.4, "hint")):
            targets, paths = (
                self.state.targets.copy(),
                self.state.condition_paths.copy(),
            )
            value, velocity, acceleration = paths[selected].sample(now)
            self.state.update("clear_condition", selected, now)
            actual, speed, actual_acceleration = self.state.condition_paths[
                selected
            ].sample(now)
            torch.testing.assert_close(actual, value)
            torch.testing.assert_close(speed, velocity)
            torch.testing.assert_close(actual_acceleration, acceleration)
            torch.testing.assert_close(
                self.state.targets[selected], self.initial[selected]
            )
            for key in self.initial.keys() - {selected}:
                self.assertIs(self.state.targets[key], targets[key])
                self.assertIs(self.state.condition_paths[key], paths[key])
            self.assertIs(self.state.path, noise)
            self.assertEqual(self.state.controls, controls)
        self.state.advance(1.5)
        for key in self.initial:
            torch.testing.assert_close(self.state.values[key], self.initial[key])

    def test_rejected_condition_leaves_target_and_path_intact(self):
        self.state.update("gpt", unit(torch.randn(1, 1280)), 0.1)
        path = self.state.condition_paths["gpt"]
        target = self.state.targets["gpt"]
        with self.assertRaises(ValueError):
            self.state.update("gpt", torch.full_like(target, float("nan")), 0.3)
        self.assertIs(self.state.condition_paths["gpt"], path)
        self.assertIs(self.state.targets["gpt"], target)
        self.state.advance(2)
        torch.testing.assert_close(self.state.values["gpt"], target)

    def test_hint_uses_pixel_path_independently_of_noise(self):
        target = -torch.ones(1, 16, 16)
        self.state.update("hint", target, 0)
        self.state.advance(0.5)
        torch.testing.assert_close(self.state.values["hint"], torch.zeros_like(target))
        hint_path = self.state.condition_paths["hint"]
        self.state.update("noise_auto", False, 0.5)
        self.assertIs(self.state.condition_paths["hint"], hint_path)
        self.state.advance(1.0)
        torch.testing.assert_close(self.state.values["hint"], target)
        self.assertNotIn("hint", self.state.condition_paths)
        self.assertTrue(torch.isfinite(self.state.values["noise"]).all())

    def test_noise_settings_preserve_c1_and_do_not_replan_conditions(self):
        self.state.update("clip", unit(torch.randn(1, 512)), 0.1)
        clip_path = self.state.condition_paths["clip"]
        for kind, value, now in (
            ("noise_auto", False, 0.2),
            ("noise_auto", True, 0.3),
            ("noise_mix", 0.3, 0.35),
            ("noise", None, 0.4),
            ("noise_auto", False, 0.5),
        ):
            position, velocity = self.state.path.sample(now)
            self.state.update(kind, value, now)
            torch.testing.assert_close(self.state.path.start, position)
            torch.testing.assert_close(self.state.path.velocity, velocity)
            self.assertIs(self.state.condition_paths["clip"], clip_path)
        self.state.advance(1.6)
        torch.testing.assert_close(
            self.state.values["clip"], self.state.targets["clip"]
        )
        torch.testing.assert_close(
            self.state.velocities["noise"], torch.zeros_like(self.state.values["noise"])
        )

    def test_manual_update_and_resume_mix_from_current_noise(self):
        self.state.update("noise_auto", False, 0.1)
        self.state.update("noise_mix", 0.6, 0.2)
        fresh = torch.randn_like(self.state.values["noise"])
        for kind, value, now in (("noise", None, 2), ("noise_auto", True, 2.5)):
            current, speed = self.state.path.sample(now)
            with patch.object(self.state, "random_noise", return_value=fresh):
                self.state.update(kind, value, now)
            torch.testing.assert_close(self.state.path.start, current)
            torch.testing.assert_close(self.state.path.velocity, speed)
            torch.testing.assert_close(
                self.state.targets["noise"], current * 0.8 + fresh * 0.6
            )
            self.assertEqual(self.state.path.targets.mix, 0.6)

    def test_mix_change_retargets_automatic_noise_without_position_or_speed_jump(self):
        current, speed = self.state.path.sample(0.3)
        fresh = torch.randn_like(current)
        with patch.object(self.state, "random_noise", return_value=fresh):
            self.state.update("noise_mix", 0.6, 0.3)
        torch.testing.assert_close(self.state.path.start, current)
        torch.testing.assert_close(self.state.path.velocity, speed)
        torch.testing.assert_close(
            self.state.targets["noise"], current * 0.8 + fresh * 0.6
        )
        self.state.update("noise_mix", 0, 0.4)
        stationary = self.state.values["noise"].clone()
        for now in (1.4, 2.1, 100):
            self.state.advance(now)
            torch.testing.assert_close(self.state.values["noise"], stationary)
            torch.testing.assert_close(
                self.state.velocities["noise"], torch.zeros_like(stationary)
            )

    def test_mix_setting_does_not_move_stopped_noise_or_delay_stopping(self):
        self.state.update("noise_auto", False, 0.1)
        path = self.state.path
        self.state.update("noise_mix", 0.4, 0.3)
        self.assertIs(self.state.path, path)
        self.state.advance(2)
        stationary = self.state.values["noise"].clone()
        self.state.update("noise_mix", 0, 2.1)
        self.assertIs(self.state.path, path)
        self.state.update("noise", None, 2.2)
        self.state.advance(3.5)
        torch.testing.assert_close(self.state.values["noise"], stationary)

    def test_rejected_mix_keeps_controls_and_targets(self):
        path, controls, target = (
            self.state.path,
            self.state.controls,
            self.state.targets["noise"],
        )
        for mix in (-0.1, 1.1, float("nan"), float("inf"), "0.5", True):
            with self.assertRaises(ValueError):
                self.state.update("noise_mix", mix, 0)
            self.assertIs(self.state.path, path)
            self.assertEqual(self.state.controls, controls)
            self.assertIs(self.state.targets["noise"], target)

    def test_shared_duration_replans_all_paths_with_continuous_position_velocity_acceleration(
        self,
    ):
        for key, initial in self.initial.items():
            target = -initial if key == "hint" else unit(torch.randn_like(initial))
            self.state.update(key, target, 0.1)
        for now, duration in ((0.3, 0.5), (0.5, 3.0), (0.7, 0.25)):
            expected = {
                key: path.sample(now)
                for key, path in self.state.condition_paths.items()
            }
            expected["noise"] = (
                *self.state.path.sample(now),
                self.state.path.acceleration(now),
            )
            self.state.update("transition_seconds", duration, now)
            self.assertEqual(self.state.path.interval, duration)
            for key, state in expected.items():
                path = (
                    self.state.path
                    if key == "noise"
                    else self.state.condition_paths[key]
                )
                if key != "noise":
                    self.assertEqual(path.duration, duration)
                actual = (
                    (*path.sample(now), path.acceleration(now))
                    if key == "noise"
                    else path.sample(now)
                )
                for result, value in zip(actual, state, strict=True):
                    torch.testing.assert_close(result, value)
        self.state.advance(1)
        for key in self.initial:
            torch.testing.assert_close(self.state.values[key], self.state.targets[key])
        self.assertEqual(self.state.condition_paths, {})

    def test_duration_change_keeps_stopped_noise_at_rest(self):
        self.state.update("noise_auto", False, 0.1)
        self.state.advance(2)
        stopped = self.state.values["noise"].clone()
        self.state.update("transition_seconds", 0.5, 2)
        for now in (2.1, 2.3, 2.5, 100):
            self.state.advance(now)
            torch.testing.assert_close(self.state.values["noise"], stopped)
            torch.testing.assert_close(
                self.state.velocities["noise"], torch.zeros_like(stopped)
            )
        self.assertFalse(self.state.controls.noise_auto)
        self.state.update("noise_auto", True, 100)
        self.state.advance(100.2)
        self.assertFalse(torch.equal(self.state.values["noise"], stopped))

    def test_shared_duration_is_in_public_contract(self):
        action = Action.model_validate(
            {
                "request_id": "duration",
                "operation": {"kind": "transition_seconds", "value": 2.5},
            }
        )
        self.assertEqual(action.operation.value, 2.5)

    def test_quintic_controls_replan_with_continuous_acceleration(self):
        self.state.update("clip", unit(torch.randn(1, 512)), 0.05)
        clip_path = self.state.condition_paths["clip"]
        for kind, value, now in (
            ("noise_mix", 1.0, 0.15),
            ("transition_seconds", 0.3, 0.2),
            ("noise_auto", False, 0.25),
            ("noise", None, 0.3),
            ("noise_auto", True, 0.35),
            ("transition_seconds", 60.0, 0.4),
            ("noise_mix", 0.0, 0.45),
        ):
            position, velocity = self.state.path.sample(now)
            acceleration = self.state.path.acceleration(now)
            self.state.update(kind, value, now)
            actual, speed = self.state.path.sample(now)
            torch.testing.assert_close(actual, position, atol=0, rtol=0)
            torch.testing.assert_close(speed, velocity, atol=0, rtol=0)
            torch.testing.assert_close(
                self.state.path.acceleration(now), acceleration, atol=0, rtol=0
            )
            if kind == "transition_seconds":
                clip_path = self.state.condition_paths["clip"]
            else:
                self.assertIs(self.state.condition_paths["clip"], clip_path)

    def test_repeated_short_long_duration_changes_keep_covariance_symmetric(self):
        rng, now = random.Random(395), 0
        operations = (
            ("transition_seconds", 0.1),
            ("noise_mix", 1.0),
            ("transition_seconds", 60.0),
            ("noise_auto", False),
            ("noise", None),
            ("noise_auto", True),
            ("noise_mix", 0.5),
            ("transition_seconds", 1.0),
        )
        for index in range(300):
            kind, value = operations[index % len(operations)]
            now += rng.uniform(0.05, 0.7)
            self.state.update(kind, value, now)
            covariance = self.state.path.moments(now)
            self.assertTrue((covariance == covariance.T).all())
            self.assertTrue(torch.isfinite(self.state.values["noise"]).all())


class ProjectionTest(unittest.TestCase):
    def test_prepare_once_then_project_matches_training_conditioner(self):
        torch.manual_seed(93)
        conditioner = Conditioner(torch.randn(1280), torch.randn(512), 24, 16, 1)
        for parameter in conditioner.parameters():
            torch.nn.init.normal_(parameter, std=0.03)
        gpt, clip, hint = (
            torch.randn(1, 1280),
            torch.randn(1, 512) * 0.001,
            torch.randn(1, 16, 16),
        )
        x = torch.zeros(1, 1, 256, 256)
        expected = conditioner(x, hint, gpt, clip)
        centered_gpt = conditioner.gpt_embedder.prepare(gpt)
        centered_clip = conditioner.clip_embedder.prepare(clip)
        with (
            patch.object(
                conditioner.gpt_embedder,
                "prepare",
                side_effect=AssertionError("Repeated normalization or centering"),
            ),
            patch.object(
                conditioner.clip_embedder,
                "prepare",
                side_effect=AssertionError("Repeated normalization or centering"),
            ),
        ):
            actual = conditioner.project_centered(hint, centered_gpt, centered_clip)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_engine_interpolates_centered_values_and_returns_to_zero(self):
        torch.manual_seed(19)
        engine = Engine.__new__(Engine)
        engine.conditioner = Conditioner(torch.randn(1280), torch.randn(512), 24, 16, 1)
        for parameter in engine.conditioner.parameters():
            torch.nn.init.normal_(parameter, std=0.03)
        raw_gpt, raw_clip = torch.randn(1, 1280) * 8, torch.randn(1, 512) * 0.001
        engine.encoders = SimpleNamespace(
            gpt=lambda _: raw_gpt,
            clip_text=lambda _: raw_clip,
            clip_image=lambda _: raw_clip,
        )
        c = engine.conditioner
        engine.interpolation = Interpolation(
            {
                "gpt": torch.zeros(1, 1280),
                "clip": torch.zeros(1, 512),
                "hint": c.mask_uncond,
            },
            torch.device("cpu"),
            42,
            0,
        )
        with patch("mojika.server.engine.time.monotonic", return_value=0.1):
            engine.apply("gpt", "mountain")
            engine.apply("clip_text", "mountain")
        torch.testing.assert_close(
            engine.interpolation.targets["gpt"],
            c.gpt_embedder.prepare(raw_gpt),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            engine.interpolation.targets["clip"],
            c.clip_embedder.prepare(raw_clip),
            atol=0,
            rtol=0,
        )
        with (
            patch.object(
                c.gpt_embedder,
                "prepare",
                side_effect=AssertionError("Repeated normalization or centering"),
            ),
            patch.object(
                c.clip_embedder,
                "prepare",
                side_effect=AssertionError("Repeated normalization or centering"),
            ),
        ):
            engine.interpolation.advance(0.4)
            _, halfway = engine.embeddings()
            self.assertGreater(halfway.abs().sum().item(), 0)
            position = engine.interpolation.values["gpt"].clone()
            velocity = engine.interpolation.velocities["gpt"].clone()
            acceleration = engine.interpolation.accelerations["gpt"].clone()
            engine.interpolation.update("clear_conditions", None, 0.4)
            torch.testing.assert_close(engine.interpolation.values["gpt"], position)
            torch.testing.assert_close(engine.interpolation.velocities["gpt"], velocity)
            torch.testing.assert_close(
                engine.interpolation.accelerations["gpt"], acceleration
            )
            engine.interpolation.advance(1.5)
            _, unconditional = engine.embeddings()
            self.assertEqual(torch.count_nonzero(unconditional).item(), 0)
        with patch("mojika.server.engine.time.monotonic", return_value=1.6):
            engine.apply("clip_image", SimpleNamespace(image=object()))
        torch.testing.assert_close(
            engine.interpolation.targets["clip"], c.clip_embedder.prepare(raw_clip)
        )
