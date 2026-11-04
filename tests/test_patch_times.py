import unittest
from dataclasses import asdict

import torch

from mojika.conditional.config import ExperimentConfig
from mojika.conditional.flow import (
    euler_sample,
    interpolate_patches,
    sample_patch_times,
)
from mojika.conditional.model import DiffusionTransformer
from mojika.modules.timestep_parameter import DiMRTimestepParameter
from mojika.utils.timestep_distribution import time_cdf
from tests.fixtures import small_config


class PatchTimeSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.time_sampling = small_config("adamw").training.time_sampling

    def test_mixed_region_sizes_preserve_square_time_sharing(self):
        sizes = (16, 32, 64, 128, 256)
        times, selected, offsets = sample_patch_times(
            128, 256, 16, sizes, "cpu", time_sampling=self.time_sampling
        )
        self.assertEqual(set(selected.tolist()), set(sizes))
        self.assertEqual(times.shape, (128, 16, 16))
        self.assertEqual(offsets.shape, (128, 2))
        self.assertTrue((offsets % 16 == 0).all())
        for time_map, size, offset in zip(
            times, selected.tolist(), offsets.tolist(), strict=True
        ):
            span = size // 16
            for start_y in range(-offset[0] // 16, 16, span):
                for start_x in range(-offset[1] // 16, 16, span):
                    region = time_map[
                        max(0, start_y) : min(16, start_y + span),
                        max(0, start_x) : min(16, start_x + span),
                    ]
                    self.assertTrue((region == region[0, 0]).all())
            if size < 256:
                self.assertGreater(time_map.std().item(), 0)

    def test_every_region_size_samples_the_entire_time_interval(self):
        for size in (16, 32, 64, 128, 256):
            with self.subTest(size=size):
                times, _, _ = sample_patch_times(
                    4096, 256, 16, (size,), "cpu", time_sampling=self.time_sampling
                )
                # One region per image: independent draws regardless of region size.
                values = times[:, 0, 0]
                self.assertTrue(((values >= 0) & (values <= 1)).all())
                self.assertLess(values.min().item(), 0.01)
                self.assertGreater(values.max().item(), 0.99)
                thresholds = torch.tensor([0.1, 0.5, 0.9])
                observed = (values[:, None] <= thresholds).float().mean(0)
                expected = time_cdf(thresholds, **asdict(self.time_sampling))
                torch.testing.assert_close(observed, expected, rtol=0, atol=0.025)

    def test_offsets_are_independent_uniform_and_redrawn_for_each_image(self):
        _, _, offsets = sample_patch_times(
            16384, 256, 16, (64,), "cpu", time_sampling=self.time_sampling
        )
        self.assertEqual(set(offsets[:, 0].tolist()), {0, 16, 32, 48})
        self.assertEqual(set(offsets[:, 1].tolist()), {0, 16, 32, 48})
        pairs = offsets[:, 0] // 16 * 4 + offsets[:, 1] // 16
        frequency = torch.bincount(pairs, minlength=16).float() / len(pairs)
        torch.testing.assert_close(
            frequency, torch.full((16,), 1 / 16), rtol=0, atol=0.01
        )
        _, _, next_offsets = sample_patch_times(
            16384, 256, 16, (64,), "cpu", time_sampling=self.time_sampling
        )
        self.assertFalse(torch.equal(offsets, next_offsets))

    def test_adjacent_time_sharing_probability_does_not_depend_on_position(self):
        for size in (32, 64, 128):
            with self.subTest(size=size):
                times, _, _ = sample_patch_times(
                    8192, 256, 16, (size,), "cpu", time_sampling=self.time_sampling
                )
                horizontal = (times[:, :, 1:] == times[:, :, :-1]).float().mean((0, 1))
                vertical = (times[:, 1:, :] == times[:, :-1, :]).float().mean((0, 2))
                expected = torch.full((15,), 1 - 16 / size)
                torch.testing.assert_close(horizontal, expected, rtol=0, atol=0.02)
                torch.testing.assert_close(vertical, expected, rtol=0, atol=0.02)

    def test_clipped_border_regions_do_not_wrap_to_the_opposite_edge(self):
        times, _, offsets = sample_patch_times(
            128, 256, 16, (64,), "cpu", time_sampling=self.time_sampling
        )
        for time_map, (dy, dx) in zip(times, offsets.tolist(), strict=True):
            if dy:
                self.assertFalse(torch.equal(time_map[0, :], time_map[-1, :]))
            if dx:
                self.assertFalse(torch.equal(time_map[:, 0], time_map[:, -1]))

    def test_single_patch_and_full_image_regions_need_no_offset(self):
        for size in (16, 256):
            with self.subTest(size=size):
                times, _, offsets = sample_patch_times(
                    128, 256, 16, (size,), "cpu", time_sampling=self.time_sampling
                )
                self.assertTrue((offsets == 0).all())
                if size == 256:
                    self.assertEqual(times.shape, (128, 1, 1))

    def test_restoring_rng_reproduces_sizes_offsets_and_times(self):
        for sizes in ((256,), (16, 32, 64, 128, 256)):
            with self.subTest(sizes=sizes):
                rng = torch.get_rng_state()
                expected = sample_patch_times(
                    128, 256, 16, sizes, "cpu", time_sampling=self.time_sampling
                )
                torch.set_rng_state(rng)
                actual = sample_patch_times(
                    128, 256, 16, sizes, "cpu", time_sampling=self.time_sampling
                )
                for a, b in zip(expected, actual, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_region_sizes_are_rejected_by_configuration(self):
        for sizes in ([], [16, 16], [8], [48], [512], [16.0]):
            with self.subTest(sizes=sizes):
                config = small_config("adamw").to_dict()
                config["training"]["time_patch_sizes"] = sizes
                with self.assertRaisesRegex(ValueError, "[Tt]ime.*patch"):
                    ExperimentConfig.from_dict(config)

    def test_interpolation_obeys_spatial_order_and_time_direction(self):
        clean = torch.ones(1, 1, 32, 32)
        noise = -torch.ones_like(clean)
        times = torch.tensor([[[0.0, 0.25], [0.75, 1.0]]])
        mixed = interpolate_patches(clean, noise, times, 16)
        for row, column, value in ((0, 0, -1), (0, 1, -0.5), (1, 0, 0.5), (1, 1, 1)):
            region = mixed[
                :, :, row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16
            ]
            torch.testing.assert_close(
                region, torch.full_like(region, value), rtol=0, atol=0
            )

    def test_flow_target_is_the_velocity_when_all_patch_times_advance(self):
        clean = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(clean)
        times = torch.rand(2, 2, 2) / 2
        x = interpolate_patches(clean, noise, times, 16)
        advanced = interpolate_patches(clean, noise, times + 0.125, 16)
        torch.testing.assert_close(
            (advanced - x) / 0.125, clean - noise, atol=4e-6, rtol=1e-5
        )

    def test_image_wide_interpolation_matches_repeated_patch_times(self):
        clean, noise = torch.randn(3, 1, 256, 256), torch.randn(3, 1, 256, 256)
        times = torch.tensor([0.0, 0.35, 1.0]).reshape(3, 1, 1)
        torch.testing.assert_close(
            interpolate_patches(clean, noise, times, 16),
            interpolate_patches(clean, noise, times.expand(-1, 16, 16), 16),
            rtol=0,
            atol=0,
        )

    def test_uniform_inference_advances_all_patches_together(self):
        class ConstantVelocity:
            img_size = 32
            patch_size = 16

            def __init__(self):
                self.times = []

            def __call__(self, x, times, patch, extra):
                self.times.append(times.clone())
                return torch.ones_like(x) * 2

        model = ConstantVelocity()
        noise = torch.zeros(2, 1, 32, 32)
        result = euler_sample(model, noise, None, None, steps=4)
        torch.testing.assert_close(result, torch.full_like(noise, 2), rtol=0, atol=0)
        torch.testing.assert_close(noise, torch.zeros_like(noise), rtol=0, atol=0)
        for index, times in enumerate(model.times):
            self.assertEqual(times.shape, (2, 1, 1))
            torch.testing.assert_close(
                times, torch.full_like(times, index / 4), rtol=0, atol=0
            )


class PatchTimeModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        config = small_config("adamw")
        self.model = DiffusionTransformer(**asdict(config.model))
        self.x = torch.randn(2, 1, 256, 256)

    def test_first_block_receives_image_and_hint_tokens(self):
        hint = torch.randn(2, 256, self.model.embed_dim)
        received = []
        handle = self.model.blocks[0].register_forward_pre_hook(
            lambda module, args: received.append(args[0].clone())
        )
        try:
            with torch.no_grad():
                tokens = self.model.patch_embed(self.x).flatten(2).transpose(1, 2)
                for patch in (None, hint):
                    self.model(self.x, torch.zeros(2, 16, 16), patch)
                    self.model(
                        self.x,
                        torch.ones(2, 16, 16),
                        patch,
                        torch.randn(2, 8 * self.model.embed_dim),
                    )
                    expected = tokens if patch is None else tokens + patch
                    for actual in received[-2:]:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            handle.remove()

    def test_every_modulation_receives_local_time_and_shared_text_condition(self):
        times = torch.rand(2, 16, 16)
        count = 6 * len(self.model.blocks) + 2
        extra = torch.randn(2, count, self.model.embed_dim)
        received = []
        modules = [
            *(b.modulation for b in self.model.blocks),
            self.model.final_modulation,
        ]
        handles = [
            module.register_forward_hook(
                lambda module, args, output: received.append((args, output))
            )
            for module in modules
        ]
        try:
            with torch.no_grad():
                self.model(self.x, times, ext_emb=extra.flatten(1))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(received), len(modules))
        offset = 0
        for module, (args, output) in zip(modules, received, strict=True):
            torch.testing.assert_close(args[0], times.flatten(1), rtol=0, atol=0)
            for index, values in enumerate(output):
                self.assertEqual(values.shape, (2, 256, self.model.embed_dim))
                time_module = list(module.time.values())[index]
                expected = (
                    time_module(times.flatten(1)) + extra[:, offset + index, None]
                )
                torch.testing.assert_close(values, expected, rtol=0, atol=0)
            offset += len(module.time)
        self.assertEqual(offset, count)

    def test_each_patch_time_affects_the_prediction(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.05)
            elif isinstance(module, DiMRTimestepParameter):
                torch.nn.init.normal_(module.param0, std=0.05)
                torch.nn.init.normal_(module.param1, std=0.05)
        times = torch.rand(2, 16, 16, requires_grad=True)
        self.model(self.x, times).square().mean().backward()
        self.assertTrue(torch.isfinite(times.grad).all())
        self.assertTrue((times.grad != 0).all())

    def test_model_rejects_a_time_map_with_the_wrong_grid(self):
        for shape in ((2,), (2, 256), (2, 8, 8), (2, 1, 16), (1, 1, 1)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "Timesteps must have shape"):
                    self.model(self.x, torch.zeros(shape))

    def test_image_wide_times_preserve_predictions_and_all_gradients(self):
        for parameter in self.model.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
        times = torch.rand(2, 1, 1)
        extra = torch.randn(2, 8 * self.model.embed_dim)
        upstream = torch.randn_like(self.x)
        results = []
        for image_wide in (False, True):
            self.model.zero_grad(set_to_none=True)
            x = self.x.clone().requires_grad_()
            t = times.clone().requires_grad_()
            condition = extra.clone().requires_grad_()
            output = self.model(
                x,
                t if image_wide else t.expand(-1, 16, 16).contiguous(),
                ext_emb=condition,
            )
            (output * upstream).sum().backward()
            results.append(
                (
                    output.detach(),
                    x.grad,
                    t.grad,
                    condition.grad,
                    *(p.grad.clone() for p in self.model.parameters()),
                )
            )
        for actual, expected in zip(results[1], results[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_model_rejects_text_coefficients_with_the_wrong_layout(self):
        times = torch.rand(2, 16, 16)
        for shape in ((2, 24), (2, 6 * 24), (1, 8 * 24), (2, 8, 24)):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, "Text coefficients"),
            ):
                self.model(self.x, times, ext_emb=torch.zeros(shape))


if __name__ == "__main__":
    unittest.main()
