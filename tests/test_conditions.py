import itertools
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict

import torch
from safetensors.torch import load_file, save_file
from torchvision.io import write_png

from mojika.conditional.checkpoint import load_models
from mojika.conditional.conditioner import Conditioner
from mojika.conditional.condition_stats import (
    compute_condition_means,
    main as stats_main,
)
from mojika.conditional.condition_sampling import ConditionSample
from mojika.conditional.dataset import ConditionalKanjiDataset
from mojika.conditional.model import DiffusionTransformer
from mojika.conditional.modules.modulation import DiMRModulation
from mojika.conditional.modules.embedders import normalize_embedding
from mojika.modules.timestep_parameter import DiMRTimestepParameter
from tests.fixtures import new_state, small_config, step


class ConditionTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.config = small_config("adamw")
        self.model_config = asdict(self.config.model)
        self.conditioner_config = dict(embed_dim=24, patch_grid=16, depth=1)
        self.model = DiffusionTransformer(**self.model_config)
        # Exercise conditioning through a nonzero denoising head.
        torch.nn.init.normal_(self.model.proj_out.weight, std=0.03)
        self.conditioner = Conditioner(
            torch.randn(1280), torch.randn(512), **self.conditioner_config
        )
        self.x = torch.randn(2, 1, 256, 256)
        self.t = torch.rand(2, 16, 16)
        self.conditions = dict(
            gpt=torch.randn(2, 1280),
            clip=torch.randn(2, 512),
            mask=torch.randn(2, 16, 16),
        )

    def test_initial_conditioner_preserves_base_prediction_for_every_combination(self):
        with torch.no_grad():
            expected = self.model(self.x, self.t)
            for gpt_on, clip_on, mask_on in itertools.product((False, True), repeat=3):
                conditions = {
                    key: self.conditions[key] if enabled else None
                    for key, enabled in (
                        ("gpt", gpt_on),
                        ("clip", clip_on),
                        ("mask", mask_on),
                    )
                }
                patch, extra = self.conditioner(self.x, **conditions)
                actual = self.model(self.x, self.t, patch, extra)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_conditions_preserve_bfloat16_prediction(self):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = self.model(self.x, self.t)
            patch, extra = self.conditioner(self.x, **self.conditions)
            actual = self.model(self.x, self.t, patch, extra)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_text_conditions_are_independent_additive_projections(self):
        for projection in self.conditioner.modulation_projections:
            torch.nn.init.normal_(projection.weight, std=0.03)
        _, unconditional = self.conditioner(self.x)
        _, gpt = self.conditioner(self.x, gpt=self.conditions["gpt"])
        _, clip = self.conditioner(self.x, clip=self.conditions["clip"])
        _, combined = self.conditioner(
            self.x, gpt=self.conditions["gpt"], clip=self.conditions["clip"]
        )
        torch.testing.assert_close(
            combined - unconditional, gpt + clip - 2 * unconditional
        )
        self.assertEqual(torch.count_nonzero(unconditional).item(), 0)
        for embedder, inputs in (
            (self.conditioner.gpt_embedder, self.conditions["gpt"]),
            (self.conditioner.clip_embedder, self.conditions["clip"]),
        ):
            torch.testing.assert_close(
                embedder(inputs),
                embedder.linear(normalize_embedding(inputs) - embedder.mean),
            )
        self.assertGreater((gpt - unconditional).abs().sum().item(), 0)
        self.assertGreater((clip - unconditional).abs().sum().item(), 0)

    def test_condition_dropout_is_zero_after_centering_in_mixed_batches(self):
        for projection in self.conditioner.modulation_projections:
            torch.nn.init.normal_(projection.weight, std=0.03)
        modes = torch.arange(8)
        gpt = torch.randn(8, 1280, requires_grad=True)
        clip = torch.randn(8, 512, requires_grad=True)
        x = self.x[:1].expand(8, -1, -1, -1)
        for dtype in (torch.float32, torch.bfloat16):
            with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
                _, actual = self.conditioner(
                    x, gpt=gpt, clip=clip, condition_mode=modes
                )
                expected = torch.cat(
                    [
                        self.conditioner(
                            x[i : i + 1],
                            gpt=gpt[i : i + 1] if mode & 4 else None,
                            clip=clip[i : i + 1] if mode & 2 else None,
                        )[1]
                        for i, mode in enumerate(range(8))
                    ]
                )
                _, absent = self.conditioner(x)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(torch.count_nonzero(actual[:2]).item(), 0)
            self.assertEqual(torch.count_nonzero(absent).item(), 0)
        _, actual = self.conditioner(x, gpt=gpt, clip=clip, condition_mode=modes)
        actual.sum().backward()
        for name, values, bit in (("gpt", gpt, 4), ("clip", clip, 2)):
            embedder = getattr(self.conditioner, f"{name}_embedder")
            enabled = (modes & bit).bool()
            self.assertEqual(torch.count_nonzero(values.grad[~enabled]).item(), 0)
            self.assertTrue((values.grad[enabled].abs().sum(-1) > 0).all())
            self.assertFalse(embedder.mean.requires_grad)

    def test_projection_preserves_small_centered_features(self):
        gpt, clip = self.conditions["gpt"], self.conditions["clip"]
        for projection in self.conditioner.modulation_projections:
            torch.nn.init.normal_(projection.weight, std=0.03)
        gpt = self.conditioner.gpt_embedder.prepare(gpt)
        clip = self.conditioner.clip_embedder.prepare(clip)
        mask = self.conditions["mask"]
        _, full = self.conditioner.project_centered(mask, gpt, clip)
        _, small = self.conditioner.project_centered(mask, gpt * 1e-4, clip * 1e-4)
        torch.testing.assert_close(small, full * 1e-4, atol=1e-9, rtol=1e-4)

    def test_each_condition_branch_receives_learning_signal(self):
        for parameter in (
            self.conditioner.mask_embedder.mask_gate,
            *(p.weight for p in self.conditioner.modulation_projections),
        ):
            torch.nn.init.normal_(parameter, std=0.03)
        patch, extra = self.conditioner(self.x, **self.conditions)
        self.model(self.x, self.t, patch, extra).square().mean().backward()
        for name in (
            "gpt_embedder.linear.weight",
            "clip_embedder.linear.weight",
            "modulation_projections.0.weight",
            "modulation_projections.1.weight",
            "mask_embedder.mask_gate",
        ):
            grad = dict(self.conditioner.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_layer_input_and_output_factors_only_affect_their_own_coefficients(self):
        conditioner = Conditioner(torch.randn(1280), torch.randn(512), 24, 16, 2)
        for projection in conditioner.modulation_projections:
            torch.nn.init.normal_(projection.weight, std=0.03)
        with torch.no_grad():
            for layer, projection in enumerate(conditioner.modulation_projections):
                rows = slice(layer * 256, (layer + 1) * 256)
                for factor in (
                    conditioner.gpt_embedder.linear.weight[rows],
                    conditioner.clip_embedder.linear.weight[rows],
                    projection.weight,
                ):
                    before = conditioner(self.x, **self.conditions)[1]
                    factor.mul_(2)
                    after = conditioner(self.x, **self.conditions)[1]
                    for index, (previous, actual) in enumerate(
                        zip(
                            before.split((144, 144, 48), dim=-1),
                            after.split((144, 144, 48), dim=-1),
                            strict=True,
                        )
                    ):
                        if index == layer:
                            self.assertGreater(
                                (actual - previous).abs().sum().item(), 0
                            )
                        else:
                            torch.testing.assert_close(actual, previous, rtol=0, atol=0)

    def test_zero_output_factors_learn_before_input_factors(self):
        optimizer = torch.optim.SGD(self.conditioner.parameters(), lr=0.01)
        for update in range(2):
            optimizer.zero_grad()
            extra = self.conditioner(self.x, **self.conditions)[1]
            if update == 0:
                self.assertEqual(torch.count_nonzero(extra).item(), 0)
            extra.sum().backward()
            for projection in self.conditioner.modulation_projections:
                self.assertTrue(torch.isfinite(projection.weight.grad).all())
                self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
            for embedder in (
                self.conditioner.gpt_embedder,
                self.conditioner.clip_embedder,
            ):
                for grad in embedder.linear.weight.grad.split(256):
                    self.assertTrue(torch.isfinite(grad).all())
                    if update == 0:
                        self.assertEqual(torch.count_nonzero(grad).item(), 0)
                    else:
                        self.assertGreater(grad.abs().sum().item(), 0)
            optimizer.step()

    def test_production_conditioner_uses_rank_256_parameter_budget(self):
        conditioner = Conditioner(torch.zeros(1280), torch.zeros(512), 768, 16, 12)
        self.assertEqual(sum(p.numel() for p in conditioner.parameters()), 20_514_304)
        self.assertTrue(
            all(p.in_features == 256 for p in conditioner.modulation_projections)
        )

    def test_only_dimr_modules_have_bias_parameters(self):
        bias_count = 0
        for root in (self.model, self.conditioner):
            for name, module in root.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    self.assertIsNone(module.bias, name)
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    if parameter_name == "bias":
                        self.assertIsInstance(module, DiMRTimestepParameter, name)
                        self.assertTrue(parameter.requires_grad, name)
                        bias_count += 1
        self.assertEqual(bias_count, 6 * self.config.model.depth + 2)

    def test_scratch_initialization_learns_time_and_rope(self):
        state = new_state("adamw")
        for _ in range(5):
            step(state)
        for prefix, root in (
            ("model", state.model),
            ("conditioner", state.conditioner),
        ):
            for name, parameter in root.named_parameters():
                self.assertIsNotNone(parameter.grad, f"{prefix}.{name}")
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(), f"{prefix}.{name}"
                )
                self.assertGreater(
                    parameter.grad.abs().sum().item(), 0, f"{prefix}.{name}"
                )

    def test_zero_gates_open_gradients_to_affine_time_parameters(self):
        state = new_state("adamw")
        for backward in range(1, 4):
            step(state)
            for name, module in state.model.named_modules():
                if not isinstance(module, DiMRTimestepParameter):
                    continue
                for parameter_name, parameter in module.named_parameters():
                    if name.endswith("_gate"):
                        first_backward = (
                            2 if parameter_name in {"param0", "param1"} else 3
                        )
                    else:
                        first_backward = 3 if name.startswith("blocks.") else 2
                    with self.subTest(
                        backward=backward, parameter=f"{name}.{parameter_name}"
                    ):
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        magnitude = parameter.grad.abs().sum().item()
                        if backward < first_backward:
                            self.assertEqual(magnitude, 0)
                        else:
                            self.assertGreater(magnitude, 0)

    def test_every_time_coefficient_has_independent_dimr_parameters(self):
        model = DiffusionTransformer(**(self.model_config | {"depth": 2}))
        modules = [
            module
            for module in model.modules()
            if isinstance(module, DiMRTimestepParameter)
        ]
        self.assertEqual(len(modules), 6 * 2 + 2)
        parameters = [p for module in modules for p in module.parameters()]
        self.assertEqual(len({id(p) for p in parameters}), 4 * len(modules))
        self.assertEqual(sum(p.numel() for p in parameters), (6 * 2 + 2) * 4 * 24)
        with torch.no_grad():
            before = [module(self.t.flatten(1)) for module in modules]
            modules[0].param0.fill_(1)
            after = [module(self.t.flatten(1)) for module in modules]
        self.assertFalse(torch.equal(after[0], before[0]))
        for actual, expected in zip(after[1:], before[1:], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dimr_keeps_positive_time_slopes_and_trainable_bias(self):
        embedder = self.model.final_modulation.time["scale"]
        with torch.no_grad():
            embedder.param0.fill_(1)
            embedder.param1.zero_()
            embedder.raw_slope.copy_(torch.tensor([-8.0, -2.0, 0.0, 4.0]).repeat(6))
            embedder.bias.copy_(torch.linspace(-1, 1, 24))
        times = torch.linspace(0, 1, 33)
        values = embedder(times)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((values[1:] > values[:-1]).all())
        self.assertTrue(((values > 0) & (values < 1)).all())
        torch.testing.assert_close(values[0], torch.sigmoid(torch.linspace(-1, 1, 24)))
        values.sum().backward()
        for parameter in (embedder.raw_slope, embedder.bias):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue((parameter.grad > 0).all())

    def test_time_gpt_and_clip_have_no_mixed_effect_on_any_modulation_coefficient(self):
        model = DiffusionTransformer(**(self.model_config | {"depth": 2}))
        conditioner = Conditioner(torch.randn(1280), torch.randn(512), 24, 16, 2)
        for module in model.modules():
            if isinstance(module, DiMRTimestepParameter):
                for parameter in module.parameters():
                    torch.nn.init.normal_(parameter, std=0.2)
        for projection in conditioner.modulation_projections:
            torch.nn.init.normal_(projection.weight, std=0.03)
        received = []
        handles = [
            module.register_forward_hook(
                lambda module, args, output: received.append(torch.stack(output, dim=2))
            )
            for module in model.modules()
            if isinstance(module, DiMRModulation)
        ]
        times = (torch.rand_like(self.t), torch.rand_like(self.t))
        gpts = (torch.randn(2, 1280), torch.randn(2, 1280))
        clips = (torch.randn(2, 512), torch.randn(2, 512))
        values = {}
        try:
            with torch.no_grad():
                for ti, gi, ci in itertools.product((0, 1), repeat=3):
                    received.clear()
                    patch, extra = conditioner(self.x, gpt=gpts[gi], clip=clips[ci])
                    model(self.x, times[ti], patch, extra)
                    values[ti, gi, ci] = torch.cat(received, dim=2)
        finally:
            for handle in handles:
                handle.remove()
        baseline = values[0, 0, 0]
        self.assertEqual(baseline.shape, (2, 256, 14, 24))
        for index in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            self.assertTrue(((values[index] - baseline).abs().sum((0, 1, 3)) > 0).all())
        for ti, gi, ci in itertools.product((0, 1), repeat=3):
            expected = (
                values[ti, 0, 0] + values[0, gi, 0] + values[0, 0, ci] - 2 * baseline
            )
            torch.testing.assert_close(
                values[ti, gi, ci], expected, rtol=1e-5, atol=1e-6
            )

    def test_rope_rotation_stays_positive_with_negative_raw_frequencies(self):
        rope = self.model.blocks[0].attn.rope
        self.assertTrue(torch.isfinite(rope.raw_freqs).all())
        with torch.no_grad():
            rope.raw_freqs.fill_(-2)
        query = torch.zeros(1, 2, 3, 8)
        query[..., ::2] = 1
        positions = torch.eye(2)
        rotated, _ = rope(query, query, positions)
        self.assertTrue((rotated[..., 1::2] > 0).all())
        torch.testing.assert_close(rotated.norm(dim=-1), query.norm(dim=-1))
        rotated[..., 1::2].sum().backward()
        self.assertTrue(torch.isfinite(rope.raw_freqs.grad).all())
        self.assertTrue((rope.raw_freqs.grad > 0).all())

    def test_rope_preserves_norms_and_has_trainable_frequencies(self):
        rope = self.model.blocks[0].attn.rope
        query = torch.randn(2, 256, 3, 8)
        key = torch.randn_like(query)
        rotated, _ = rope(query, key, self.model.positions)
        torch.testing.assert_close(
            rotated.norm(dim=-1), query.norm(dim=-1), rtol=1e-6, atol=1e-6
        )
        (rotated * torch.randn_like(rotated)).sum().backward()
        self.assertGreater(rope.raw_freqs.grad.abs().sum().item(), 0)

    def test_incomplete_model_checkpoint_is_rejected(self):
        weights = self.model.state_dict()
        del weights["blocks.0.attn.rope.raw_freqs"]
        with self.assertRaisesRegex(RuntimeError, "rope.raw_freqs"):
            load_models({"config": self.config.to_dict(), "model_state_dict": weights})


class ConditionalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in (
            "images",
            "sarashina_embeddings",
            "clip_embeddings",
        ):
            (self.root / directory).mkdir(parents=True)
        image = torch.full((1, 256, 256), 255, dtype=torch.uint8)
        image[:, 32:220, 70:150] = 0
        write_png(image, str(self.root / "images/U+4E00.png"))
        save_file(
            {"embedding": torch.ones(1280)},
            self.root / "sarashina_embeddings/U+4E00_0.safetensors",
        )
        save_file(
            {"embedding": torch.ones(1, 512)},
            self.root / "clip_embeddings/U+4E00_0.safetensors",
        )
        save_file(
            compute_condition_means(self.root),
            self.root / "condition_means.safetensors",
        )

    def test_missing_condition_fails_before_training(self):
        (self.root / "clip_embeddings/U+4E00_0.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "Missing clip_embeddings"):
            ConditionalKanjiDataset(self.root, seed=42)

    def test_validation_catches_corrupt_embedding_variant(self):
        save_file(
            {"embedding": torch.full((1280,), float("nan"))},
            self.root / "sarashina_embeddings/U+4E00_1.safetensors",
        )
        with self.assertRaisesRegex(ValueError, "Invalid embedding"):
            ConditionalKanjiDataset(self.root, seed=42).validate_all()

    def test_normalization_and_augmentation_survive_resume(self):
        first = ConditionalKanjiDataset(self.root, seed=42)
        second = ConditionalKanjiDataset(self.root, seed=42)
        first.epoch = second.epoch = 3
        sample = ConditionSample(0, 7, 0)
        before, after = first[sample], second[sample]
        self.assertEqual(before["image"].min().item(), -1)
        self.assertEqual(before["image"].max().item(), 1)
        for key in ("image", "mask", "gpt", "clip"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_disabled_inputs_are_unconditional_and_skip_embedding_reads(self):
        from unittest.mock import patch

        dataset = ConditionalKanjiDataset(self.root, seed=42)
        for mode in range(8):
            sample = dataset[ConditionSample(0, mode, 0)]
            self.assertEqual(sample["condition_mode"], mode)
            if not mode & 4:
                torch.testing.assert_close(sample["gpt"], torch.zeros(1280))
            if not mode & 2:
                torch.testing.assert_close(sample["clip"], torch.zeros(512))
            if not mode & 1:
                torch.testing.assert_close(sample["mask"], torch.ones(16, 16))
        with (
            patch(
                "mojika.conditional.dataset.read_embedding", side_effect=AssertionError
            ),
            patch(
                "mojika.conditional.dataset.partial_mask", side_effect=AssertionError
            ),
        ):
            dataset[ConditionSample(0, 0, 0)]

    def test_statistics_normalize_before_averaging_and_weight_characters_equally(self):
        generator = torch.Generator().manual_seed(93)
        image = self.root / "images/U+4E00.png"
        (image.parent / "U+4E01.png").write_bytes(image.read_bytes())
        expected = {}
        features = {}
        for name, directory, shape in (
            ("gpt", "sarashina_embeddings", (1280,)),
            ("clip", "clip_embeddings", (1, 512)),
        ):
            values = [
                torch.randn(shape, generator=generator) * scale
                for scale in (1, 4, 0.01)
            ]
            for code, index, value in zip(
                ("U+4E00", "U+4E01", "U+4E01"), (0, 0, 1), values, strict=True
            ):
                save_file(
                    {"embedding": value},
                    self.root / directory / f"{code}_{index}.safetensors",
                )
            features[name] = torch.stack(
                [normalize_embedding(v.flatten()) for v in values]
            )
            expected[name] = features[name][0] / 2 + features[name][1:].mean(0) / 2
        (self.root / "condition_means.safetensors").unlink()
        stats_main(["--dataset-root", str(self.root)])
        actual = load_file(str(self.root / "condition_means.safetensors"))
        dataset = ConditionalKanjiDataset(self.root, seed=42)
        conditioner = Conditioner(dataset.gpt_mean, dataset.clip_mean, 24, 16, 1)
        for name in ("gpt", "clip"):
            torch.testing.assert_close(actual[name], expected[name])
            self.assertFalse(torch.allclose(actual[name], features[name].mean(0)))
            embedder = getattr(conditioner, f"{name}_embedder")
            torch.nn.init.normal_(embedder.linear.weight, std=0.03)
            projected = embedder.linear(features[name] - embedder.mean)
            weighted = projected[0] / 2 + projected[1:].mean(0) / 2
            torch.testing.assert_close(
                weighted, torch.zeros_like(weighted), atol=1e-6, rtol=0
            )
        with self.assertRaises(FileExistsError):
            stats_main(["--dataset-root", str(self.root)])

    def test_invalid_or_missing_statistics_are_rejected(self):
        path = self.root / "condition_means.safetensors"
        for value in (
            torch.zeros(511),
            torch.full((512,), float("nan")),
            torch.zeros(512, dtype=torch.int32),
        ):
            save_file({"gpt": torch.zeros(1280), "clip": value}, path)
            with self.assertRaisesRegex(ValueError, "Invalid clip mean"):
                ConditionalKanjiDataset(self.root, seed=42)
        save_file({"gpt": torch.zeros(1280)}, path)
        with self.assertRaisesRegex(ValueError, "exactly gpt and clip"):
            ConditionalKanjiDataset(self.root, seed=42)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            ConditionalKanjiDataset(self.root, seed=42)


if __name__ == "__main__":
    unittest.main()
