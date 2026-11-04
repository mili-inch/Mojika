import copy
import math
import unittest
from dataclasses import replace

import torch

from mojika.conditional.muon import Muon
from mojika.conditional.optimizers import muon_hyperparameters
from mojika.conditional.schedule import WSD
from tests.fixtures import new_state, small_config


class MuonUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def optimizer(self, parameters, weight_decay=0):
        return Muon(
            parameters,
            lr=0.02,
            momentum=0.95,
            beta2=0.9,
            weight_decay=weight_decay,
            compile=False,
        )

    def test_sparse_update_norm_tracks_active_rank(self):
        for shape in ((32, 64), (64, 32), (32, 32)):
            for rank in (1, 2, 8, 32):
                for scale in (0.01, 1, 100):
                    with self.subTest(shape=shape, rank=rank, scale=scale):
                        parameter = torch.nn.Parameter(torch.zeros(shape))
                        gradient = torch.zeros(shape)
                        gradient.diagonal()[:rank] = scale
                        parameter.grad = gradient
                        self.optimizer([parameter]).step()
                        self.assertTrue(torch.isfinite(parameter).all())
                        expected = 0.02 * math.sqrt(max(1, shape[0] / shape[1]) * rank)
                        self.assertLess(parameter.norm().item(), expected * 1.25)
                        self.assertGreater(parameter.norm().item(), expected * 0.5)
                        self.assertEqual(torch.count_nonzero(parameter).item(), rank)

    def test_zero_gradient_is_finite_and_does_not_create_a_direction(self):
        parameter = torch.nn.Parameter(torch.ones(8, 16))
        parameter.grad = torch.zeros_like(parameter)
        optimizer = self.optimizer([parameter])
        optimizer.step()
        torch.testing.assert_close(
            parameter, torch.ones_like(parameter), rtol=0, atol=0
        )
        for value in optimizer.state[parameter].values():
            self.assertTrue(torch.isfinite(value).all())

    def test_missing_gradient_preserves_parameter_and_its_moments(self):
        parameter = torch.nn.Parameter(torch.randn(8, 16))
        optimizer = self.optimizer([parameter], weight_decay=0.1)
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        previous = parameter.detach().clone()
        state = copy.deepcopy(optimizer.state[parameter])
        parameter.grad = None
        optimizer.step()
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
        for name in state:
            torch.testing.assert_close(
                optimizer.state[parameter][name], state[name], rtol=0, atol=0
            )

    def test_cautious_decay_only_shrinks_aligned_coordinates(self):
        for shape in ((8, 16), (16, 8)):
            initial = torch.randn(shape)
            grad = torch.randn(shape)
            plain = torch.nn.Parameter(initial.clone())
            decayed = torch.nn.Parameter(initial.clone())
            plain.grad = grad.clone()
            decayed.grad = grad.clone()
            self.optimizer([plain]).step()
            self.optimizer([decayed], weight_decay=0.1).step()
            direction = initial - plain.detach()
            mask = direction * initial >= 0
            self.assertTrue(mask.any() and (~mask).any())
            lr = 0.02 * math.sqrt(max(1, shape[0] / shape[1]))
            expected = plain.detach() - lr * 0.1 * initial * mask
            torch.testing.assert_close(decayed, expected, rtol=1e-6, atol=2e-7)

    def test_state_belongs_to_each_matrix(self):
        parameters = [torch.nn.Parameter(torch.randn(8, 16)) for _ in range(2)]
        optimizer = self.optimizer(parameters)
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        self.assertEqual(len(optimizer.state), 2)
        for parameter in parameters:
            self.assertEqual(
                optimizer.state[parameter]["momentum_buffer"].shape, (8, 16)
            )
            self.assertEqual(
                optimizer.state[parameter]["variance_buffer"].shape, (1, 16)
            )

    def test_batched_updates_preserve_each_matrix_independently(self):
        shapes = [(8, 16), (8, 16), (16, 8), (16, 8), (8, 8)]
        grouped = [torch.nn.Parameter(torch.randn(shape)) for shape in shapes]
        separate = [torch.nn.Parameter(p.detach().clone()) for p in grouped]
        optimizer = self.optimizer(grouped, weight_decay=0.1)
        references = [self.optimizer([p], weight_decay=0.1) for p in separate]
        for step in range(4):
            for i, (actual, expected) in enumerate(zip(grouped, separate, strict=True)):
                grad = torch.randn_like(actual) * (0.01 if i == 0 else 100)
                actual.grad = expected.grad = None if step == 2 and i == 1 else grad
            for opt in [optimizer, *references]:
                opt.param_groups[0]["lr"] = 0.02 / (step + 1)
                opt.param_groups[0]["momentum"] = 0.85 + step * 0.03
                opt.step()
            for actual, expected, reference in zip(
                grouped, separate, references, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for key, value in optimizer.state[actual].items():
                    torch.testing.assert_close(
                        value, reference.state[expected][key], rtol=0, atol=0
                    )


class MuonScheduleTests(unittest.TestCase):
    def test_momentum_warmup_stable_and_decay_endpoints(self):
        config = small_config("muon_adamw").optimizer
        stable = WSD(2, (config.muon_lr,), None)
        self.assertEqual(
            muon_hyperparameters(config, stable, 0),
            (config.momentum_start, config.muon_weight_decay),
        )
        self.assertAlmostEqual(muon_hyperparameters(config, stable, 2)[0], 0.91)
        for step in (4, 10**12):
            self.assertEqual(
                muon_hyperparameters(config, stable, step),
                (config.momentum, config.muon_weight_decay),
            )
        decay = stable.fork_decay(100, 10, "cosine", 0.01)
        self.assertEqual(
            muon_hyperparameters(config, decay, 100),
            muon_hyperparameters(config, stable, 100),
        )
        self.assertAlmostEqual(muon_hyperparameters(config, decay, 105)[0], 0.935)
        self.assertEqual(
            muon_hyperparameters(config, decay, 110),
            (
                config.momentum_end,
                config.muon_weight_decay * config.decay_weight_decay_ratio,
            ),
        )

    def test_branch_during_momentum_warmup_has_no_jump(self):
        config = small_config("muon_adamw").optimizer
        stable = WSD(2, (config.muon_lr,), None)
        decay = stable.fork_decay(2, 10, "linear", 0)
        self.assertEqual(
            muon_hyperparameters(config, decay, 2),
            muon_hyperparameters(config, stable, 2),
        )

    def test_constant_decay_coefficient_and_no_momentum_warmup(self):
        config = replace(
            small_config("muon_adamw").optimizer,
            decay_weight_decay_ratio=1,
            momentum_warmup_steps=0,
        )
        stable = WSD(0, (config.muon_lr,), None)
        self.assertEqual(muon_hyperparameters(config, stable, 0)[0], config.momentum)
        decay = stable.fork_decay(20, 5, "sqrt", 0)
        for step in range(20, 26):
            self.assertEqual(
                muon_hyperparameters(config, decay, step)[1], config.muon_weight_decay
            )

    def test_special_parameters_do_not_receive_weight_decay(self):
        state = new_state("muon_adamw")
        named = {
            id(p): f"{prefix}.{name}"
            for prefix, module in (
                ("model", state.model),
                ("conditioner", state.conditioner),
            )
            for name, p in module.named_parameters()
        }
        assigned = {
            named[id(p)]: group["weight_decay"]
            for group in state.optimizers.groups()
            for p in group["params"]
        }
        for name in (
            "model.final_modulation.time.scale.param0",
            "model.blocks.0.attn.rope.raw_freqs",
            "model.blocks.0.attn.q_scale",
            "model.final_modulation.time.scale.raw_slope",
            "model.final_modulation.time.scale.bias",
            "conditioner.mask_embedder.mask_gate",
        ):
            self.assertEqual(assigned[name], 0, name)
        for name in (
            "model.patch_embed.weight",
            "conditioner.gpt_embedder.linear.weight",
            "conditioner.clip_embedder.linear.weight",
            "conditioner.modulation_projections.0.weight",
            "conditioner.modulation_projections.1.weight",
        ):
            self.assertEqual(assigned[name], state.config.optimizer.weight_decay)


if __name__ == "__main__":
    unittest.main()
