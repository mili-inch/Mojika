import copy
import unittest

import torch

from mojika.conditional.adamw import CautiousAdamW
from tests.fixtures import new_state


class CautiousAdamWTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)
        self.settings = dict(lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    def test_zero_decay_matches_pytorch_adam_across_shapes_and_learning_rates(self):
        for shape in ((), (7,), (8, 16), (4, 1, 3, 3)):
            with self.subTest(shape=shape):
                actual = torch.nn.Parameter(torch.randn(shape))
                expected = torch.nn.Parameter(actual.detach().clone())
                optimizer = CautiousAdamW([actual], compile=False, **self.settings)
                reference = torch.optim.AdamW(
                    [expected], **self.settings, foreach=False
                )
                for step in range(12):
                    grad = torch.randn(shape)
                    actual.grad, expected.grad = grad.clone(), grad.clone()
                    optimizer.param_groups[0]["lr"] = reference.param_groups[0][
                        "lr"
                    ] = 0.02 / (step + 1)
                    optimizer.step()
                    reference.step()
                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
                    for key in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            optimizer.state[actual][key], reference.state[expected][key]
                        )

    def test_first_update_decays_only_aligned_coordinates(self):
        initial = torch.tensor([2.0, -2.0, 2.0, -2.0, 0.0])
        grad = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0])
        parameter = torch.nn.Parameter(initial.clone())
        parameter.grad = grad
        optimizer = CautiousAdamW(
            [parameter], compile=False, **(self.settings | {"weight_decay": 0.5})
        )
        optimizer.step()
        expected = torch.tensor([1.98, -1.98, 2.01, -2.01, -0.01])
        torch.testing.assert_close(parameter, expected)

    def test_decay_uses_adam_momentum_and_the_weight_before_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0, 1.0, -1.0]))
        reference_parameter = torch.nn.Parameter(parameter.detach().clone())
        optimizer = CautiousAdamW([parameter], compile=False, **self.settings)
        reference = torch.optim.AdamW(
            [reference_parameter], **self.settings, foreach=False
        )
        first_grad = torch.tensor([10.0, 10.0, -10.0, -10.0])
        parameter.grad = first_grad.clone()
        reference_parameter.grad = first_grad.clone()
        optimizer.step()
        reference.step()
        before = parameter.detach().clone()
        optimizer.param_groups[0]["weight_decay"] = 0.5
        grad = torch.tensor([-0.1, 0.1, -0.1, 0.1])
        parameter.grad = grad.clone()
        reference_parameter.grad = grad.clone()
        reference.step()
        direction = before - reference_parameter.detach()
        mask = direction * before >= 0
        self.assertFalse(torch.equal(mask, grad * before >= 0))
        expected = reference_parameter.detach() - 0.01 * 0.5 * before * mask
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)

    def test_missing_gradient_skips_decay_moments_and_step(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = CautiousAdamW(
            [parameter], compile=False, **(self.settings | {"weight_decay": 0.5})
        )
        optimizer.step()
        self.assertEqual(len(optimizer.state), 0)
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        before = parameter.detach().clone()
        saved = copy.deepcopy(optimizer.state[parameter])
        parameter.grad = None
        optimizer.step()
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        self.assertEqual(optimizer.state[parameter]["step"], saved["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[parameter][key], saved[key], rtol=0, atol=0
            )

    def test_zero_update_has_finite_decay(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        parameter.grad = torch.zeros_like(parameter)
        optimizer = CautiousAdamW(
            [parameter], compile=False, **(self.settings | {"weight_decay": 0.5})
        )
        optimizer.step()
        torch.testing.assert_close(parameter, torch.tensor([0.995, -0.995]))

    def test_every_adamw_group_uses_cautious_adamw(self):
        for algorithm in ("adamw", "muon_adamw"):
            with self.subTest(algorithm=algorithm):
                state = new_state(algorithm)
                optimizer = state.optimizers.items["adamw"]
                self.assertIsInstance(optimizer, CautiousAdamW)
                self.assertEqual(
                    optimizer.param_groups[0]["weight_decay"],
                    state.config.optimizer.weight_decay,
                )
                self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0)


if __name__ == "__main__":
    unittest.main()
