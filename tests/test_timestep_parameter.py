import unittest

import torch

from mojika.conditional.model import DiffusionTransformer
from mojika.modules.timestep_parameter import DiMRTimestepParameter


class TimestepInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_affine_endpoints_are_small_independent_and_learn_time_immediately(self):
        module = DiMRTimestepParameter(4096, init_std=0.01)
        for endpoint in (module.param0, module.param1):
            self.assertLess(endpoint.mean().abs().item(), 5e-4)
            self.assertAlmostEqual(endpoint.std().item(), 0.01, delta=5e-4)
        correlation = torch.corrcoef(torch.stack((module.param0, module.param1)))
        self.assertLess(correlation[0, 1].abs().item(), 0.05)

        values = module(torch.tensor([0.0, 0.5, 1.0]))
        self.assertTrue((values[0] != values[-1]).all())
        torch.testing.assert_close(values[1], (module.param0 + module.param1) / 2)
        values.sum().backward()
        for name, parameter in module.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertTrue((parameter.grad != 0).all(), name)


    def test_conditional_model_initializes_only_time_gates_to_zero(self):
        models = (DiffusionTransformer(32, 4, 1, 24, 2, 3, 2, 10, True),)
        times = torch.linspace(0, 1, 17)
        for model in models:
            gates = 0
            affines = 0
            for name, module in model.named_modules():
                if not isinstance(module, DiMRTimestepParameter):
                    continue
                with self.subTest(model=type(model).__name__, coefficient=name):
                    torch.testing.assert_close(
                        torch.nn.functional.softplus(module.raw_slope),
                        torch.full((24,), 4.0),
                    )
                    torch.testing.assert_close(module.bias, torch.full((24,), -2.0))
                    values = module(times)
                    if name.endswith("_gate"):
                        gates += 1
                        self.assertEqual(module.param0.count_nonzero().item(), 0)
                        self.assertEqual(module.param1.count_nonzero().item(), 0)
                        self.assertEqual(values.count_nonzero().item(), 0)
                    else:
                        affines += 1
                        self.assertTrue((values[0] != values[-1]).all())
            self.assertEqual((gates, affines), (4, 10))
            self.assertEqual(model.proj_out.weight.count_nonzero().item(), 0)

    def test_conditional_blocks_start_as_identity_and_prediction_as_zero(self):
        model = DiffusionTransformer(32, 4, 1, 24, 2, 3, 2, 10, True)
        tokens = torch.randn(2, 64, 24)
        times = torch.rand(2, 8, 8)
        with torch.no_grad():
            for block in model.blocks:
                actual = block(tokens, times.flatten(1), None, model.positions)
                torch.testing.assert_close(actual, tokens, rtol=0, atol=0)
            prediction = model(torch.randn(2, 1, 32, 32), times)
        self.assertEqual(prediction.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
