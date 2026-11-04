import unittest

import torch

from mojika.conditional.conditioner import Conditioner
from mojika.conditional.model import DiffusionTransformer
from mojika.conditional.stream import AB2Stream


class Velocity:
    img_size = 4
    patch_size = 2

    def __init__(self):
        self.calls = 0

    def __call__(self, x, t, patch, ext):
        self.calls += 1
        assert t.shape == (x.size(0), 1, 1)
        return x * 0.15 + t[:, :1, :1, None] + ext[:, :1, None, None]


def reference_sample(model, noise, patch, ext, steps):
    x = noise.clone()
    history = []
    for index in range(steps):
        t = torch.full((1, 1, 1), index / steps)
        history.append(model(x, t, patch, ext))
        derivative = history[-1] if index == 0 else (3 * history[-1] - history[-2]) / 2
        x = x + derivative / steps
    return x[0]


class StreamTest(unittest.TestCase):
    def test_dit_stream_preserves_each_trajectorys_modulation_coefficients(self):
        torch.set_num_threads(2)
        torch.manual_seed(83)
        model = DiffusionTransformer(32, 16, 1, 24, 2, 3, 2, 10, True).eval()
        conditioner = Conditioner(torch.randn(1280), torch.randn(512), 24, 2, 2).eval()
        for root in (model, conditioner):
            for parameter in root.parameters():
                torch.nn.init.normal_(parameter, std=0.03)
        steps = 3
        with torch.inference_mode():
            noise = torch.randn(1, 1, 32, 32)
            patch, extra = conditioner(noise)
            self.assertEqual(extra.shape, (1, (6 * 2 + 2) * 24))
            stream = AB2Stream(model, noise, patch, extra, steps)
            expected = [reference_sample(model, noise, patch, extra, steps)]
            for index in range(9):
                noise = torch.randn_like(noise)
                patch, extra = conditioner(
                    noise,
                    gpt=torch.randn(1, 1280),
                    clip=torch.randn(1, 512),
                    mask=torch.randn(1, 2, 2),
                )
                image = stream.step(noise, patch, extra)
                expected.append(reference_sample(model, noise, patch, extra, steps))
                if image is not None:
                    torch.testing.assert_close(image, expected[index - steps + 1])

    def test_completed_lanes_match_sequential_ab2_with_independent_history_and_conditions(
        self,
    ):
        steps = 5
        model, reference = Velocity(), Velocity()
        noise = torch.full((1, 1, 4, 4), -0.5)
        patch, ext = torch.zeros(1, 4, 1), torch.ones(1, 1)
        stream = AB2Stream(model, noise, patch, ext, steps)
        expected = [reference_sample(reference, noise, patch, ext, steps)]
        for index in range(35):
            noise = torch.full_like(noise, index / 10)
            ext = torch.full_like(ext, index / 3)
            image = stream.step(noise, patch, ext)
            self.assertEqual(model.calls, index + 1)
            expected.append(reference_sample(reference, noise, patch, ext, steps))
            if index < steps - 1:
                self.assertIsNone(image)
            else:
                torch.testing.assert_close(image, expected[index - steps + 1])

    def test_single_step_is_a_complete_trajectory(self):
        model = Velocity()
        noise = torch.zeros(1, 1, 4, 4)
        patch, ext = torch.zeros(1, 4, 1), torch.ones(1, 1)
        stream = AB2Stream(model, noise, patch, ext, 1)
        for _ in range(3):
            torch.testing.assert_close(
                stream.step(noise, patch, ext), torch.ones(1, 4, 4)
            )

    def test_ab2_converges_at_second_order_with_one_evaluation_per_step(self):
        class TimeVelocity(Velocity):
            def __call__(self, x, t, patch, ext):
                self.calls += 1
                return t[:, :1, :1, None].expand_as(x)

        errors = []
        noise = torch.zeros(1, 1, 4, 4)
        patch, ext = torch.zeros(1, 4, 1), torch.zeros(1, 1)
        # dx/dt = t, x(0) = 0 has x(1) = 1/2.
        for steps in (8, 16, 32):
            model = TimeVelocity()
            stream = AB2Stream(model, noise, patch, ext, steps)
            for _ in range(steps):
                image = stream.step(noise, patch, ext)
            self.assertEqual(model.calls, steps)
            errors.append(float((image - 0.5).abs().max()))
        for coarse, fine in zip(errors, errors[1:]):
            self.assertAlmostEqual(coarse / fine, 4, places=5)

    def test_invalid_step_counts_fail_explicitly(self):
        noise = torch.zeros(1, 1, 4, 4)
        patch, ext = torch.zeros(1, 4, 1), torch.zeros(1, 1)
        for steps in (0, -1, 2.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                AB2Stream(Velocity(), noise, patch, ext, steps)


if __name__ == "__main__":
    unittest.main()
