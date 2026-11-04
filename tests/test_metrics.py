import json
import unittest
from unittest.mock import patch

import torch

from mojika.conditional.metrics import (
    MetricBuffer,
    image_loss_metrics,
    patch_time_metrics,
)


class MetricBufferTests(unittest.TestCase):
    def check_buffer(self, device):
        pending = MetricBuffer()
        counts = {"111": 0}
        source = torch.tensor(0.0, device=device, requires_grad=True)
        sizes = torch.tensor([16, 256], device=device)
        offsets = torch.tensor([[0, 16], [0, 0]], device=device)
        # Accumulation, including epoch boundaries, must not read any GPU values.
        with (
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("host read")
            ),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host read")),
        ):
            for step in range(1, 21):
                with torch.no_grad():
                    source.fill_(10 if step == 7 else step / 100)
                counts["111"] = step * 16
                pending.append(
                    {
                        "event": "step",
                        "step": step,
                        "loss": source,
                        "counts": counts,
                        "patch_times": patch_time_metrics(
                            torch.ones(2, 2, 2, device=device), sizes, offsets
                        ),
                    }
                )
                if step == 13:
                    pending.append(
                        {
                            "event": "epoch",
                            "step": step,
                            "loss": torch.tensor(
                                100000002.0 / 3, dtype=torch.float64, device=device
                            ),
                        }
                    )
        self.assertFalse(pending.records[0]["loss"].requires_grad)
        with torch.no_grad():
            source.zero_()
        sizes.zero_()
        original_cpu = torch.Tensor.cpu
        transfers = []

        def copy_to_host(tensor):
            transfers.append(tensor.device)
            return original_cpu(tensor)

        with patch.object(torch.Tensor, "cpu", copy_to_host):
            records = pending.drain()
            self.assertEqual(pending.drain(), [])
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].type, device)
        self.assertEqual(len(records), 21)
        steps = [r for r in records if r["event"] == "step"]
        self.assertEqual([r["step"] for r in steps], list(range(1, 21)))
        self.assertEqual(steps[6]["loss"], 10)
        for step, record in enumerate(steps, 1):
            self.assertEqual(record["counts"]["111"], step * 16)
            self.assertAlmostEqual(record["loss"], 10 if step == 7 else step / 100)
            self.assertEqual(record["patch_times"]["region_size_pixels"], [16, 256])
            self.assertIs(type(record["patch_times"]["region_size_pixels"][0]), int)
            self.assertEqual(
                record["patch_times"]["region_offset_pixels_yx"], [[0, 16], [0, 0]]
            )
        self.assertEqual(records[13]["event"], "epoch")
        self.assertEqual(records[13]["loss"], 100000002.0 / 3)
        json.dumps(records, allow_nan=False)
        pending.append(
            {"event": "step", "step": 21, "loss": torch.tensor(0.25, device=device)}
        )
        self.assertEqual(pending.drain(), [{"event": "step", "step": 21, "loss": 0.25}])

    def test_cpu_buffer_preserves_every_loss_and_epoch_order(self):
        self.check_buffer("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_buffer_transfers_all_twenty_losses_once(self):
        self.check_buffer("cuda")

    def test_image_diagnostics_are_detached_until_drain(self):
        prediction = torch.zeros(16, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[-1] = 20
        modes = torch.arange(8).repeat(2)
        pending = MetricBuffer()
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host read")):
            pending.append(image_loss_metrics(prediction, target, modes))
        record = pending.drain()[0]
        self.assertEqual(record["image_loss_max"], 400)
        self.assertEqual(record["condition_mean_losses"]["111"], 200)
        self.assertEqual(record["condition_mean_losses"]["000"], 0)
        self.assertIsNone(prediction.grad)


if __name__ == "__main__":
    unittest.main()
